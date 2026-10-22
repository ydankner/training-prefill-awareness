"""Generate the tampered/clean history turns of the tampering-marker dataset.

One sampled answer per prompt (each prompt is used once), in the model's
non-thinking mode with its recommended instruct sampling (temperature 0.7,
top-p 0.8, top-k 20; no presence penalty -- it would push the sampler away from
the model's own distribution in a way the reader of the turn cannot see).

Half of the answers are tampered with 1-4 planted tokens. Among tampered
answers the target mix is 1 plant 1/2, 2 plants 1/4, 3 plants 1/8, 4 plants 1/8,
and for 2+ plants half are *consecutive* (a run of adjacent planted tokens, each
chosen given the ones before it, then one re-completion) and half *scattered*.
A scattered plant is placed uniformly at random in the first 1/(plants left) of
the answer remaining after the previous plant, measured on the current answer
(each plant is followed by its own re-completion, which changes the length), and
never next to the previous plant -- so there is always room for the rest.

Each prompt is assigned the largest plant count still below its quota. An
attempt that lands only j < k plants is kept as a j-plant record (0 -> clean),
so failures cascade into the larger, later-filled quotas instead of being
discarded. Clean/tampered balance is kept the same way.

A plant is a word-initial whole-word token with probability in [p_min, p_max]
at its position that is also outside the top-``--nucleus`` (0.8) mass of the
temperature-0.7 distribution, i.e. a token the sampler could not have produced
by nucleus sampling. A drawn candidate inside the nucleus is redrawn up to
``--redraws`` times. There is no padding at the start or end of the answer.

Qwen3.8's chat template keeps the (empty) think block of earlier assistant turns,
so a history turn is read after exactly the tokens it was generated after; every
record's ``surprisal_check`` is therefore measured in that one context.

``--thinking low|medium|xhigh`` writes the answers with thinking on (Qwen3.8's
reasoning effort; pass the thinking-mode sampler, e.g. ``--temperature 1.0
--top-p 0.95 --nucleus 0.95``) for the hidden-thinking variant: the reasoning
is kept in the record but never shown again -- the history turn is read with an
empty think block, like an API that drops reasoning from the conversation.
Plants go into the answer only and are chosen, re-completed and checked in the
writer's context (prompt + its reasoning); ``surprisal_check_hidden`` scores the
same turn in the reader's context (no reasoning), the one the detector sees.
Answers whose reasoning does not close within ``--max-think-tokens`` are skipped.

Output is appended and flushed per record; a restarted job skips the prompts it
already wrote. ``--shards/--shard`` split the prompts that are still to do --
not in ``--exclude`` files (e.g. an earlier, interrupted run) -- across
independent jobs; ids stay unique because they carry the prompt's global index.
``--from-index``/``--to-index`` restrict a run to a contiguous block of the
(fixed, shuffled) prompt order. An unsharded run works through the order from
the start, so the rest can be cut into blocks for other jobs: the running job
stops where the first started block begins, and absorbs any block whose job
never started.

    python -m prefill_awareness.tamper_gen --model Qwen/Qwen3.8-27B \
        --prompts data/dolly-brainstorming-prompts.jsonl:brainstorming \
                  data/dolly-rest-prompts.jsonl:rest \
        --out data/tamper-qwen38/histories.jsonl
"""

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data_gen
from .data_gen import (case_variant_outranks, eligible_token, load_prompts, logits_at, split_of,
                       strip_terminators)
from .provenance import describe, require_committed, utc_now
from .surprisal import locate_plant

WORD_ONLY = SimpleNamespace(word_tokens_only=True)

# Fractions of all records. Keys: (n_plants, mode).
TARGETS = {
    (0, "clean"): 0.5,
    (1, "single"): 0.25,
    (2, "consecutive"): 0.0625, (2, "scattered"): 0.0625,
    (3, "consecutive"): 0.03125, (3, "scattered"): 0.03125,
    (4, "consecutive"): 0.03125, (4, "scattered"): 0.03125,
}
MAX_PLANTS = 4


def realized_class(n_plants: int, requested_mode: str) -> tuple[int, str]:
    if n_plants == 0:
        return (0, "clean")
    if n_plants == 1:
        return (1, "single")
    return (n_plants, requested_mode)


def plan_labels(counts: dict, n: int) -> list[tuple[int, str]]:
    """Requested class per prompt, from the balance realised so far."""
    prov = dict(counts)
    plans = []
    for _ in range(n):
        total = sum(prov.values())
        tampered = total - prov[(0, "clean")]
        if total and tampered / total >= 0.5:
            pick = (0, "clean")
        else:
            t = tampered + 1
            pick = (1, "single")
            for k in range(MAX_PLANTS, 1, -1):
                have = prov[(k, "consecutive")] + prov[(k, "scattered")]
                if have < 2 * TARGETS[(k, "consecutive")] / 0.5 * t:
                    mode = "consecutive" if prov[(k, "consecutive")] <= prov[(k, "scattered")] else "scattered"
                    pick = (k, mode)
                    break
        plans.append(pick)
        prov[pick] += 1
    return plans


def sampler_kwargs(args) -> dict:
    # top_k and min_p are always explicit, so the model's generation_config
    # cannot silently add its own.
    return {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "min_p": None}


@torch.no_grad()
def sample(model, tokenizer, sequences, max_new_tokens, args, device):
    """Continue each id sequence with the sampler; returns new tokens per row."""
    if max_new_tokens <= 0:
        return [torch.empty(0, dtype=torch.long, device=device) for _ in sequences]
    pad = tokenizer.pad_token_id
    width = max(int(s.shape[0]) for s in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):  # left padding: rows end at their generation point
        ids[i, width - seq.shape[0]:] = seq
        mask[i, width - seq.shape[0]:] = 1
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                         use_cache=True, pad_token_id=pad, **sampler_kwargs(args))
    return [strip_terminators(row, tokenizer) for row in out[:, width:]]


def nucleus_threshold(probs_t: torch.Tensor, top_p: float) -> float:
    """Smallest tempered probability still inside the top-``top_p`` nucleus."""
    sorted_p, _ = torch.sort(probs_t, descending=True)
    before = torch.cumsum(sorted_p, 0) - sorted_p
    inside = int((before < top_p).sum())
    return float(sorted_p[max(inside, 1) - 1])


def choose_plant(logits, orig_id, taken: set[str], tokenizer, args, rng):
    """Pick a plant for one position, or None.

    Band + word + casing filters as in data_gen, the replacement drawn from the
    ``--candidates`` most probable survivors; a draw inside the tempered
    nucleus is redrawn up to ``--redraws`` times. Returns (id, text, prob, redraws).
    """
    probs = torch.softmax(logits, -1)
    if orig_id is not None and float(probs[orig_id]) > args.max_orig_prob:
        return None
    in_band = ((probs < args.p_max) & (probs > args.p_min)).nonzero(as_tuple=True)[0]
    if in_band.numel() == 0:
        return None
    order = torch.argsort(probs[in_band], descending=True)[: args.scan]
    survivors = []
    for idx in order.tolist():
        token_id = int(in_band[idx])
        if token_id == orig_id:
            continue
        text = tokenizer.decode([token_id])
        if not eligible_token(tokenizer, token_id, text, WORD_ONLY):
            continue
        if text.strip().lower() in taken:
            continue
        prob = float(probs[token_id])
        if case_variant_outranks(probs, text, prob, tokenizer):
            continue
        survivors.append((token_id, text, prob))
        if len(survivors) >= args.candidates:
            break
    if not survivors:
        return None
    probs_t = torch.softmax(logits / args.temperature, -1)
    threshold = nucleus_threshold(probs_t, args.nucleus)
    for redraw in range(args.redraws + 1):
        if not survivors:
            return None
        pick = rng.choice(survivors)
        if float(probs_t[pick[0]]) < threshold:
            return (*pick, redraw)
        survivors.remove(pick)
    return None


def window_positions(tokenizer, resp, lo, hi, args, rng):
    """Eligible (word-initial whole-word) original tokens in [lo, hi), shuffled."""
    order = list(range(max(lo, 0), min(hi, int(resp.shape[0]))))
    rng.shuffle(order)
    chosen = []
    for pos in order:
        token_id = int(resp[pos])
        if eligible_token(tokenizer, token_id, tokenizer.decode([token_id]), WORD_ONLY):
            chosen.append(pos)
            if len(chosen) >= args.positions:
                break
    return chosen


def sample_own_token(logits, args, generator=None):
    """What the model would have written here under its sampler (for bookkeeping)."""
    scaled = logits / args.temperature
    top = torch.topk(scaled, args.top_k)
    probs = torch.softmax(top.values, -1)
    sorted_p, order = torch.sort(probs, descending=True)
    keep = (torch.cumsum(sorted_p, 0) - sorted_p) < args.top_p
    kept = sorted_p * keep
    choice = torch.multinomial(kept / kept.sum(), 1, generator=generator)
    return int(top.indices[order[choice]])


@torch.no_grad()
def plant_rows(model, tokenizer, rows, args, rng, device, stats):
    """Place each row's planned plants, stage by stage, batching across rows."""
    for stage in range(MAX_PLANTS):
        active = [r for r in rows if not r["stopped"] and len(r["plants"]) < r["k"]
                  and (r["mode"] != "consecutive" or stage == 0)]
        if not active:
            break
        # 1. Candidate positions in each row's window.
        index, owner = [], []
        for slot, r in enumerate(active):
            resp = r["resp"]
            start = r["plants"][-1]["token_index"] + 2 if r["plants"] else 0
            end = int(resp.shape[0])
            if r["mode"] == "consecutive":
                hi = end
            else:
                left = r["k"] - len(r["plants"])
                hi = start + math.ceil((end - start) / left)
            r["cands"] = window_positions(tokenizer, resp, start, hi, args, rng)
            if not r["cands"]:
                r["stopped"] = True
                stats["no_position"] += 1
                continue
            for pos in r["cands"]:
                index.append((slot, int(r["prefix"].shape[0]) - 1 + pos))
                owner.append((r, pos))
        if not index:
            break
        sequences = [torch.cat([r["prefix"], r["resp"]]) for r in active]
        logits = logits_at(model, tokenizer, sequences, index, device)
        picked = {}
        for (r, pos), row_logits in zip(owner, logits):
            if id(r) in picked or r["stopped"]:
                continue
            taken = {p["new_token_text"].strip().lower() for p in r["plants"]}
            orig = int(r["resp"][pos])
            choice = choose_plant(row_logits, orig, taken, tokenizer, args, rng)
            if choice is None:
                continue
            picked[id(r)] = (pos, orig, float(torch.softmax(row_logits, -1)[orig]), choice)
        for r in active:
            if not r["stopped"] and id(r) not in picked:
                r["stopped"] = True
                stats["no_candidate"] += 1

        # 2. Consecutive runs: extend with k-1 adjacent plants, one batched step each.
        runs = {}
        for r in active:
            if id(r) not in picked:
                continue
            pos, orig, orig_prob, (new_id, new_text, new_prob, redraws) = picked[id(r)]
            runs[id(r)] = [dict(token_index=pos, orig_token_id=orig,
                                orig_token_text=tokenizer.decode([orig]), orig_prob=orig_prob,
                                new_token_id=new_id, new_token_text=new_text, new_prob=new_prob,
                                surprisal_nats=-math.log(new_prob), redraws=redraws, stage=stage)]
            stats["redraws"] += redraws
        extend = [r for r in active if id(r) in runs and r["mode"] == "consecutive"]
        for _ in range(1, MAX_PLANTS):
            extend = [r for r in extend if len(runs[id(r)]) < r["k"] and not r.get("run_done")]
            if not extend:
                break
            seqs = []
            for r in extend:
                run = runs[id(r)]
                ids = torch.tensor([p["new_token_id"] for p in run], device=device)
                seqs.append(torch.cat([r["prefix"], r["resp"][: run[0]["token_index"]], ids]))
            logits = logits_at(model, tokenizer, seqs, [(i, int(s.shape[0]) - 1) for i, s in enumerate(seqs)],
                               device)
            for r, row_logits in zip(extend, logits):
                run = runs[id(r)]
                probs = torch.softmax(row_logits, -1)
                if float(probs.max()) > args.max_orig_prob:
                    r["run_done"] = True
                    stats["run_certain"] += 1
                    continue
                own = sample_own_token(row_logits, args)
                taken = {p["new_token_text"].strip().lower() for p in r["plants"] + run}
                choice = choose_plant(row_logits, None, taken, tokenizer, args, rng)
                if choice is None or choice[0] == own:
                    r["run_done"] = True
                    stats["run_no_candidate"] += 1
                    continue
                new_id, new_text, new_prob, redraws = choice
                stats["redraws"] += redraws
                run.append(dict(token_index=run[0]["token_index"] + len(run), orig_token_id=own,
                                orig_token_text=tokenizer.decode([own]), orig_prob=float(probs[own]),
                                new_token_id=new_id, new_token_text=new_text, new_prob=new_prob,
                                surprisal_nats=-math.log(new_prob), redraws=redraws, stage=stage))

        # 3. Re-complete every row that got plants in this stage, in one batch.
        done = [r for r in active if id(r) in runs]
        if not done:
            continue
        heads, budgets = [], []
        for r in done:
            run = runs[id(r)]
            start = run[0]["token_index"]
            ids = torch.tensor([p["new_token_id"] for p in run], device=device)
            r["head"] = torch.cat([r["resp"][:start], ids])
            heads.append(torch.cat([r["prefix"], r["head"]]))
            # Only the remaining budget, so a tampered answer is not longer than a clean one.
            budgets.append(args.max_new_tokens - int(r["head"].shape[0]))
        tails = sample(model, tokenizer, heads, max(budgets), args, device)
        for r, tail, budget in zip(done, tails, budgets):
            r["resp"] = torch.cat([r["head"], tail[: max(budget, 0)]])
            r["plants"].extend(runs[id(r)])
            if r["mode"] == "consecutive":
                r["stopped"] = True


@torch.no_grad()
def surprisal_check(model, tokenizer, prefix_ids, response_ids, plant_positions, args, device, chunk=128):
    """Score the finished turn after its generation prefix (= how it is read).

    Per record: every plant's surprisal and rank, whether the plants are exactly
    the k most surprising tokens, recall@k, and how many of the model's own
    tokens fall outside what its sampler could produce (top-k 20 and the top-p
    nucleus at temperature 0.7; ~0 up to bf16 batch effects).
    """
    ids = torch.tensor([prefix_ids + response_ids], device=device)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone, head = getattr(inner, "model", None), getattr(inner, "lm_head", None)
    hidden = backbone(input_ids=ids).last_hidden_state[0] if backbone is not None and head is not None else None
    full = model(ids).logits[0] if hidden is None else None
    start = len(prefix_ids)
    surprisal, outside = [], []
    for lo in range(0, len(response_ids), chunk):
        hi = min(lo + chunk, len(response_ids))
        rows = torch.arange(start + lo - 1, start + hi - 1, device=device)
        logits = head(hidden[rows]).float() if hidden is not None else full[rows].float()
        targets = torch.tensor(response_ids[lo:hi], device=device)
        logp = torch.log_softmax(logits, -1)
        surprisal.extend((-logp[torch.arange(hi - lo, device=device), targets]).tolist())
        tempered = torch.softmax(logits / args.temperature, -1)
        sorted_p, _ = torch.sort(tempered, -1, descending=True)
        before = torch.cumsum(sorted_p, -1) - sorted_p
        n_in = (before < args.top_p).sum(-1).clamp(min=1).clamp(max=args.top_k)
        threshold = sorted_p.gather(-1, (n_in - 1).unsqueeze(-1)).squeeze(-1)
        target_p = tempered[torch.arange(hi - lo, device=device), targets]
        outside.extend((target_p < threshold).tolist())
    plants = set(plant_positions)
    order = sorted(range(len(surprisal)), key=lambda i: -surprisal[i])
    k = len(plants)
    out = {
        "n_tokens": len(surprisal),
        "natural_outside_support": sum(1 for i, o in enumerate(outside) if o and i not in plants),
        "top_natural": [{"index": i, "text": tokenizer.decode([response_ids[i]]), "surprisal": round(surprisal[i], 3)}
                        for i in order if i not in plants][:3],
    }
    if k:
        rank = {i: r + 1 for r, i in enumerate(order)}
        out.update(
            plant_ranks=[rank[i] for i in plant_positions],
            plant_surprisals=[round(surprisal[i], 3) for i in plant_positions],
            plants_outside_support=sum(1 for i in plant_positions if outside[i]),
            exact_topk=set(order[:k]) == plants,
            recall_at_k=len(set(order[:k]) & plants) / k,
        )
    return out


def prompt_ids(tokenizer, user: str, thinking: str, device):
    """Generation prompt for one user turn: non-thinking ("off") or thinking at that effort."""
    mode = ({"enable_thinking": False} if thinking == "off"
            else {"enable_thinking": True, "reasoning_effort": thinking})
    text = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True, **mode)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)


def split_reasoning(ids, think_end_id: int, tokenizer):
    """Index where the answer starts in a thinking-mode output: after ``</think>``
    and the whitespace tokens that follow it. None if the reasoning never closes
    or no answer follows."""
    ends = (ids == think_end_id).nonzero(as_tuple=True)[0]
    if ends.numel() == 0:
        return None
    start = int(ends[0]) + 1
    while start < ids.shape[0] and not tokenizer.decode([int(ids[start])]).strip():
        start += 1
    return start if start < ids.shape[0] else None


def verify_plants(tokenizer, text, plants):
    """Re-tokenize the stored text (as training does) and relocate every plant."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    fixed = []
    for plant in plants:
        where = locate_plant(ids, plant)
        if where is None or where in {p["token_index"] for p in fixed}:
            return ids, None
        fixed.append({**plant, "token_index": where})
    if [p["token_index"] for p in fixed] != sorted(p["token_index"] for p in fixed):
        return ids, None
    return ids, fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tampering-marker history turns.")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="prompt files as path[:category]; category 'rest' applies --min-response-tokens")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=0, help="cap on prompts (0: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-pct", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-response-tokens", type=int, default=100,
                        help="for 'rest' prompts, clean and tampered alike (as the earlier datasets)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--p-min", type=float, default=1e-4)
    parser.add_argument("--p-max", type=float, default=0.01)
    parser.add_argument("--nucleus", type=float, default=0.8,
                        help="a plant must lie outside this top-p mass of the tempered distribution")
    parser.add_argument("--redraws", type=int, default=3)
    parser.add_argument("--max-orig-prob", type=float, default=0.9)
    parser.add_argument("--candidates", type=int, default=16, help="draw from the K most probable survivors")
    parser.add_argument("--scan", type=int, default=256)
    parser.add_argument("--positions", type=int, default=16, help="candidate positions scored per window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--from-index", type=int, default=0,
                        help="only prompts at this position of the prompt order or later")
    parser.add_argument("--to-index", type=int, default=None,
                        help="...and before this position")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="histories files whose prompts are already done (never regenerated)")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all",
                        help="only prompts of this split (split_of with --test-pct)")
    parser.add_argument("--id-prefix", default="t38-")
    parser.add_argument("--plan", choices=("mix", "single"), default="mix",
                        help="single: every prompt gets one plant (eval sets; the clean original is kept)")
    parser.add_argument("--unicode-words", action="store_true",
                        help="whole words in any script may be planted/replaced (default: Latin letters only)")
    parser.add_argument("--thinking", choices=("off", "low", "medium", "xhigh"), default="off",
                        help="write answers with thinking at this reasoning effort; reasoning is hidden on reading")
    parser.add_argument("--max-think-tokens", type=int, default=4096,
                        help="with --thinking: reasoning budget on top of --max-new-tokens")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)
    if args.unicode_words:  # eligible_token reads this module global
        data_gen.WORD_TOKEN = re.compile(r"^[^\W\d_][^\W\d_'-]*$")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map=device)
    model.eval()
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    prompts = []
    for spec in args.prompts:
        path, sep, category = spec.rpartition(":")
        if not sep or not category.isalpha():  # no category, or a drive letter
            path, category = spec, "rest"
        for p in load_prompts(path, 10**9, args.seed):
            prompts.append((p, category))
    seen, unique = set(), []
    for p, c in prompts:
        if p not in seen:
            seen.add(p)
            unique.append((p, c))
    random.Random(args.seed).shuffle(unique)
    prompts = unique[: args.n or None]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {key: 0 for key in TARGETS}
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add(rec["prompt"])
                if rec.get("label"):
                    counts[(rec["n_plants"], rec["mode"])] += 1
    meta_path = out_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        **describe("prefill_awareness.tamper_gen", args, git, model),
        "prompts": {"sources": args.prompts, "n": len(prompts),
                    "sha256": hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()},
        "executions": [],
    }
    meta["executions"].append({"git": git, "started_at": utc_now(), "resumed_with": len(done),
                               "condor_job": describe("x", args, git, model)["condor_job"],
                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    meta["finished_at"] = None
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    excluded = set()
    for path in args.exclude:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    excluded.add(json.loads(line)["prompt"])
                except json.JSONDecodeError:  # a line cut off when that job was stopped
                    pass
    # The shard split is over the prompts not excluded, so it is the same on a restart.
    todo = [(i, p, c) for i, (p, c) in enumerate(prompts)
            if i >= args.from_index and (args.to_index is None or i < args.to_index)
            and p not in excluded][args.shard::args.shards]
    todo = [(i, p, c) for i, p, c in todo if p not in done]
    if args.split != "all":
        todo = [(i, p, c) for i, p, c in todo if split_of(p, args.test_pct) == args.split]
    stats = {k: 0 for k in ("no_position", "no_candidate", "run_certain", "run_no_candidate", "redraws",
                            "skipped_short", "retok_fail")}
    stats.update(meta.get("stats") or {})
    started = time.time()
    with out_path.open("a", encoding="utf-8", newline="\n") as handle:
        for start in tqdm(range(0, len(todo), args.batch_size), desc="histories"):
            chunk = todo[start:start + args.batch_size]
            chunk_seed = int(hashlib.md5(f"{args.seed}:{chunk[0][1]}".encode()).hexdigest()[:8], 16)
            rng = random.Random(chunk_seed)
            torch.manual_seed(chunk_seed)
            plans = plan_labels(counts, len(chunk)) if args.plan == "mix" else [(1, "single")] * len(chunk)
            suffix = f"\n\nKeep your answer under {args.max_words} words." if args.max_words else ""
            # reader_ids: how the turn is read later (non-thinking, empty think block).
            reader_ids = [prompt_ids(tokenizer, p + suffix, "off", device) for _, p, _ in chunk]
            if args.thinking == "off":
                prefix_ids = reader_ids
                responses = sample(model, tokenizer, prefix_ids, args.max_new_tokens, args, device)
                reasoning = [None] * len(chunk)
            else:
                think_ids = [prompt_ids(tokenizer, p + suffix, args.thinking, device) for _, p, _ in chunk]
                outputs = sample(model, tokenizer, think_ids, args.max_think_tokens + args.max_new_tokens,
                                 args, device)
                prefix_ids, responses, reasoning = [], [], []
                for head, out in zip(think_ids, outputs):
                    split = split_reasoning(out, think_end_id, tokenizer)
                    if split is None:
                        prefix_ids.append(None)
                        responses.append(None)
                        reasoning.append(None)
                        continue
                    # The writer's prefix ends where its answer starts (after "</think>\n\n").
                    prefix_ids.append(torch.cat([head, out[:split]]))
                    responses.append(out[split:])
                    reasoning.append(tokenizer.decode(out[:split], skip_special_tokens=True).split("</think>")[0])
            rows = []
            for j, ((_, prompt, category), plan) in enumerate(zip(chunk, plans)):
                minimum = args.min_response_tokens if category == "rest" else 1
                if responses[j] is None:
                    rows.append("no_think_end")
                    continue
                if int(responses[j].shape[0]) < max(minimum, 1):
                    rows.append(None)
                    continue
                rows.append({"prefix": prefix_ids[j], "resp": responses[j], "k": plan[0], "mode": plan[1],
                             "plants": [], "stopped": plan[0] == 0})
            plant_rows(model, tokenizer, [r for r in rows if isinstance(r, dict)], args, rng, device, stats)

            for j, ((index, prompt, category), plan, row) in enumerate(zip(chunk, plans, rows)):
                if not isinstance(row, dict):
                    reason = row or "short"
                    stats[f"skipped_{reason}"] = stats.get(f"skipped_{reason}", 0) + 1
                    handle.write(json.dumps({"prompt": prompt, "label": None, "skipped": reason},
                                            ensure_ascii=False) + "\n")
                    continue
                plants = row["plants"]
                final = row["resp"] if plants else responses[j]
                text = tokenizer.decode(final, skip_special_tokens=True)
                ids, plants_fixed = verify_plants(tokenizer, text, plants)
                if plants and plants_fixed is None:
                    stats["retok_fail"] += 1
                    handle.write(json.dumps({"prompt": prompt, "label": None, "skipped": "retokenization"},
                                            ensure_ascii=False) + "\n")
                    continue
                plants = plants_fixed or []
                n_plants, mode = realized_class(len(plants), plan[1])
                record = {
                    "id": f"{args.id_prefix}{index:06d}",
                    "prompt": prompt,
                    "category": category,
                    "split": split_of(prompt, args.test_pct),
                    "model": args.model,
                    "enable_thinking": False,  # how the turn is read and replied to
                    "history_thinking": args.thinking,
                    "label": "tampered" if plants else "clean",
                    "n_plants": n_plants,
                    "mode": mode,
                    "requested": {"n_plants": plan[0], "mode": plan[1]},
                    "messages": [{"role": "user", "content": prompt + suffix},
                                 {"role": "assistant", "content": text}],
                    "clean_response": tokenizer.decode(responses[j], skip_special_tokens=True),
                    "n_response_tokens": len(ids),
                    "truncated": int(final.shape[0]) >= args.max_new_tokens,
                    "plants": plants,
                    "surprisal_check": surprisal_check(model, tokenizer, prefix_ids[j].tolist(), ids,
                                                       [p["token_index"] for p in plants], args, device),
                }
                if args.thinking != "off":
                    record["reasoning"] = reasoning[j].strip()
                    record["n_reasoning_tokens"] = len(tokenizer.encode(record["reasoning"], add_special_tokens=False))
                    record["surprisal_check_hidden"] = surprisal_check(
                        model, tokenizer, reader_ids[j].tolist(), ids, [p["token_index"] for p in plants],
                        args, device)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                counts[(n_plants, mode)] += 1
            meta["stats"] = stats
            meta["counts"] = {f"{k}-{m}": v for (k, m), v in counts.items()}
            meta["seconds_this_execution"] = round(time.time() - started)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta["finished_at"] = utc_now()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": meta["counts"], "stats": stats}))


if __name__ == "__main__":
    main()
