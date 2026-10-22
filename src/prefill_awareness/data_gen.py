"""Generate the prefill-detection dataset.

For each source prompt we:
  1. sample a clean assistant response from the model;
  2. read off the model's own next-token distribution at a sample of candidate
     positions in that response;
  3. swap one token for a plausible-but-surprising alternative -- the
     highest-probability candidates that still sit below ``--p-max`` (default
     1%) under the model's own distribution;
  4. re-complete the rest of the turn conditioned on the swapped token, so the
     tail stays self-consistent and the modified token is the *only*
     high-surprisal event in the turn;
  5. append the detection question and the gold answer.

Clean (negative) examples skip steps 2-4 and answer ``none``, so the model
cannot win by always claiming tampering.

Everything is batched: prompts are processed ``--batch-size`` at a time, and all
three model passes (sample, score, re-complete) run over the whole chunk. Step 2
never materialises full ``[batch, length, vocab]`` logits -- at a 500-word answer
that tensor is hundreds of gigabytes -- and instead applies the LM head only at
the sampled candidate positions.

Records are written as JSON lines, flushed one at a time, so an interrupted job
keeps everything it already produced. Train/test membership is derived from a
hash of the source prompt, which keeps the split stable across shards and
guarantees a prompt never lands on both sides.
"""

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .formats import DETECT_QUESTION, render_answer
from .provenance import describe, require_committed, utc_now
from .surprisal import turn_surprisal

# A token may be swapped (or serve as a replacement) only if it is a whole word.
# This keeps the answer unambiguous and avoids perturbing whitespace,
# punctuation or markdown scaffolding.
WORD_TOKEN = re.compile(r"^[A-Za-z][A-Za-z'-]*$")

# Word-boundary markers for byte-level BPE and SentencePiece respectively.
BOUNDARY_MARKERS = ("Ġ", "▁")

_SPECIAL_IDS: dict[int, set[int]] = {}


def special_ids(tokenizer) -> set[int]:
    """Cached ``all_special_ids``.

    That attribute is a property which rebuilds a list on every access, and the
    candidate loop touches it thousands of times per example -- enough to
    dominate generation time.
    """
    key = id(tokenizer)
    if key not in _SPECIAL_IDS:
        _SPECIAL_IDS[key] = set(tokenizer.all_special_ids)
    return _SPECIAL_IDS[key]


def is_word_initial(tokenizer, token_id: int, text: str) -> bool:
    """True if this token starts a new word rather than continuing one.

    Decoding a single token preserves the leading space for byte-level BPE
    (Qwen, Llama-3) but drops it for SentencePiece (Gemma), so fall back to
    inspecting the raw token's boundary marker.
    """
    if text.startswith(" "):
        return True
    raw = tokenizer.convert_ids_to_tokens(token_id)
    return isinstance(raw, str) and raw[:1] in BOUNDARY_MARKERS


def eligible_token(tokenizer, token_id: int, text: str, args) -> bool:
    """Whether a token may be swapped out, or serve as a replacement.

    Requiring word-initial tokens rules out mid-word subword swaps such as
    "match|es" -> "match|ings", which mangle a word instead of substituting one.
    """
    if token_id in special_ids(tokenizer):
        return False
    if not text.strip() or '"' in text or "\n" in text:
        return False
    if not args.word_tokens_only:
        return True
    return is_word_initial(tokenizer, token_id, text) and bool(WORD_TOKEN.match(text.strip()))


def case_variant_outranks(probs, text: str, prob: float, tokenizer) -> bool:
    """True if some casing variant of ``text`` is likelier here than ``text``.

    A candidate whose own casing is not the preferred one at this position (say
    "OFF" where "off" ranks higher) can be spotted by its odd capitalisation
    alone, with no reference to how surprising the *word* is. Rejecting those
    keeps surprisal the only signal that marks the plant. This also covers the
    original token: if it is confidently predicted, it outranks its own variants,
    so its case-variants can never be chosen as replacements.
    """
    lead = " " if text.startswith(" ") else ""
    body = text.strip()
    for variant in {body.lower(), body.upper(), body.capitalize()}:
        if variant == body:
            continue
        ids = tokenizer.encode(lead + variant, add_special_tokens=False)
        if len(ids) == 1 and float(probs[ids[0]]) > prob:
            return True
    return False


def split_of(prompt: str, test_pct: int) -> str:
    """Deterministic per-prompt split, independent of shard or ordering."""
    bucket = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if bucket < test_pct else "train"


def load_prompts(spec: str, limit: int, seed: int, categories: set[str] | None = None) -> list[str]:
    """Load source prompts from a local file or a HuggingFace dataset.

    ``spec`` is a path to a .txt (one prompt per line) or .jsonl file, or
    ``hf:<dataset>[:<split>[:<field>]]``. ``categories`` filters on a ``category``
    column when the dataset has one (Dolly, No Robots), which is how the domain
    mix is controlled.
    """

    def from_row(row: dict, field: str | None) -> str | None:
        if categories and str(row.get("category", "")).lower() not in categories:
            return None
        # Rows carrying a passage (Dolly's closed_qa) are excluded: the answer is
        # copied from the passage, so its distribution has no room for a plant.
        if str(row.get("context", "")).strip():
            return None
        if field:
            return row.get(field)
        for key in ("prompt", "instruction"):
            if isinstance(row.get(key), str):
                return row[key]
        messages = row.get("messages") or row.get("conversations")
        if messages:
            return messages[0].get("content") or messages[0].get("value")
        return None

    prompts: list[str] = []
    if spec.startswith("hf:"):
        from datasets import load_dataset

        parts = spec[3:].split(":")
        split = parts[1] if len(parts) > 1 else "train"
        field = parts[2] if len(parts) > 2 else None
        # Streaming keeps this cheap on a login node and avoids materialising a
        # dataset far larger than the slice we want.
        for row in load_dataset(parts[0], split=split, streaming=True):
            text = from_row(row, field)
            if text:
                prompts.append(text)
            if len(prompts) >= limit * 4:
                break
    else:
        path = Path(spec)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            text = from_row(json.loads(line), None) if path.suffix == ".jsonl" else line
            if text:
                prompts.append(text)

    prompts = [p.strip() for p in prompts if 16 <= len(p.strip()) <= 2000]
    # Dedupe while keeping order, then shuffle so a shard is not one topic block.
    prompts = list(dict.fromkeys(prompts))
    random.Random(seed).shuffle(prompts)
    return prompts[:limit]


def strip_terminators(ids, tokenizer):
    """Drop trailing EOS / pad / end-of-turn tokens left behind by ``generate``."""
    special = special_ids(tokenizer)
    end = ids.shape[0]
    while end > 0 and int(ids[end - 1]) in special:
        end -= 1
    return ids[:end]


@torch.no_grad()
def batch_generate(model, tokenizer, sequences, args, max_new_tokens, device):
    """Continue each sequence of token ids; return the new tokens per row.

    Takes ids rather than text: the re-completion prefix is assembled in token
    space, and decoding it back to a string to re-tokenize can merge or split
    tokens at the boundary, which would move the planted token.

    Left padding so every row ends at its own generation point -- with right
    padding the model would continue from pad tokens.
    """
    if max_new_tokens <= 0:
        return [torch.empty(0, dtype=torch.long, device=device) for _ in sequences]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    width = max(int(s.shape[0]) for s in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        ids[i, width - seq.shape[0]:] = seq
        mask[i, width - seq.shape[0]:] = 1
    enc = {"input_ids": ids, "attention_mask": mask}

    sampling = (
        {"do_sample": False}
        if args.greedy
        # top_k and min_p are set explicitly: otherwise the model's generation_config
        # fills them in (Qwen3: top_k=20), and "temperature 1" would not be.
        else {"do_sample": True, "top_p": args.top_p, "temperature": args.temperature,
              "top_k": args.sample_top_k, "min_p": None}
    )
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        **sampling,
    )
    fresh = out[:, enc["input_ids"].shape[1]:]
    return [strip_terminators(row, tokenizer) for row in fresh]


@torch.no_grad()
def logits_at(model, tokenizer, sequences, index, device):
    """Next-token logits at selected positions of a batch of sequences.

    ``index`` is a list of ``(row, position)`` pairs, where ``position`` indexes
    into the padded sequence and the returned logits predict the token *after*
    it. The LM head is applied only at those positions: a full
    ``[batch, length, vocab]`` tensor would be hundreds of gigabytes for a
    500-word answer over a 150k vocabulary.
    """
    width = max(int(s.shape[0]) for s in sequences)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):  # right padding: real tokens keep their index
        ids[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = 1

    rows = torch.tensor([r for r, _ in index], device=device)
    cols = torch.tensor([p for _, p in index], device=device)

    base, head = getattr(model, "model", None), getattr(model, "lm_head", None)
    if base is not None and head is not None:
        hidden = base(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        return head(hidden[rows, cols]).float()
    # Fallback for architectures that do not expose the split.
    return model(input_ids=ids, attention_mask=mask).logits[rows, cols].float()


def lowest_prob(model, tokenizer, sequence, start, skip, device):
    """Lowest probability the model gives any response token except the plant.

    ``start`` is where the response begins in ``sequence`` and ``skip`` the
    plant's index within the response. This is the least likely token the model
    chose itself: what the plant has to stand out against.
    """
    positions = [start + j - 1 for j in range(int(sequence.shape[0]) - start) if j != skip]
    logits = logits_at(model, tokenizer, [sequence], [(0, p) for p in positions], device)
    targets = sequence[[p + 1 for p in positions]]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[torch.arange(len(positions), device=device), targets].min())


def pick_replacement(probs, orig_id, tokenizer, args, rng):
    """Choose a low-probability but plausible substitute for ``orig_id``.

    Candidates are restricted to the probability band ``[p_min, p_max]`` and
    ranked by probability; we sample from the ``top_k`` survivors, i.e. the
    *most* plausible tokens the model still considered unlikely.
    """
    in_band = ((probs < args.p_max) & (probs > args.p_min)).nonzero(as_tuple=True)[0]
    if in_band.numel() == 0:
        return None
    band_probs = probs[in_band]
    # Only inspect the best few hundred: the rest of the band is not plausible.
    order = torch.argsort(band_probs, descending=True)[: args.scan]

    survivors = []
    for idx in order.tolist():
        token_id = int(in_band[idx])
        if token_id == orig_id:
            continue
        text = tokenizer.decode([token_id])
        if not eligible_token(tokenizer, token_id, text, args):
            continue
        prob = float(band_probs[idx])
        if case_variant_outranks(probs, text, prob, tokenizer):
            continue
        survivors.append((token_id, text, prob))
        if len(survivors) >= args.top_k:
            break
    return rng.choice(survivors) if survivors else None


def candidate_positions(tokenizer, response_ids, args, rng):
    """Sample eligible positions to try, cheaply.

    Positions are shuffled and checked until enough eligible ones are found, so
    a long response does not cost a decode call per token.
    """
    lo, hi = args.min_pos, int(response_ids.shape[0]) - args.min_tail
    if hi <= lo:
        return []
    order = list(range(lo, hi))
    rng.shuffle(order)
    chosen = []
    for pos in order:
        token_id = int(response_ids[pos])
        if eligible_token(tokenizer, token_id, tokenizer.decode([token_id]), args):
            chosen.append(pos)
            if len(chosen) >= args.positions:
                break
    return chosen


SURPRISING_NATS = -math.log(0.01)


def turn_summary(surprisals, plant, ids, tokenizer, min_pos):
    """How the plant stands against the model's own tokens in one scoring of the turn.

    ``rank`` counts every token; ``rank_eligible`` only those from ``min_pos`` on,
    where a plant can be placed (the first tokens of a turn read without its think
    block are surprising for reasons that have nothing to do with the plant).
    """
    others = [(s, i) for i, s in enumerate(surprisals) if i != plant]
    top = sorted(others, reverse=True)[:3]
    summary = {
        "n_tokens": len(surprisals),
        "n_above_1pct": sum(1 for s, _ in others if s > SURPRISING_NATS),
        "top_others": [{"index": i, "text": tokenizer.decode([ids[i]]), "surprisal": round(s, 3)}
                       for s, i in top],
    }
    if plant is not None:
        value = surprisals[plant]
        summary.update(
            plant_index=plant,
            plant_surprisal=round(value, 3),
            rank=1 + sum(1 for s, _ in others if s > value),
            rank_eligible=1 + sum(1 for s, i in others if s > value and i >= min_pos),
        )
    return summary


def surprisal_check(model, tokenizer, record, args, device):
    """Score the finished turn after its generation prefix (with the empty think
    block) and as it is read in the detection chat (block dropped by Qwen3).

    For a modified record: is the plant the most surprising token in either
    context? Tokens are re-tokenised from the stored text, as in training.
    """
    out = {}
    for name, reading in (("generation", False), ("reading", True)):
        surprisals, plant = turn_surprisal(model, tokenizer, record, device, reading=reading)
        if not surprisals:
            return None
        ids = tokenizer(record["messages"][1]["content"], add_special_tokens=False).input_ids
        out[name] = turn_summary(surprisals, plant, ids, tokenizer, args.min_pos)
    return out


def build_chat(prompt, assistant_text, tokens, args):
    """The full 2-turn conversation plus the detection exchange."""
    user_turn = prompt
    if args.max_words:
        user_turn = f"{prompt}\n\nKeep your answer under {args.max_words} words."
    return [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": DETECT_QUESTION},
        {"role": "assistant", "content": render_answer(tokens)},
    ]


@torch.no_grad()
def build_records(model, tokenizer, prompts, want_modified, args, rng, device):
    """Generate one chunk of records. ``want_modified`` is a per-prompt flag."""
    chats = [build_chat(p, "", [], args)[:1] for p in prompts]
    prefixes = [
        tokenizer.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": args.enable_thinking} if args.enable_thinking is not None else {}),
        ) for c in chats
    ]
    prefix_ids = [
        tokenizer(p, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        for p in prefixes
    ]
    responses = batch_generate(model, tokenizer, prefix_ids, args, args.max_new_tokens, device)

    # Which rows are long enough to plant in, and were asked for a modification.
    usable = [
        i for i, r in enumerate(responses)
        if want_modified[i]
        and r.shape[0] >= max(args.min_pos + args.min_tail + 1, args.min_response_tokens)
    ]
    positions = {i: candidate_positions(tokenizer, responses[i], args, rng) for i in usable}
    usable = [i for i in usable if positions[i]]

    # One batched scoring pass over every (row, position) pair we might use.
    chosen: dict[int, tuple] = {}
    if usable:
        sequences = [torch.cat([prefix_ids[i], responses[i]]) for i in usable]
        offsets = {i: int(prefix_ids[i].shape[0]) - 1 for i in usable}
        index, owner = [], []
        for slot, i in enumerate(usable):
            for pos in positions[i]:
                index.append((slot, offsets[i] + pos))
                owner.append((i, pos))
        logits = logits_at(model, tokenizer, sequences, index, device)

        for (i, pos), row in zip(owner, logits):
            if i in chosen:
                continue
            probs = torch.softmax(row, dim=-1)
            orig_id = int(responses[i][pos])
            # Where the model was already almost certain, nothing else "makes
            # sense in the context" and every in-band candidate is junk.
            if float(probs[orig_id]) > args.max_orig_prob:
                continue
            choice = pick_replacement(probs, orig_id, tokenizer, args, rng)
            if choice is not None:
                chosen[i] = (pos, orig_id, float(probs[orig_id]), choice)

    # Re-complete each plant, conditioned on the swapped token, in one batch.
    tails: dict[int, torch.Tensor] = {}
    if chosen:
        rows = sorted(chosen)
        heads, budgets = [], []
        for i in rows:
            pos, _, _, (new_id, _, _) = chosen[i]
            swapped = torch.tensor([new_id], device=device)
            heads.append(torch.cat([prefix_ids[i], responses[i][:pos], swapped]))
            # Only the *remaining* budget: with a fresh allowance a modified turn
            # could run to twice the length of a clean one, handing the model a
            # length cue to classify on instead of surprisal.
            budgets.append(args.max_new_tokens - pos - 1)
        generated = batch_generate(model, tokenizer, heads, args, max(budgets), device)
        for i, tail, budget in zip(rows, generated, budgets):
            tails[i] = tail[:budget]

    records = []
    for i, prompt in enumerate(prompts):
        if i in chosen and i in tails:
            pos, orig_id, orig_prob, (new_id, new_text, new_prob) = chosen[i]
            swapped = torch.tensor([new_id], device=device)
            ids = torch.cat([responses[i][:pos], swapped, tails[i]])
            records.append({
                "prompt": prompt,
                "label": "modified",
                "n_response_tokens": int(ids.shape[0]),
                "truncated": int(ids.shape[0]) >= args.max_new_tokens,
                "clean_response": tokenizer.decode(responses[i], skip_special_tokens=True),
                "messages": build_chat(
                    prompt, tokenizer.decode(ids, skip_special_tokens=True), [new_text], args
                ),
                "modification": {
                    # Index into the assistant response's own token sequence, so
                    # it survives a later change of chat template. Used by the
                    # probing follow-up to find the high-surprisal position.
                    "token_index": pos,
                    "orig_token_id": orig_id,
                    "orig_token_text": tokenizer.decode([orig_id]),
                    "new_token_id": new_id,
                    "new_token_text": new_text,
                    "orig_prob": orig_prob,
                    "new_prob": new_prob,
                    "surprisal_nats": -math.log(new_prob),
                    "unmodified_lowest_prob": lowest_prob(
                        model, tokenizer, torch.cat([prefix_ids[i], ids]),
                        int(prefix_ids[i].shape[0]), pos, device,
                    ),
                },
            })
        elif responses[i].shape[0] >= max(1, args.min_response_tokens):
            # Either a clean example was requested, or the plant failed. Keeping
            # the already-generated response rather than discarding it avoids
            # wasting the sample, and stops a run of failures from starving the
            # shard (the label is chosen from the realised balance).
            records.append({
                "prompt": prompt,
                "label": "clean",
                "n_response_tokens": int(responses[i].shape[0]),
                "truncated": int(responses[i].shape[0]) >= args.max_new_tokens,
                "messages": build_chat(
                    prompt, tokenizer.decode(responses[i], skip_special_tokens=True), [], args
                ),
                "modification": None,
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the prefill-detection dataset.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="explicit thinking mode for models such as Qwen3")
    parser.add_argument("--prompts", required=True,
                        help="path to .txt/.jsonl, or hf:<name>[:<split>[:<field>]]")
    parser.add_argument("--categories", default="",
                        help="comma-separated category filter, e.g. 'brainstorming' for Dolly")
    parser.add_argument("--out", required=True,
                        help="output .jsonl (a shard suffix is added when --shards > 1)")
    parser.add_argument("--n", type=int, default=1000, help="number of prompts to attempt")
    parser.add_argument("--batch-size", type=int, default=8, help="prompts generated together")
    parser.add_argument("--neg-frac", type=float, default=0.5, help="fraction of clean examples")
    parser.add_argument("--test-pct", type=int, default=10, help="percent of prompts held out for test")
    # Length is meant to be governed by the instruction the model follows, not by
    # the decode cap. --max-new-tokens is a safety net that should not bind; the
    # per-record "truncated" flag tells you when it did.
    parser.add_argument("--max-words", type=int, default=500,
                        help="word budget requested in the user turn; 0 to omit the instruction")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="hard decode cap (default: 2x --max-words, i.e. non-binding)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="nucleus threshold; 1.0 samples from the full distribution")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true",
                        help="argmax decoding instead of sampling (ignores --top-p/--temperature)")
    parser.add_argument("--sample-top-k", type=int, default=0,
                        help="top-k when sampling; 0 = off, so --top-p 1.0 samples the full distribution")
    parser.add_argument("--p-max", type=float, default=0.01, help="replacement must be below this probability")
    # The band is the main quality knob. Too low a floor and the only candidates
    # at a confidently-predicted position are junk; [1e-4, 1e-2] is 4.6-9.2 nats
    # of surprisal, high above ordinary tokens but still a real alternative.
    parser.add_argument("--p-min", type=float, default=1e-4, help="...and above this, to stay plausible")
    parser.add_argument("--max-orig-prob", type=float, default=0.9,
                        help="only modify positions where the model was not already this certain")
    parser.add_argument("--top-k", type=int, default=16, help="sample the replacement from the K best in-band tokens")
    parser.add_argument("--scan", type=int, default=256, help="how many in-band tokens to inspect")
    parser.add_argument("--positions", type=int, default=16, help="candidate positions scored per response")
    parser.add_argument("--min-pos", type=int, default=8, help="do not modify the first N response tokens")
    parser.add_argument("--min-tail", type=int, default=8, help="leave at least N tokens after the modification")
    parser.add_argument("--min-response-tokens", type=int, default=0,
                        help="skip prompts whose response is shorter, for clean and modified alike, "
                             "so answer length cannot give the label away")
    parser.add_argument("--word-tokens-only", action="store_true", default=True)
    parser.add_argument("--any-token", dest="word_tokens_only", action="store_false",
                        help="allow punctuation and sub-word fragments to be swapped")
    parser.add_argument("--shards", type=int, default=1, help="total shards across independent generation jobs")
    parser.add_argument("--shard", type=int, default=0, help="index of this shard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-prefix", default="",
                        help="prefix for record ids (e.g. the Condor job), so ids from different "
                             "datasets never collide")
    parser.add_argument("--no-surprisal-check", dest="surprisal_check", action="store_false",
                        help="skip scoring every finished turn in the generation and reading contexts")
    parser.add_argument("--dtype", default="bfloat16", help="use float32 on CPU; bf16 has no CPU kernels")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="generate from uncommitted code (throwaway runs only)")
    args = parser.parse_args()
    # Checked before the model loads, so a refused run costs nothing.
    git = require_committed(parser, args.allow_dirty)
    torch.manual_seed(args.seed + args.shard)

    if args.max_new_tokens is None:
        args.max_new_tokens = max(256, args.max_words * 2)
    categories = {c.strip().lower() for c in args.categories.split(",") if c.strip()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map=device
    )
    model.eval()

    prompts = load_prompts(args.prompts, args.n, args.seed, categories)[args.shard::args.shards]
    rng = random.Random(args.seed + args.shard)

    out_path = Path(args.out)
    if args.shards > 1:
        out_path = out_path.with_suffix(f".{args.shard:03d}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Everything needed to reproduce this file, written before generation starts
    # so a crashed run is still traceable, and completed at the end.
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        **describe("prefill_awareness.data_gen", args, git, model),
        "prompts": {
            "source": args.prompts,
            "n": len(prompts),
            # Of the prompts actually used, in order, as a JSON list.
            "sha256": hashlib.sha256(
                json.dumps(prompts, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        },
        "counts": None,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    counts = {"clean": 0, "modified": 0, "skipped": 0}
    target_modified = 1.0 - args.neg_frac
    with out_path.open("w", encoding="utf-8") as handle:
        chunks = range(0, len(prompts), args.batch_size)
        for start in tqdm(chunks, desc=f"shard {args.shard}"):
            chunk = prompts[start:start + args.batch_size]
            # Pick labels from the balance realised so far rather than by a coin
            # flip. Plants fail more often than clean samples and failures fall
            # back to clean, so an i.i.d. flip would skew the set towards clean
            # and flatter a model that simply always answers "none".
            want, provisional = [], dict(counts)
            for _ in chunk:
                written = provisional["clean"] + provisional["modified"]
                pick = target_modified > 0 and (
                    written == 0 or provisional["modified"] / written < target_modified
                )
                want.append(pick)
                provisional["modified" if pick else "clean"] += 1

            records = build_records(model, tokenizer, chunk, want, args, rng, device)
            # Prompts with an empty or too-short response produce no record.
            counts["skipped"] += len(chunk) - len(records)
            for offset, record in enumerate(records):
                record["id"] = f"{args.id_prefix}{args.shard:03d}-{start + offset:06d}"
                record["split"] = split_of(record["prompt"], args.test_pct)
                record["model"] = args.model
                record["enable_thinking"] = args.enable_thinking
                if args.surprisal_check:
                    record["surprisal_check"] = surprisal_check(model, tokenizer, record, args, device)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                counts[record["label"]] += 1
    meta.update(finished_at=utc_now(), counts=counts)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {counts['clean'] + counts['modified']}/{len(prompts)} records to {out_path}: {counts}")


if __name__ == "__main__":
    main()
