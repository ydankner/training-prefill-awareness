"""Assemble tampering-marker conversations from generated history turns.

Every history turn (``tamper_gen``) is used exactly once, in one of two roles:

    detection  [Q1, history, DETECT_QUESTION]  -> "¦tampered: a, b" / "not tampered"
    followup   [Q1, history, Q2]               -> "¦" + own answer / own answer

60% of the histories of each (split, plant count, mode) cell go to detection,
the rest to follow-up. Q2 is a No Robots request (single-turn, no passage),
each used once, from the same hash split as the history's prompt. The follow-up
answer is the model's own sampled answer in that very conversation, generated
here with the same non-thinking sampler as the histories; the marker is only
prepended to the target, so a tampered target is the marker plus exactly what the
model said.

Generated answers are appended to ``<out>.followups.jsonl`` and reused on a
restart; the conversation file is written at the end.

    python -m prefill_awareness.tamper_data --histories data/tamper-qwen38/histories.jsonl \
        --followups data/no-robots-prompts.jsonl --out data/tamper-qwen38/conversations.jsonl
"""

import argparse
import hashlib
import json
import random
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from .data_gen import split_of
from .provenance import describe, file_sha256, require_committed, utc_now
from .tamper_formats import DETECT_QUESTION, MARKER, detection_answer, followup_answer


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def assign_roles(histories: list[dict], detect_frac: float, seed: int) -> dict[str, str]:
    cells = defaultdict(list)
    for h in histories:
        cells[(h["split"], h["n_plants"], h["mode"])].append(h["id"])
    roles = {}
    for key in sorted(cells):
        ids = sorted(cells[key])
        random.Random(f"{seed}:{key}").shuffle(ids)
        cut = round(len(ids) * detect_frac)
        roles.update({i: "detection" for i in ids[:cut]})
        roles.update({i: "followup" for i in ids[cut:]})
    return roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tampering-marker conversations.")
    parser.add_argument("--histories", nargs="+", required=True, help="tamper_gen outputs (all shards)")
    parser.add_argument("--followups", required=True, help="No Robots prompt file (prepare_no_robots.py)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--detect-frac", type=float, default=0.6)
    parser.add_argument("--test-pct", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    # Shards of one prompt order can overlap where a run was stopped late; the
    # first record of a prompt (in the order of --histories) is kept.
    histories, seen = [], set()
    for h in (h for path in args.histories for h in read_jsonl(path) if h.get("label")):
        if h["prompt"] not in seen:
            seen.add(h["prompt"])
            histories.append(h)
    model_name = histories[0]["model"]
    roles = assign_roles(histories, args.detect_frac, args.seed)
    pool = defaultdict(list)
    for row in read_jsonl(args.followups):
        pool[split_of(row["prompt"], args.test_pct)].append(row)
    for split in pool:
        random.Random(f"{args.seed}:{split}").shuffle(pool[split])
    suffix = f"\n\nKeep your answer under {args.max_words} words." if args.max_words else ""

    # Pair follow-up histories with questions, in id order for reproducibility.
    pairs = {}
    for h in sorted(histories, key=lambda h: h["id"]):
        if roles[h["id"]] != "followup":
            continue
        if not pool[h["split"]]:
            roles[h["id"]] = "detection"  # question pool exhausted
            continue
        pairs[h["id"]] = pool[h["split"]].pop()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out.with_suffix(".followups.jsonl")
    cache = {r["history_id"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    todo = [h for h in histories if h["id"] in pairs and h["id"] not in cache]

    meta = {"command": None, "args": vars(args), "git": git, "started_at": utc_now(), "finished_at": None}
    if todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .tamper_gen import sample

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, args.dtype),
                                                     device_map=device)
        model.eval()
        meta = describe("prefill_awareness.tamper_data", args, git, model)
        prompts = {}
        for h in todo:
            chat = h["messages"][:2] + [{"role": "user", "content": pairs[h["id"]]["prompt"] + suffix}]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            prompts[h["id"]] = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        todo.sort(key=lambda h: int(prompts[h["id"]].shape[0]))
        with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
            for start in tqdm(range(0, len(todo), args.batch_size), desc="follow-ups"):
                batch = todo[start:start + args.batch_size]
                seed = int(hashlib.md5(f"{args.seed}:{batch[0]['id']}".encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                answers = sample(model, tokenizer, [prompts[h["id"]] for h in batch], args.max_new_tokens,
                                 args, device)
                for h, ids in zip(batch, answers):
                    row = {"history_id": h["id"], "answer": tokenizer.decode(ids, skip_special_tokens=True),
                           "n_tokens": int(ids.shape[0]),
                           "truncated": int(ids.shape[0]) >= args.max_new_tokens}
                    cache[h["id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    marker_id = tokenizer.encode(MARKER, add_special_tokens=False)
    conversations, stats = [], Counter()
    for h in sorted(histories, key=lambda h: h["id"]):
        role = roles[h["id"]]
        tampered = h["label"] == "tampered"
        common = {key: h[key] for key in ("split", "label", "n_plants", "mode", "plants", "model",
                                          "enable_thinking", "clean_response", "category",
                                          "n_response_tokens")}
        common["history_id"] = h["id"]
        common["plant_ranks"] = (h.get("surprisal_check") or {}).get("plant_ranks")
        if h.get("history_thinking", "off") != "off":  # tamper_gen --thinking: reasoning hidden on reading
            common["history_thinking"] = h["history_thinking"]
            common["plant_ranks_hidden"] = (h.get("surprisal_check_hidden") or {}).get("plant_ranks")
        if role == "detection":
            target = detection_answer([p["new_token_text"] for p in h["plants"]])
            messages = h["messages"][:2] + [{"role": "user", "content": DETECT_QUESTION},
                                            {"role": "assistant", "content": target}]
            extra = {}
        else:
            answer = cache[h["id"]]["answer"]
            if not answer.strip():
                stats["empty_followup_answer"] += 1
                continue
            target = followup_answer(answer, tampered)
            q2 = pairs[h["id"]]
            messages = h["messages"][:2] + [{"role": "user", "content": q2["prompt"] + suffix},
                                            {"role": "assistant", "content": target}]
            extra = {"followup_prompt": q2["prompt"], "followup_category": q2["category"],
                     "followup_truncated": cache[h["id"]]["truncated"]}
        if tampered:
            ids = tokenizer.encode(target, add_special_tokens=False)
            rest = tokenizer.encode(target[len(MARKER):], add_special_tokens=False)
            if ids[: len(marker_id)] != marker_id:
                stats["marker_merged"] += 1
            elif ids[len(marker_id):] != rest:
                stats["marker_changed_rest"] += 1
        conversations.append({"id": f"{h['id']}-{role[:3]}", "role": role, "messages": messages,
                              **common, **extra})
        stats[f"{h['split']}-{role}-{h['label']}"] += 1

    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in conversations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta.update(
        command=meta.get("command") or shlex.join(["python", "-m", "prefill_awareness.tamper_data", *sys.argv[1:]]),
        inputs={**{p: file_sha256(p) for p in args.histories}, args.followups: file_sha256(args.followups)},
        followup_answers=str(cache_path), stats=dict(stats), n=len(conversations),
        sha256=file_sha256(out), finished_at=utc_now(),
    )
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
