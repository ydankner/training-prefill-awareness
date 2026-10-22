"""Measure whether another model's highest-surprisal answer token hits a plant.

The source model's token id is meaningful only to its own tokenizer. Locate that
token in the stored answer with the source tokenizer, then transfer its character
span to the evaluated model's tokens through offset mappings. A hit means the
single highest-surprisal evaluated-model token overlaps the plant span.
"""

import argparse
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .dataset import load_records, thinking_for
from .evaluate import load_model
from .provenance import file_sha256, utc_now
from .surprisal import locate_plant, split_backbone


def aligned_answer(source_tokenizer, target_tokenizer, record):
    """Return target token ids, plant indices and source span, or a skip reason."""
    answer = record["messages"][1]["content"]
    modification = record["modification"]
    source = source_tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    source_index = locate_plant(source["input_ids"], modification)
    if source_index is None:
        # A re-completion can join the planted source token to its next piece
        # (e.g. " spo" + "oning" becomes " spoon" + "ing"). Recover only when
        # the planted text occurs once and is beside its stored token index.
        planted_text = modification["new_token_text"]
        if not planted_text or answer.count(planted_text) != 1 or not source["input_ids"]:
            return None, "source_token_not_found"
        start = answer.index(planted_text)
        end = start + len(planted_text)
        guess = min(modification["token_index"], len(source["input_ids"]) - 1)
        if abs(start - source["offset_mapping"][guess][0]) > 8:
            return None, "source_text_far_from_index"
        alignment = "unique_text_fallback"
    else:
        start, end = source["offset_mapping"][source_index]
        alignment = "source_token_id"
    if start >= end:
        return None, "empty_source_span"
    # Decode of a generated token can differ in surrounding whitespace, but
    # the actual planted word must still occur inside its mapped token span.
    planted = modification["new_token_text"].strip()
    if planted and planted not in answer[start:end].strip():
        return None, "source_span_text_mismatch"
    target = target_tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    indices = [i for i, (lo, hi) in enumerate(target["offset_mapping"])
               if lo < end and hi > start]
    if not indices:
        return None, "target_span_not_found"
    return {
        "answer_ids": target["input_ids"],
        "plant_indices": indices,
        "source_span": [start, end],
        "alignment": alignment,
    }, None


@torch.inference_mode()
def token_surprisals(model, prefix_ids, answer_ids, chunk=128):
    """Teacher-forced token surprisals in nats, restricted to the answer."""
    ids = torch.tensor([prefix_ids + answer_ids], device=model.device)
    backbone, head = split_backbone(model)
    if backbone is None or head is None:
        raise TypeError("model must expose a backbone and LM head")
    hidden = backbone(input_ids=ids, use_cache=False).last_hidden_state[0]
    start = len(prefix_ids)
    values = []
    for lo in range(0, len(answer_ids), chunk):
        hi = min(lo + chunk, len(answer_ids))
        rows = torch.arange(start + lo - 1, start + hi - 1, device=model.device)
        logits = head(hidden[rows]).float()
        wanted = torch.tensor(answer_ids[lo:hi], device=model.device)
        values.extend((logits.logsumexp(-1) - logits.gather(1, wanted[:, None]).squeeze(1)).tolist())
    return values


def score(model, prefix_ids, aligned):
    values = token_surprisals(model, prefix_ids, aligned["answer_ids"])
    winner = max(range(len(values)), key=values.__getitem__)
    plant_indices = aligned["plant_indices"]
    plant_best = max(values[i] for i in plant_indices)
    return {
        "hit": winner in plant_indices,
        "winner_index": winner,
        "winner_surprisal": round(values[winner], 5),
        "plant_best_surprisal": round(plant_best, 5),
        "plant_rank": 1 + sum(value > plant_best for i, value in enumerate(values)
                              if i not in plant_indices),
    }


def summary(rows):
    result = {}
    for split in ("all", "train", "test"):
        subset = [row for row in rows if split == "all" or row["split"] == split]
        n = len(subset)
        if not n:
            continue
        base = sum(row["base"]["hit"] for row in subset)
        trained = sum(row["trained"]["hit"] for row in subset)
        delta = [(int(row["trained"]["hit"]) - int(row["base"]["hit"])) for row in subset]
        rng = random.Random(0)
        draws = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        result[split] = {
            "n": n,
            "base_hits": base,
            "base_rate": base / n,
            "trained_hits": trained,
            "trained_rate": trained / n,
            "trained_minus_base": (trained - base) / n,
            "paired_bootstrap_95": [draws[49], draws[1949]],
            "gain": sum(x == 1 for x in delta),
            "loss": sum(x == -1 for x in delta),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--thinking-mode", choices=("source", "on", "off", "omit"),
                        default="off", help="chat mode of the evaluated model")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="cap modified records for a pilot")
    parser.add_argument("--out", required=True, help="streamed JSONL results")
    parser.add_argument("--audit-only", action="store_true", help="check alignment on CPU")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    meta_path = out.with_suffix(".meta.json")
    records = [r for r in load_records(args.data, None if args.split == "all" else args.split)
               if r.get("modification")]
    if args.limit is not None:
        records = records[:args.limit]
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.model)
    valid, skipped = [], Counter()
    for index, record in enumerate(records):
        aligned, reason = aligned_answer(source_tokenizer, target_tokenizer, record)
        if reason:
            skipped[reason] += 1
        else:
            valid.append((index, record, aligned))
    print(json.dumps({"modified": len(records), "valid": len(valid), "skipped": skipped,
                      "plant_token_counts": Counter(len(a["plant_indices"]) for _, _, a in valid),
                      "alignment": Counter(a["alignment"] for _, _, a in valid)},
                     default=dict), flush=True)
    if args.audit_only:
        return

    config = {"source_model": args.source_model, "model": args.model,
              "thinking_mode": args.thinking_mode, "adapter": args.adapter,
              "data": {path: file_sha256(path) for path in args.data},
              "split": args.split, "limit": args.limit}
    done = set()
    if args.resume:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("resume configuration differs from existing results")
        done = {json.loads(line)["index"] for line in out.read_text(encoding="utf-8").splitlines()}
    else:
        if out.exists() or meta_path.exists():
            raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"config": config, "started_at": utc_now(),
                                         "finished_at": None, "n_modified": len(records),
                                         "n_valid": len(valid), "skipped": skipped}, indent=2) + "\n",
                             encoding="utf-8")
    model, target_tokenizer = load_model(args.model, args.adapter)
    started = time.monotonic()
    with out.open("a", encoding="utf-8") as handle:
        for index, record, aligned in valid:
            if index in done:
                continue
            thinking = thinking_for(record, args.thinking_mode)
            mode = {"enable_thinking": thinking} if thinking is not None else {}
            prefix = target_tokenizer.apply_chat_template(
                record["messages"][:1], tokenize=False, add_generation_prompt=True,
                **mode)
            prefix_ids = target_tokenizer(prefix, add_special_tokens=False).input_ids
            with model.disable_adapter():
                base = score(model, prefix_ids, aligned)
            trained = score(model, prefix_ids, aligned)
            row = {"index": index, "id": record.get("id"), "split": record["split"],
                   "n_answer_tokens": len(aligned["answer_ids"]),
                   "plant_indices": aligned["plant_indices"],
                   "source_span": aligned["source_span"], "alignment": aligned["alignment"],
                   "base": base, "trained": trained}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if (len(done) + 1) % 25 == 0:
                print(f"scored {len(done) + 1}/{len(valid)} in {time.monotonic() - started:.1f}s", flush=True)
            done.add(index)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(valid):
        raise RuntimeError(f"only {len(rows)} of {len(valid)} valid records scored")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(finished_at=utc_now(), summary=summary(rows),
                git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
