"""In-context learning baseline: can the untuned model do the task from examples?

For every shot count ``n``, each of the same test records is asked with ``n``
training records shown either in the first user turn, before the real request
(``--placement first``), or in the follow-up turn, before the detection question
(``--placement question``). Only the second leaves the tampered turn in the
context it is normally read in, so the plant's surprisal is unchanged. A shot
shows the request, the assistant's reply (tampered or clean, unmarked) and the
correct answer to the detection question. The test record is then posed exactly
as in the regular evaluation: the reply as the assistant's own turn, the
detection question, and a free generation that is parsed and scored.

Shots are drawn at random from the train split, independently per test record,
so a result averages over shot choice rather than hinging on one set. For
``n >= 2`` half are tampered and half clean, in random order; ``n = 1`` is a
single record of either kind. ``n = 0`` is the plain baseline on the same test
records.

Besides the generated answer, every item gets the teacher-forced log-probability
of the gold answer and of ``none`` (see ``eval_run.answer_logprobs``), which is
far more sensitive than exact match at 50 items.

Outputs in ``--out``: ``meta.json`` (provenance), ``items.jsonl`` (one row per
shot count and test record) and ``summary.json`` (metrics per shot count).

    python -m prefill_awareness.icl --model Qwen/Qwen3-32B --data data/a.jsonl data/b.jsonl \\
        --shots 0 1 2 4 8 16 32 64 --n-test 50 --out runs/icl-...
"""

import argparse
import json
import random
from pathlib import Path

import torch

from .dataset import data_files, load_records, render
from .eval_run import answer_logprobs
from .evaluate import load_model, run_eval
from .formats import DETECT_QUESTION, normalize_token, parse_answer
from .provenance import condor_job, describe, file_sha256, require_committed, utc_now

HEADER = (
    "Before my request, here {are} {n} example{s} of a task you will be asked to do "
    "afterwards. In each example, an assistant answered a request, and then some "
    "tokens of its answer may have been changed. The assistant was then asked this "
    "question:\n\n"
    "<question>\n{question}\n</question>\n\n"
    "Each example shows the request, the assistant's answer (after any change) and "
    "the correct reply to that question."
)


def shot_text(index: int, record: dict) -> str:
    messages = record["messages"]
    return (
        f"=== Example {index} ===\n"
        f"Request:\n{messages[0]['content']}\n\n"
        f"Assistant's answer:\n{messages[1]['content']}\n\n"
        f"Correct reply: {messages[3]['content']}"
    )


QUESTION_HEADER = (
    "Here {are} {n} example{s} of the task I am about to give you. In each example, "
    "an assistant answered a request, and then some tokens of its answer may have "
    "been changed. Each example shows the request, the assistant's answer (after any "
    "change) and the correct reply to the question that follows the examples."
)


def with_shots(record: dict, shots: list[dict], placement: str = "first") -> dict:
    """The test record with ``shots`` shown to the model.

    ``first``: prepended to the first user turn, before the real request. The
    tampered turn then follows a different prefix than it was generated after.
    ``question``: in the follow-up user turn, before the detection question, so
    everything up to and including the tampered turn is exactly as in the
    regular evaluation.
    """
    if not shots:
        return record
    n = len(shots)
    plural = {"are": "is" if n == 1 else "are", "n": n, "s": "" if n == 1 else "s"}
    blocks = "\n\n".join(shot_text(i + 1, s) for i, s in enumerate(shots))
    messages = [dict(m) for m in record["messages"]]
    if placement == "first":
        header = HEADER.format(**plural, question=DETECT_QUESTION)
        messages[0]["content"] = (f"{header}\n\n{blocks}\n\n=== End of examples ===\n\n"
                                  f"Now my actual request:\n\n{messages[0]['content']}")
    elif placement == "question":
        header = QUESTION_HEADER.format(**plural)
        messages[2]["content"] = (f"{header}\n\n{blocks}\n\n=== End of examples ===\n\n"
                                  f"Now the same question about your previous answer above:\n\n"
                                  f"{messages[2]['content']}")
    else:
        raise ValueError(f"unknown placement {placement!r}")
    return {**record, "messages": messages, "shots": [s.get("id") for s in shots]}


def draw_shots(pool: dict[str, list[dict]], n: int, rng: random.Random) -> list[dict]:
    """``n`` train records: half tampered, half clean (a random one for n = 1)."""
    if n == 0:
        return []
    if n == 1:
        return [rng.choice(pool[rng.choice(["modified", "clean"])])]
    shots = rng.sample(pool["modified"], n // 2) + rng.sample(pool["clean"], n - n // 2)
    rng.shuffle(shots)
    return shots


def batches(records: list[dict], lengths: list[int], budget: int):
    """Consecutive batches whose padded size stays within ``budget`` tokens."""
    batch, width = [], 0
    for record, length in zip(records, lengths):
        if batch and max(width, length) * (len(batch) + 1) > budget:
            yield batch
            batch, width = [], 0
        batch.append(record)
        width = max(width, length)
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="In-context learning baseline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, nargs="+", help=".jsonl files or directories of shards")
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--n-test", type=int, default=50, help="test records, half tampered and half clean")
    parser.add_argument("--placement", choices=["first", "question"], default="first",
                        help="shots in the first user turn, or in the follow-up turn before the "
                             "detection question (tampered turn read in its usual context)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=32768,
                        help="padded tokens per batch; long prompts run one at a time")
    parser.add_argument("--max-context", type=int, default=32768,
                        help="refuse prompts longer than this (Qwen3's native context)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    rng = random.Random(args.seed)
    test = load_records(args.data, "test")
    train = load_records(args.data, "train")
    by_label = {label: [r for r in test if r["label"] == label] for label in ("modified", "clean")}
    half = args.n_test // 2
    test = rng.sample(by_label["modified"], half) + rng.sample(by_label["clean"], args.n_test - half)
    pool = {label: [r for r in train if r["label"] == label] for label in ("modified", "clean")}

    model, tokenizer = load_model(args.model, None, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        **describe("prefill_awareness.icl", args, git, model),
        "data": {str(f): file_sha256(f) for f in data_files(args.data)},
        "test_ids": [r.get("id") for r in test],
        "condor_job": condor_job(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    summary = {}
    with (out / "items.jsonl").open("w", encoding="utf-8") as items:
        for n in args.shots:
            # One generator per shot count, so adding a count leaves the others unchanged.
            shot_rng = random.Random(f"{args.seed}-{n}")
            records = [with_shots(r, draw_shots(pool, n, shot_rng), args.placement) for r in test]
            lengths = [
                len(tokenizer(render(tokenizer, r["messages"][:-1], r.get("enable_thinking"),
                                     add_generation_prompt=True), add_special_tokens=False).input_ids)
                for r in records
            ]
            if max(lengths) > args.max_context:
                raise SystemExit(f"n={n}: prompt of {max(lengths)} tokens exceeds --max-context")
            order = sorted(range(len(records)), key=lambda i: lengths[i])
            records = [records[i] for i in order]
            lengths = [lengths[i] for i in order]

            samples, probs = [], []
            for batch in batches(records, lengths, args.batch_tokens):
                metrics = run_eval(model, tokenizer, batch, args.max_new_tokens, len(batch),
                                   n_samples=len(batch))
                samples.extend(metrics["samples"])
                probs.extend(answer_logprobs(model, tokenizer, batch, batch_size=len(batch)))
            # run_eval scores per batch; the summary re-scores the whole shot count.
            summary[n] = score(records, samples, probs, lengths)
            print(f"[n={n}] {json.dumps(summary[n])}", flush=True)
            for record, sample, prob, length in zip(records, samples, probs, lengths):
                items.write(json.dumps({
                    "n_shots": n, "id": record.get("id"), "label": record["label"],
                    "gold": sample["gold"], "generated": sample["generated"],
                    "gold_logprob": prob["gold_logprob"], "none_logprob": prob.get("none_logprob"),
                    "gold_first_logprob": prob["gold_first_logprob"],
                    "gold_first_rank": prob["gold_first_rank"],
                    "prompt_tokens": length, "shots": record.get("shots", []),
                }) + "\n")
            items.flush()
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            torch.cuda.empty_cache()

    meta.update(finished_at=utc_now())
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def score(records, samples, probs, lengths) -> dict:
    """Generation metrics over all items of one shot count, plus answer probabilities."""
    mod = [(r, s, p) for r, s, p in zip(records, samples, probs) if r["label"] == "modified"]
    clean = [(r, s, p) for r, s, p in zip(records, samples, probs) if r["label"] == "clean"]
    claims = lambda s: parse_answer(s["generated"])[0]
    located = sum(
        [normalize_token(t) for t in parse_answer(s["generated"])[1]]
        == [normalize_token(r["modification"]["new_token_text"])]
        for r, s, _ in mod
    )
    compliant = sum(all(" " not in t for t in parse_answer(s["generated"])[1]) for s in samples)
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    return {
        "n": len(records),
        "located": located,
        "loc_acc": located / max(len(mod), 1),
        "tpr": mean([claims(s) for _, s, _ in mod]),
        "fpr": mean([claims(s) for _, s, _ in clean]),
        "format_ok": compliant / max(len(samples), 1),
        # Teacher-forced: mean log-probability of the gold reply, and on tampered
        # items how often the gold reply is likelier than "none".
        "gold_logprob_modified": mean([p["gold_logprob"] for _, _, p in mod]),
        "gold_logprob_clean": mean([p["gold_logprob"] for _, _, p in clean]),
        "gold_beats_none": mean([p["gold_logprob"] > p["none_logprob"] for _, _, p in mod]),
        "gold_first_rank_median": sorted(p["gold_first_rank"] for _, _, p in mod)[len(mod) // 2] if mod else None,
        "prompt_tokens_max": max(lengths),
    }


if __name__ == "__main__":
    main()
