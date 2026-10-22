"""Evaluate every checkpoint of a training run with one model load.

For each checkpoint this generates answers on the whole test split and on a
random train sample of the same size, so the train/test gap shows how much of
the result is memorisation. It also profiles surprisal on every modified test
record, with the adapter on and off, keeping per-record ranks so that a plant
which is not the most surprising token can be inspected. With --answer-probs it
also scores the gold answer of every test record teacher-forced, for each
checkpoint and the base model (``answer-probs-<checkpoint>-<tag>.jsonl``). With
--token-stats it writes per-token surprisal, entropy, top-1 probability and rank
of the actual token for every test record (``token-stats-<checkpoint>-<tag>.jsonl``).

    python -m prefill_awareness.eval_run --model Qwen/Qwen3-32B \
        --run runs/sft-... --data data/a.jsonl data/b.jsonl

Outputs land in the run directory: ``eval-<checkpoint>-<split>-<tag>.json``,
``surprisal-<tag>.json`` and ``eval-meta-<tag>.json`` (provenance).
"""

import argparse
import json
import random
from pathlib import Path

import torch

from .dataset import Collator, data_files, encode, load_records
from .evaluate import load_model, run_eval
from .formats import ANSWER_FORMATS, NO_CHANGE
from .provenance import condor_job, describe, file_sha256, require_committed, utc_now
from .surprisal import profile, split_backbone, turn_surprisal


@torch.no_grad()
def answer_logprobs(model, tokenizer, records, batch_size: int = 8) -> list[dict]:
    """Teacher-forced log-probability of the gold answer for every record.

    For each record: the log-probability of the whole gold reply (its tokens and
    the end of turn), of its first token, and that token's rank in the vocabulary.
    Modified records also get the log-probability of answering ``none``, the
    alternative the model has to beat.
    """
    backbone, head = split_backbone(model)
    variants = []
    for index, record in enumerate(records):
        mode = (record.get("enable_thinking"), record.get("keep_think", False))
        variants.append((index, "gold", encode(tokenizer, record["messages"], *mode)))
        if record["label"] == "modified":
            messages = record["messages"][:-1] + [{"role": "assistant", "content": NO_CHANGE}]
            variants.append((index, "none", encode(tokenizer, messages, *mode)))

    results = [
        {"index": i, "id": r.get("id"), "label": r["label"], "gold": r["messages"][-1]["content"]}
        for i, r in enumerate(records)
    ]
    collate = Collator(tokenizer.pad_token_id)
    for start in range(0, len(variants), batch_size):
        chunk = variants[start:start + batch_size]
        batch = collate([{"input_ids": ids, "labels": labels} for _, _, (ids, labels) in chunk])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        for row, (index, kind, _) in enumerate(chunk):
            # The logits at position t predict token t + 1.
            targets = (batch["labels"][row, 1:] != -100).nonzero().squeeze(-1)
            logits = head(hidden[row, targets]).float()
            wanted = batch["labels"][row, targets + 1]
            logp = torch.log_softmax(logits, dim=-1)
            token_logp = logp[torch.arange(len(targets), device=logp.device), wanted]
            out = results[index]
            out[f"{kind}_logprob"] = float(token_logp.sum())
            if kind == "gold":
                out["gold_first_logprob"] = float(token_logp[0])
                out["gold_first_rank"] = int((logits[0] > logits[0, wanted[0]]).sum()) + 1
    return results


@torch.no_grad()
def token_stats(model, tokenizer, records) -> list[dict]:
    """Per-token surprisal, predictive entropy, top-1 probability and rank of the
    actual token over the assistant turn of every record (see turn_surprisal)."""
    rows = []
    for index, record in enumerate(records):
        stats = {"entropy": [], "top1_prob": [], "rank": []}
        surprisal, plant = turn_surprisal(model, tokenizer, record, model.device, stats=stats)
        rows.append({
            "index": index, "id": record.get("id"), "label": record["label"], "plant_index": plant,
            "orig_logprob": stats.get("orig_logprob"), "ids": stats.get("ids", []),
            "surprisal": [round(x, 4) for x in surprisal],
            "entropy": [round(x, 4) for x in stats["entropy"]],
            "top1_prob": [round(x, 5) for x in stats["top1_prob"]],
            "rank": stats["rank"],
        })
    return rows


def write_answer_probs(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints of a run.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True, help="training run directory")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="adapter directories inside --run (default: every checkpoint-*)")
    parser.add_argument("--data", required=True, nargs="+", help=".jsonl files or directories of shards")
    parser.add_argument("--train-n", type=int, default=None,
                        help="train records to sample (default: as many as the test split)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-surprisal", dest="surprisal", action="store_false")
    parser.add_argument("--no-generate", dest="generate", action="store_false",
                        help="skip generating answers on the test split and train sample")
    parser.add_argument("--answer-probs", action="store_true",
                        help="teacher-forced probability of the gold answer on the test split, "
                             "for every checkpoint and the base model")
    parser.add_argument("--token-stats", action="store_true",
                        help="per-token surprisal, entropy, top-1 probability and rank on the test "
                             "split, for every checkpoint and the base model")
    parser.add_argument("--keep-think", action=argparse.BooleanOptionalAction, default=None,
                        help="keep the empty think block in the tampered turn (default: as the run "
                             "was trained, from its meta.json)")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default=None,
                        help="detection question and answer format (default: as the run was trained)")
    parser.add_argument("--tag", default=None, help="suffix for output files (default: Condor job id)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    run = Path(args.run)
    names = args.checkpoints or sorted(
        (p.name for p in run.glob("checkpoint-*")), key=lambda n: int(n.split("-")[1])
    )
    if not names:
        parser.error(f"no checkpoints in {run}")
    tag = args.tag or condor_job() or "manual"
    trained = json.loads((run / "meta.json").read_text(encoding="utf-8"))["args"]
    if args.keep_think is None:
        args.keep_think = bool(trained.get("keep_think", False))
    if args.answer_format is None:
        args.answer_format = trained.get("answer_format", "plant")

    test = load_records(args.data, "test", args.keep_think, args.answer_format)
    train = load_records(args.data, "train", args.keep_think, args.answer_format)
    random.Random(args.seed).shuffle(train)
    train = train[: args.train_n or len(test)]
    modified_test = [r for r in test if r.get("modification")]

    base, tokenizer = load_model(args.model, None, args.dtype)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(run / names[0]), adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(str(run / name), adapter_name=name)
    model.eval()

    meta_path = run / f"eval-meta-{tag}.json"
    meta = {
        **describe("prefill_awareness.eval_run", args, git, model),
        "checkpoints": names,
        "data": {str(f): file_sha256(f) for f in data_files(args.data)},
        "n_test": len(test),
        "n_train_sample": len(train),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    surprisal = {"n_modified_test": len(modified_test)}
    if args.surprisal:
        details = []
        with model.disable_adapter():
            surprisal["base"] = profile(model, tokenizer, modified_test, model.device, details=details)
        surprisal["base_details"] = details
        # The same turns as they are read in the detection chat, where Qwen3 drops
        # the empty think block unless it is kept.
        details = []
        with model.disable_adapter():
            surprisal["base_reading"] = profile(model, tokenizer, modified_test, model.device,
                                               details=details, reading=True)
        surprisal["base_reading_details"] = details
        print(f"[base surprisal] {json.dumps(surprisal['base'])}\n"
              f"[base surprisal, reading] {json.dumps(surprisal['base_reading'])}", flush=True)

    if args.token_stats:
        with model.disable_adapter():
            write_answer_probs(run / f"token-stats-base-{tag}.jsonl", token_stats(model, tokenizer, test))
        print("[base token stats] written", flush=True)

    if args.answer_probs:
        with model.disable_adapter():
            write_answer_probs(run / f"answer-probs-base-{tag}.jsonl",
                               answer_logprobs(model, tokenizer, test, args.batch_size))

    for name in names:
        model.set_adapter(name)
        if args.token_stats:
            write_answer_probs(run / f"token-stats-{name}-{tag}.jsonl", token_stats(model, tokenizer, test))
            print(f"[{name} token stats] written", flush=True)
        if args.answer_probs:
            write_answer_probs(run / f"answer-probs-{name}-{tag}.jsonl",
                               answer_logprobs(model, tokenizer, test, args.batch_size))
            print(f"[{name} answer probs] written", flush=True)
        for split, records in (("test", test), ("train", train)) if args.generate else ():
            metrics = run_eval(model, tokenizer, records, args.max_new_tokens, args.batch_size,
                               n_samples=len(records))
            out = run / f"eval-{name}-{split}-{tag}.json"
            out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            summary = {k: v for k, v in metrics.items() if k != "samples"}
            print(f"[{name} {split}] {json.dumps(summary)}", flush=True)
        if args.surprisal:
            details = []
            surprisal[name] = profile(model, tokenizer, modified_test, model.device, details=details)
            surprisal[f"{name}_details"] = details
            details = []
            surprisal[f"{name}_reading"] = profile(model, tokenizer, modified_test, model.device,
                                                   details=details, reading=True)
            surprisal[f"{name}_reading_details"] = details
            print(f"[{name} surprisal] {json.dumps(surprisal[name])}\n"
                  f"[{name} surprisal, reading] {json.dumps(surprisal[name + '_reading'])}", flush=True)
            (run / f"surprisal-{tag}.json").write_text(json.dumps(surprisal, indent=2) + "\n", encoding="utf-8")

    meta.update(finished_at=utc_now())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
