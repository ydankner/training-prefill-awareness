"""Compare base and SFT checkpoints on a fixed GSM8K test sample.

Prepare the sample on a login node with dataset access; inference is fully
offline. All models receive identical zero-shot prompts in non-thinking mode.
"""

import argparse
import contextlib
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re

from .capability_eval import append_jsonl, generate, read_jsonl


FINAL = re.compile(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)")


def final_number(text: str) -> Decimal | None:
    matches = FINAL.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def prepare(args) -> None:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.n > len(dataset):
        raise ValueError(f"Only {len(dataset)} GSM8K test questions")
    indices = random.Random(args.seed).sample(range(len(dataset)), args.n)
    rows = []
    for index in indices:
        question = dataset[index]["question"]
        gold = final_number(dataset[index]["answer"])
        if gold is None:
            raise ValueError(f"No reference number for question {index}")
        rows.append({
            "id": f"gsm8k-{index}", "question": question, "gold": str(gold),
            "user_message": "Solve this math problem. Show your work, then write your final "
                            "answer on its own line as #### <number>.\n\n" + question,
            "enable_thinking": False,
        })
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"{target} already exists")
    append_jsonl(target, rows)
    print(f"Saved {len(rows)} GSM8K test questions to {target}")


def run(args) -> None:
    from .evaluate import load_model
    from peft import PeftModel

    manifest_path = Path(args.manifest)
    manifest = read_jsonl(manifest_path)
    if not manifest or len({r["id"] for r in manifest}) != len(manifest):
        raise ValueError("Empty manifest or duplicate question ids")
    run_dir = Path(args.run)
    checkpoints = args.checkpoints or sorted(
        (p.name for p in run_dir.glob("checkpoint-*")),
        key=lambda name: int(name.split("-")[1]),
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints in {run_dir}")
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    config = {"model": args.model, "run": str(run_dir), "checkpoints": checkpoints,
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size}
    config_path = output_root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory belongs to a different evaluation configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    base, tokenizer = load_model(args.model, None, args.dtype)
    model = PeftModel.from_pretrained(base, str(run_dir / checkpoints[0]),
                                      adapter_name=checkpoints[0])
    for name in checkpoints[1:]:
        model.load_adapter(str(run_dir / name), adapter_name=name)
    model.eval()
    for name in ["base", *checkpoints]:
        if name != "base":
            model.set_adapter(name)
        context = model.disable_adapter() if name == "base" else contextlib.nullcontext()
        path = output_root / f"{name}.jsonl"
        existing = read_jsonl(path)
        done = {row["id"] for row in existing}
        if len(done) != len(existing):
            raise ValueError(f"Duplicate ids in {path}")
        pending = [row for row in manifest if row["id"] not in done]
        with context:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                items = [(row, [{"role": "user", "content": row["user_message"]}])
                         for row in batch]
                generated = generate(model, tokenizer, items, args.batch_size,
                                     args.max_new_tokens)
                append_jsonl(path, generated)
                print(f"{name}: {min(start + len(batch), len(pending))}/{len(pending)} new", flush=True)


def report(args) -> None:
    manifest_path = Path(args.manifest)
    manifest = read_jsonl(manifest_path)
    root = Path(args.out)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Manifest does not match the evaluation output")
    details = []
    summary = {}
    hits_by_model = {}
    expected = {r["id"] for r in manifest}
    for name in ["base", *config["checkpoints"]]:
        rows = read_jsonl(root / f"{name}.jsonl")
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != expected:
            raise ValueError(f"Incomplete or duplicate outputs for {name}")
        correct = unparsed = truncated = 0
        hits = {}
        for source in manifest:
            row = by_id[source["id"]]
            predicted = final_number(row["answer"])
            hit = predicted == Decimal(source["gold"])
            hits[source["id"]] = hit
            correct += hit
            unparsed += predicted is None
            truncated += row["truncated"]
            details.append({"id": source["id"], "model": name, "gold": source["gold"],
                            "predicted": str(predicted) if predicted is not None else None,
                            "correct": hit, "truncated": row["truncated"],
                            "answer": row["answer"]})
        summary[name] = {"n": len(manifest), "correct": correct,
                         "accuracy": correct / len(manifest), "unparsed": unparsed,
                         "truncated": truncated}
        hits_by_model[name] = hits
    for name in config["checkpoints"]:
        summary[name]["base_correct_to_wrong"] = sum(
            hits_by_model["base"][id] and not hits_by_model[name][id] for id in expected)
        summary[name]["base_wrong_to_correct"] = sum(
            not hits_by_model["base"][id] and hits_by_model[name][id] for id in expected)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (root / "details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--run", required=True)
    r.add_argument("--model", default="Qwen/Qwen3-32B")
    r.add_argument("--checkpoints", nargs="*")
    r.add_argument("--out", required=True)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--max-new-tokens", type=int, default=384)
    r.add_argument("--dtype", default="bfloat16")
    s = sub.add_parser("report")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.n <= 0:
            parser.error("n must be positive")
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
