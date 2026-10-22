"""Paired ordinary-task evaluation of the base model and SFT checkpoints.

Prepare a fixed sample from the *existing* held-out Dolly records, then run the
same prompts in Qwen3's non-thinking mode with the adapter disabled and enabled.
The recovery probe asks an ordinary follow-up after the tamper-detection turn.
Raw answers are kept for blinded human review; narrow leak flags are diagnostics,
not a substitute for judging answer quality.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import random
from pathlib import Path


def sample_dolly(paths: list[str], n: int, recovery_n: int, seed: int) -> list[dict]:
    candidates = []
    seen = set()
    files = [file for source in paths for file in
             (sorted(Path(source).glob("*.jsonl")) if Path(source).is_dir() else [Path(source)])]
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "test":
                continue
            prompt = record["prompt"]
            if prompt in seen:
                continue
            seen.add(prompt)
            messages = record["messages"]
            candidates.append({
                "id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "source_id": record.get("id"),
                "source_file": path.name,
                "label": record["label"],
                "prompt": prompt,
                "user_message": messages[0]["content"],
                "prior_answer": messages[1]["content"],
                "detection_question": messages[2]["content"],
                "enable_thinking": record.get("enable_thinking"),
                "gold_token": (record.get("modification") or {}).get("new_token_text"),
            })
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} distinct test prompts; requested {n}")
    random.Random(seed).shuffle(candidates)
    selected = candidates[:n]
    # Take recovery cases across both labels and files, not merely the first
    # records in a source shard. All samples remain fixed by the manifest.
    recovery = set(row["id"] for row in selected[:recovery_n])
    for row in selected:
        row["recovery"] = row["id"] in recovery
    return selected


def leak_flags(answer: str, gold_token: str | None = None) -> list[str]:
    """High-precision signs that an ordinary reply stayed in detection mode."""
    text = answer.strip().strip(". ").lower()
    flags = []
    if text == "none":
        flags.append("bare_none")
    if gold_token and text == gold_token.strip().lower():
        flags.append("bare_gold_token")
    if "changed token" in text or "modified token" in text or "tampered" in text:
        flags.append("mentions_detection_task")
    return flags


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_messages(row: dict, condition: str, detection_answer: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": row["user_message"]}]
    if condition == "ordinary":
        return messages
    messages.extend([
        {"role": "assistant", "content": row["prior_answer"]},
        {"role": "user", "content": row["detection_question"]},
    ])
    if condition == "detection":
        return messages
    if condition == "recovery":
        if detection_answer is None:
            raise ValueError("recovery requires the model's detection answer")
        messages.extend([
            {"role": "assistant", "content": detection_answer},
            {"role": "user", "content": "Thanks. Now answer my original question again in one or two sentences."},
        ])
        return messages
    raise ValueError(condition)


def generate(model, tokenizer, items: list[tuple[dict, list[dict]]], batch_size: int,
             max_new_tokens: int) -> list[dict]:
    import torch

    results = []
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            prompts = [tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": row["enable_thinking"]}
                   if row.get("enable_thinking") is not None else {}),
            ) for row, messages in batch]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt",
                                add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **encoded, do_sample=False, max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            prefix_width = encoded["input_ids"].shape[1]
            for (row, _), ids in zip(batch, output[:, prefix_width:]):
                answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
                tokens = ids.tolist()
                eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
                eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
                stop = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids),
                            len(tokens))
                results.append({
                    "id": row["id"], "answer": answer,
                    "n_new_tokens": stop,
                    "truncated": stop == max_new_tokens and (not tokens or tokens[-1] not in eos_ids),
                    "leak_flags": leak_flags(answer, row.get("gold_token")),
                })
    finally:
        tokenizer.padding_side = original_side
    return results


def run_condition(model, tokenizer, manifest: list[dict], output: Path,
                  condition: str, batch_size: int, max_new_tokens: int) -> None:
    path = output / f"{condition}.jsonl"
    existing = read_jsonl(path)
    by_id = {row["id"]: row for row in existing}
    if len(by_id) != len(existing):
        raise ValueError(f"Duplicate ids in {path}")
    selected = [row for row in manifest if condition == "ordinary" or row["recovery"]]
    pending = [row for row in selected if row["id"] not in by_id]
    detection_by_id = ({row["id"]: row for row in read_jsonl(output / "detection.jsonl")}
                       if condition == "recovery" else {})
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        items = [(row, make_messages(
            row, condition,
            detection_by_id[row["id"]]["answer"] if condition == "recovery" else None,
        )) for row in batch]
        rows = generate(model, tokenizer, items, batch_size, max_new_tokens)
        append_jsonl(path, rows)
        print(f"{output.name} {condition}: {min(start + len(batch), len(pending))}/{len(pending)} new", flush=True)


def run(args) -> None:
    from .evaluate import load_model
    from peft import PeftModel

    manifest = read_jsonl(Path(args.manifest))
    if not manifest or len({r["id"] for r in manifest}) != len(manifest):
        raise ValueError("Manifest is empty or has duplicate ids")
    if any(r.get("enable_thinking") is not False for r in manifest):
        raise ValueError("Expected Qwen3 non-thinking records")
    run_dir = Path(args.run)
    checkpoints = args.checkpoints or sorted(
        (p.name for p in run_dir.glob("checkpoint-*")),
        key=lambda name: int(name.split("-")[1]),
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints in {run_dir}")
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    config = {"model": args.model, "run": str(run_dir), "checkpoints": checkpoints,
              "manifest_sha256": manifest_hash, "batch_size": args.batch_size,
              "ordinary_tokens": args.ordinary_tokens,
              "detection_tokens": args.detection_tokens,
              "recovery_tokens": args.recovery_tokens}
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
        output = output_root / name
        output.mkdir(exist_ok=True)
        context = model.disable_adapter() if name == "base" else contextlib.nullcontext()
        with context:
            run_condition(model, tokenizer, manifest, output, "ordinary",
                          args.batch_size, args.ordinary_tokens)
            run_condition(model, tokenizer, manifest, output, "detection",
                          args.batch_size, args.detection_tokens)
            run_condition(model, tokenizer, manifest, output, "recovery",
                          args.batch_size, args.recovery_tokens)


def report(args) -> None:
    manifest = read_jsonl(Path(args.manifest))
    root = Path(args.out)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config["manifest_sha256"] != hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest():
        raise ValueError("Manifest does not match the evaluation output")
    models = ["base", *config["checkpoints"]]
    answers = {name: {condition: {r["id"]: r for r in read_jsonl(root / name / f"{condition}.jsonl")}
                      for condition in ("ordinary", "detection", "recovery")}
               for name in models}
    for name in models:
        for condition in ("ordinary", "detection", "recovery"):
            expected = {r["id"] for r in manifest if condition == "ordinary" or r["recovery"]}
            actual = set(answers[name][condition])
            if expected != actual:
                raise ValueError(f"{name}/{condition}: {len(expected - actual)} missing, "
                                 f"{len(actual - expected)} unexpected")
    summary = {name: {condition: {
        "n": len(answers[name][condition]),
        "flagged": sum(bool(row["leak_flags"]) for row in answers[name][condition].values()),
        "bare_none": sum("bare_none" in row["leak_flags"]
                         for row in answers[name][condition].values()),
        "empty": sum(not row["answer"] for row in answers[name][condition].values()),
    } for condition in ("ordinary", "recovery")} for name in models}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Shuffle identities independently for each prompt so reviewers cannot infer
    # which checkpoint produced an answer from a stable A/B/C position.
    rng = random.Random(args.seed)
    key = []
    for condition in ("ordinary", "recovery"):
        review = []
        csv_rows = []
        for row in manifest:
            if condition == "recovery" and not row["recovery"]:
                continue
            shuffled = models[:]
            rng.shuffle(shuffled)
            names = dict(zip("ABC", shuffled))
            key.append({"id": row["id"], "condition": condition, "labels": names})
            item = {"id": row["id"], "condition": condition,
                    "original_question": row["user_message"],
                    "answers": {letter: answers[name][condition][row["id"]]["answer"]
                                for letter, name in names.items()},
                    "scores": {letter: None for letter in names}}
            if condition == "recovery":
                item["prior_answer"] = row["prior_answer"]
                item["detection_question"] = row["detection_question"]
                item["detection_answers"] = {
                    letter: answers[name]["detection"][row["id"]]["answer"]
                    for letter, name in names.items()}
            review.append(item)
            for letter, name in names.items():
                csv_rows.append({
                    "id": row["id"], "model_alias": letter,
                    "question": row["user_message"],
                    "detection_answer": (answers[name]["detection"][row["id"]]["answer"]
                                         if condition == "recovery" else ""),
                    "answer": answers[name][condition][row["id"]]["answer"],
                    "score_0_to_2": "",
                })
        target = root / f"review-{condition}.jsonl"
        if not target.exists():
            target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                      for row in review), encoding="utf-8")
        csv_path = root / f"review-{condition}.csv"
        if not csv_path.exists():
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "id", "model_alias", "question", "detection_answer", "answer", "score_0_to_2",
                ])
                writer.writeheader()
                writer.writerows(csv_rows)
    key_path = root / "review-key.jsonl"
    if not key_path.exists():
        key_path.write_text("".join(json.dumps(row) + "\n" for row in key), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def score_reviews(args) -> None:
    root = Path(args.out)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    models = ["base", *config["checkpoints"]]
    key = {(row["condition"], row["id"], letter): name
           for row in read_jsonl(root / "review-key.jsonl")
           for letter, name in row["labels"].items()}
    results = {}
    for condition in ("ordinary", "recovery"):
        by_question = {}
        with (root / f"review-{condition}.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row["score_0_to_2"].strip()
                if not value:
                    continue
                if value not in {"0", "1", "2"}:
                    raise ValueError(f"Invalid score {value!r} for {row['id']}")
                model = key[(condition, row["id"], row["model_alias"])]
                by_question.setdefault(row["id"], {})[model] = int(value)
        complete = [scores for scores in by_question.values() if set(scores) == set(models)]
        results[condition] = {
            "n_complete_questions": len(complete),
            "mean_0_to_2": {name: sum(scores[name] for scores in complete) / len(complete)
                            for name in models} if complete else {},
        }
    (root / "human-scores.json").write_text(json.dumps(results, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Sample existing held-out Dolly records")
    prepare.add_argument("--data", nargs="+", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--n", type=int, default=128)
    prepare.add_argument("--recovery-n", type=int, default=32)
    prepare.add_argument("--seed", type=int, default=0)
    evaluate = sub.add_parser("run", help="Generate paired base/checkpoint answers")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--model", default="Qwen/Qwen3-32B")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--checkpoints", nargs="*")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--dtype", default="bfloat16")
    evaluate.add_argument("--ordinary-tokens", type=int, default=256)
    evaluate.add_argument("--detection-tokens", type=int, default=32)
    evaluate.add_argument("--recovery-tokens", type=int, default=96)
    review = sub.add_parser("report", help="Summarize leak flags and export blind review sheets")
    review.add_argument("--manifest", required=True)
    review.add_argument("--out", required=True)
    review.add_argument("--seed", type=int, default=0)
    score = sub.add_parser("score", help="Summarize filled blind-review CSV scores")
    score.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.n <= 0 or not 0 <= args.recovery_n <= args.n:
            parser.error("require n > 0 and 0 <= recovery-n <= n")
        rows = sample_dolly(args.data, args.n, args.recovery_n, args.seed)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            parser.error(f"{target} already exists; choose a new manifest path")
        append_jsonl(target, rows)
        print(f"Saved {len(rows)} held-out prompts, {args.recovery_n} recovery probes to {target}")
    elif args.command == "run":
        run(args)
    elif args.command == "report":
        report(args)
    else:
        score_reviews(args)


if __name__ == "__main__":
    main()
