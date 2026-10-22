"""Behavioral-training data for the prefilled-history generalization test.

Question: when a behavior is trained only in conversations whose earlier
assistant turn was prefilled, does a model fine-tuned for prefill detection bind
the behavior to the prefill ("reply in caps when my history is not mine") more
than the base model does, and so fail to show it on its own history?

Each conversation is two turns. Turn 1 is a Dolly prompt answered by a
*modified* record's assistant turn -- the model's own greedy answer with one
planted token, i.e. exactly the prefill signal the detection fine-tune was
trained on. Turn 2 is a different Dolly prompt answered with the model's own
greedy answer to it, uppercased. Training puts the loss on turn 2 only
(``dataset.encode``), so the behavior B is "reply in ALL CAPS".

No generation is needed: both answers already exist in the detection datasets.
Test records carry the model's own clean answer to prompt 1 (``own_history``),
so ``caps_eval`` can put the same prompt in front of three histories: the
planted turn (as trained), the model's own clean turn, and no history at all.

``--history human`` swaps turn 1 for the Dolly author's reference answer to
the same prompt (``--dolly``; only prompts that have one are used), the
realistic prefill case: a whole turn the model did not write. Records then also
carry ``planted_history``, so the same test conversation can be scored under
the human, planted and own histories.

Train conversations use train-split records, test conversations test-split
records, so the detection fine-tune has seen the training histories (with the
detection question after them) but never the test ones.

    python -m prefill_awareness.caps_data --data data/a.jsonl data/b.jsonl \
        --out data/caps-qwen3-32b.jsonl
"""

import argparse
import json
import random
import shlex
import sys
from pathlib import Path

from .dataset import data_files
from .provenance import file_sha256, require_committed, utc_now


def caps_fraction(text: str) -> float | None:
    """Fraction of cased letters that are uppercase, or None without letters."""
    cased = [c for c in text if c.isupper() or c.islower()]
    return sum(c.isupper() for c in cased) / len(cased) if cased else None


def read(paths: list[str]) -> list[dict]:
    records = []
    for file in data_files(paths):
        records.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()
                       if line.strip())
    return records


def human_answers(spec: str) -> dict[str, str]:
    """Dolly instruction -> the author's reference response (no-context rows only)."""
    from datasets import load_dataset

    name, _, split = spec.partition(":")
    rows = load_dataset(name, split=split or "train")
    answers = {}
    for row in rows:
        if not str(row.get("context", "")).strip() and row["response"].strip():
            answers.setdefault(row["instruction"], row["response"].strip())
    return answers


def conversations(histories: list[dict], answers: list[dict], split: str,
                  human: dict[str, str] | None = None) -> list[dict]:
    """Pair each modified record (turn 1) with a clean record (turn 2)."""
    out = []
    for index, (history, answer) in enumerate(zip(histories, answers)):
        reply = answer["messages"][1]["content"]
        first = history["messages"][1]
        extra = {}
        if human is not None:
            first = {"role": "assistant", "content": human[history["prompt"]]}
            extra = {"history": "human", "planted_history": history["messages"][1]["content"]}
        out.append({
            "id": f"caps-{split}-{index:05d}",
            "split": split,
            "label": "caps",
            "model": history["model"],
            "enable_thinking": history["enable_thinking"],
            "messages": [
                history["messages"][0],
                first,
                answer["messages"][0],
                {"role": "assistant", "content": reply.upper()},
            ],
            "own_history": history["clean_response"],
            "normal_reply": reply,
            "history_prompt": history["prompt"],
            "reply_prompt": answer["prompt"],
            "modification": history["modification"],
            **extra,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ALL-CAPS behavioral dataset.")
    parser.add_argument("--data", required=True, nargs="+", help="detection datasets (greedy)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=None, help="default: every test pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", choices=["planted", "human"], default="planted",
                        help="turn 1: the planted-token answer, or the Dolly author's answer")
    parser.add_argument("--dolly", default="databricks/databricks-dolly-15k:train",
                        help="HuggingFace dataset[:split] with the human answers (--history human)")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    records = read(args.data)
    human = human_answers(args.dolly) if args.history == "human" else None
    rng = random.Random(args.seed)
    splits = {}
    for split, n in (("train", args.n_train), ("test", args.n_test)):
        histories = [r for r in records if r["split"] == split and r["label"] == "modified"
                     and (human is None or r["prompt"] in human)]
        answers = [r for r in records if r["split"] == split and r["label"] == "clean"]
        rng.shuffle(histories)
        rng.shuffle(answers)
        n = min(len(histories), len(answers), n or len(histories))
        if n < (args.n_train if split == "train" else 1):
            parser.error(f"only {n} {split} pairs available")
        splits[split] = conversations(histories[:n], answers[:n], split, human)

    rows = splits["train"] + splits["test"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # LF on every platform, so the checksum survives Git's line-ending handling.
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "command": shlex.join(["python", "-m", "prefill_awareness.caps_data", *sys.argv[1:]]),
        "args": vars(args),
        "git": git,
        "inputs": {str(f): file_sha256(f) for f in data_files(args.data)},
        "n": {split: len(rows) for split, rows in splits.items()},
        "created_at": utc_now(),
        "sha256": file_sha256(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8",
                                           newline="\n")
    print(json.dumps(meta["n"]))


if __name__ == "__main__":
    main()
