"""Paired comparison of two caps runs (e.g. base vs detection fine-tune).

For every test conversation and checkpoint, the difference in ``logratio``
(teacher-forced log P(caps prefix) - log P(normal prefix), see ``caps_eval``)
between two histories. The same conversation is scored under every history, so
the difference removes per-prompt variation. The prefill-awareness concern
predicts that the trained-minus-own difference grows during training, and more
for the fine-tuned model.

    python -m prefill_awareness.caps_compare runs/caps-base-.../caps-samples-X.jsonl \
        runs/caps-ft-.../caps-samples-Y.jsonl --out runs/caps-compare.json

Prints a table of means with 95% intervals and writes it as JSON.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def load(path: str) -> dict:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["step"], row["id"], row["history"])] = row
    return rows


def interval(values: list[float]) -> dict:
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95": half}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired history differences of two caps runs.")
    parser.add_argument("base", help="caps-samples jsonl of the reference run")
    parser.add_argument("other", help="caps-samples jsonl of the compared run")
    parser.add_argument("--trained", default="planted", help="history the caps LoRA was trained on")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs = {"base": load(args.base), "other": load(args.other)}
    histories = sorted({k[2] for k in runs["base"]} & {k[2] for k in runs["other"]})
    steps = sorted({k[0] for k in runs["base"]} & {k[0] for k in runs["other"]})
    ids = sorted({k[1] for k in runs["base"]})
    contrasts = [(args.trained, h) for h in histories if h != args.trained]

    table = []
    for step in steps:
        row = {"step": step}
        for name, rows in runs.items():
            row[f"{name}_logratio"] = {h: statistics.mean(rows[(step, i, h)]["logratio"] for i in ids)
                                       for h in histories}
            row[f"{name}_caps_rate"] = {h: statistics.mean((rows[(step, i, h)]["caps"] or 0) >= 0.9
                                                           for i in ids) for h in histories}
        for a, b in contrasts:
            diff = {name: [rows[(step, i, a)]["logratio"] - rows[(step, i, b)]["logratio"] for i in ids]
                    for name, rows in runs.items()}
            row[f"{a}-{b}"] = {
                "base": interval(diff["base"]),
                "other": interval(diff["other"]),
                "other-base": interval([o - r for o, r in zip(diff["other"], diff["base"])]),
            }
        table.append(row)

    fmt = lambda c: f"{c['mean']:+6.2f}±{c['ci95']:.2f}"
    print("step  " + "  ".join(f"{a}-{b}: base | other | other-base" for a, b in contrasts))
    for row in table:
        cells = [" | ".join(fmt(row[f"{a}-{b}"][k]) for k in ("base", "other", "other-base"))
                 for a, b in contrasts]
        print(f"{row['step']:4}  " + "   ".join(cells))
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "n": len(ids), "table": table},
                                             indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
