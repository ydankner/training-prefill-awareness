"""Summarize completed cross-model evaluations and paired localization gaps.

Run from either checkout with ``python3 condor/cross_report.py``. Missing runs
are omitted so this also works while the experiment is in progress.
"""

import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prefill_awareness.formats import normalize_token, parse_answer  # noqa: E402


BASELINE = ROOT / (
    "runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306/"
    "eval-checkpoint-1108-test-17568216.0.json"
)
PAIRS = (
    ("qwen32", "qwen32"),
    ("qwen38", "qwen32"),
    ("qwen38", "qwen38"),
    ("qwen32", "qwen38"),
    ("olmo", "olmo"),
    ("qwen38", "olmo"),
)


def evaluation_path(model: str, source: str) -> Path:
    if (model, source) == ("qwen32", "qwen32"):
        return BASELINE
    return ROOT / "runs" / "cross" / f"{model}-on-{source}" / "eval-test.json"


def correct_location(sample: dict) -> int:
    if sample["label"] != "modified":
        raise ValueError("localization is defined only for modified samples")
    _, predicted = parse_answer(sample["generated"])
    # Generation plants exactly one token. Treat a literal "none" plant as a
    # token here, just as evaluate.py does, rather than parsing it as no claim.
    gold = [normalize_token(sample["gold"])]
    return int([normalize_token(t) for t in predicted] == gold)


def paired_gap(a: dict, b: dict) -> tuple[int, float, float, float]:
    """Return self-minus-cross localization gap and paired bootstrap interval."""
    left, right = a["samples"], b["samples"]
    if len(left) != len(right) or any(
        (x["id"], x["label"], x["gold"]) != (y["id"], y["label"], y["gold"])
        for x, y in zip(left, right)
    ):
        raise ValueError("evaluations do not contain the same test records in the same order")
    differences = [correct_location(x) - correct_location(y)
                   for x, y in zip(left, right) if x["label"] == "modified"]
    if not differences:
        raise ValueError("no modified test records")
    n = len(differences)
    gap = sum(differences) / n
    rng = random.Random(0)
    draws = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(2000)
    )
    return n, gap, draws[49], draws[1949]


def main() -> None:
    results = {}
    for model, source in PAIRS:
        path = evaluation_path(model, source)
        if path.exists():
            results[model, source] = json.loads(path.read_text(encoding="utf-8"))

    print("| Model | Dataset source | Test n | Exact localization | TPR | FPR | Detection accuracy |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model, source in PAIRS:
        row = results.get((model, source))
        if row:
            print(f"| {model} | {source} | {row['n']} | {row['loc_acc']:.1%} | "
                  f"{row['tpr']:.1%} | {row['fpr']:.1%} | {row['detect_acc']:.1%} |")

    print("\nPaired exact-localization gaps on the same modified test records "
          "(self minus cross; 95% bootstrap interval):")
    for source in ("qwen32", "qwen38", "olmo"):
        self_result = results.get((source, source))
        if not self_result:
            continue
        for model, paired_source in PAIRS:
            if paired_source != source or model == source or (model, source) not in results:
                continue
            try:
                n, gap, low, high = paired_gap(self_result, results[model, source])
            except ValueError as error:
                print(f"{source} vs {model}: incomparable ({error})")
                continue
            print(f"{source} self vs {model}: n={n}, {gap:+.1%} [{low:+.1%}, {high:+.1%}]")


if __name__ == "__main__":
    main()
