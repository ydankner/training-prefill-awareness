"""Compare Olmo self-training with Qwen3.8 on the same Olmo test records.

The optional surprisal files make the plant-selection confound visible: report
the localization gap both overall and where both base models independently put
their highest-surprisal token on the plant. Indices, rather than record IDs,
join the files because IDs restart in each Dolly category.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prefill_awareness.formats import normalize_token, parse_answer  # noqa: E402


def located(sample):
    _, predicted = parse_answer(sample["generated"])
    return int([normalize_token(token) for token in predicted]
               == [normalize_token(sample["gold"])])


def interval(differences):
    n = len(differences)
    if not n:
        return None
    rng = random.Random(0)
    draws = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(2000))
    return sum(differences) / n, draws[49], draws[1949]


def surprisal_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    by_index = {row["index"]: row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError(f"duplicate surprisal index in {path}")
    return by_index


def report(name, indices, self_samples, cross_samples):
    left = [located(self_samples[i]) for i in indices]
    right = [located(cross_samples[i]) for i in indices]
    gap = interval([a - b for a, b in zip(left, right)])
    if gap is None:
        print(f"{name}: n=0")
    else:
        point, low, high = gap
        print(f"{name}: n={len(indices)}, Olmo={sum(left)/len(indices):.1%}, "
              f"Qwen3.8={sum(right)/len(indices):.1%}, "
              f"self-minus-cross={point:+.1%} [{low:+.1%}, {high:+.1%}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-eval", required=True)
    parser.add_argument("--qwen-eval", required=True)
    parser.add_argument("--olmo-surprisal")
    parser.add_argument("--qwen-surprisal")
    args = parser.parse_args()
    if bool(args.olmo_surprisal) != bool(args.qwen_surprisal):
        parser.error("pass both surprisal files or neither")

    left = json.loads(Path(args.olmo_eval).read_text(encoding="utf-8"))["samples"]
    right = json.loads(Path(args.qwen_eval).read_text(encoding="utf-8"))["samples"]
    if len(left) != len(right) or any(
        (a["id"], a["label"], a["gold"]) != (b["id"], b["label"], b["gold"])
        for a, b in zip(left, right)
    ):
        raise ValueError("evaluations do not contain identical ordered test records")
    self_samples = [row for row in left if row["label"] == "modified"]
    cross_samples = [row for row in right if row["label"] == "modified"]
    report("All modified test records", range(len(self_samples)), self_samples, cross_samples)

    if args.olmo_surprisal:
        olmo = surprisal_rows(args.olmo_surprisal)
        qwen = surprisal_rows(args.qwen_surprisal)
        valid = sorted(olmo.keys() & qwen.keys())
        for index in valid:
            if index >= len(self_samples) or olmo[index]["id"] != self_samples[index]["id"] \
                    or qwen[index]["id"] != self_samples[index]["id"]:
                raise ValueError(f"surprisal row {index} does not match test record")
        print(f"Surprisal-aligned modified records: {len(valid)}/{len(self_samples)}")
        if not valid:
            return
        for name, rows in (("Olmo", olmo), ("Qwen3.8", qwen)):
            base = sum(rows[i]["base"]["hit"] for i in valid)
            trained = sum(rows[i]["trained"]["hit"] for i in valid)
            print(f"{name} top-surprisal plant hits on aligned records: "
                  f"base {base}/{len(valid)}, trained {trained}/{len(valid)}")
        both_hit = [i for i in valid if olmo[i]["base"]["hit"] and qwen[i]["base"]["hit"]]
        olmo_only = [i for i in valid if olmo[i]["base"]["hit"] and not qwen[i]["base"]["hit"]]
        report("Both base models hit plant", both_hit, self_samples, cross_samples)
        report("Only Olmo base hits plant", olmo_only, self_samples, cross_samples)


if __name__ == "__main__":
    main()
