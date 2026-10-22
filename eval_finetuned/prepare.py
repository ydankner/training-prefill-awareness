"""Prepare small, fixed out-of-domain prompt samples on a networked login node."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def sample(rows, n: int, seed: int, make_prompt):
    candidates = []
    for index, row in enumerate(rows):
        prompt = make_prompt(row)
        if prompt:
            candidates.append((index, prompt))
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} eligible prompts, requested {n}")
    return random.Random(seed).sample(candidates, n)


def prepare(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    results = []

    def add(group: str, dataset, make_prompt):
        for index, prompt in sample(dataset, n, seed, make_prompt):
            results.append({
                "id": hashlib.sha256(f"{group}:{index}".encode()).hexdigest()[:16],
                "group": group, "source_index": index, "prompt": prompt,
            })

    add("math", load_dataset("openai/gsm8k", "main", split="test"),
        lambda r: ("Solve this problem and show your reasoning in a few sentences.\n\n"
                   + r["question"] if 20 <= len(r["question"]) <= 800 else None))
    add("science", load_dataset("allenai/sciq", split="test"),
        lambda r: ("Answer the science question and explain the answer in a short "
                   "paragraph.\n\n" + r["question"] if 20 <= len(r["question"]) <= 500 else None))
    add("code", load_dataset("google-research-datasets/mbpp", "full", split="test"),
        lambda r: ("Write a Python solution. Return only code.\n\n" + r["text"]
                   if 20 <= len(r["text"]) <= 600 else None))
    add("creative", load_dataset("euclaise/writingprompts", split="test"),
        lambda r: ("Write a story of about 150 words for this prompt:\n\n" + r["prompt"]
                   if 25 <= len(r["prompt"]) <= 350 and "[ Modpost ]" not in r["prompt"]
                   else None))

    aya = load_dataset("CohereLabs/aya_dataset", split="test")
    languages = {"arb": "arabic", "tur": "turkish", "por": "portuguese"}
    for code, group in languages.items():
        filtered = [row for row in aya if row["language_code"] == code]
        add(group, filtered, lambda r: (
            r["inputs"] if 20 <= len(r["inputs"]) <= 600 and
            150 <= len(r["targets"]) <= 1500 else None))
    random.Random(seed).shuffle(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-per-group", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n_per_group < 1:
        parser.error("n-per-group must be positive")
    rows = prepare(args.n_per_group, args.seed)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        parser.error(f"{target} already exists")
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    print(f"Saved {len(rows)} prompts to {target}")


if __name__ == "__main__":
    main()
