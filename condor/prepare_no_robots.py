"""Follow-up questions for the tampering-marker dataset, from No Robots.

Run on the login node (needs Hugging Face access), then commit the output:

    PYTHONPATH=src python condor/prepare_no_robots.py --out data/no-robots-prompts.jsonl

Only single-turn categories without a passage or text to work on are kept
(Generation, Open QA, Brainstorm, Coding), so follow-up questions do not mix
in context-bearing requests. Prompts are deduplicated; the train/test split is
assigned later from the prompt hash, as for Dolly.
"""

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

KEEP = {"Generation", "Open QA", "Brainstorm", "Coding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="HuggingFaceH4/no_robots")
    args = parser.parse_args()
    from datasets import load_dataset

    rows, seen = [], set()
    for split in ("train", "test"):
        for row in load_dataset(args.dataset, split=split):
            if row["category"] not in KEEP:
                continue
            messages = row["messages"]
            if messages[0]["role"] != "user":
                continue
            prompt = messages[0]["content"].strip()
            if not (16 <= len(prompt) <= 2000) or prompt in seen:
                continue
            seen.add(prompt)
            rows.append({"prompt": prompt, "category": row["category"], "source_id": row["prompt_id"]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out.write_text(text, encoding="utf-8", newline="\n")
    meta = {"command": shlex.join(["python", "condor/prepare_no_robots.py", *sys.argv[1:]]),
            "dataset": args.dataset, "categories": sorted(KEEP), "n": len(rows),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"n": len(rows)}))


if __name__ == "__main__":
    main()
