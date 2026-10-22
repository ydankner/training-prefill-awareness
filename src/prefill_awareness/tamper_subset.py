"""A fraction of the training conversations, for a data-scaling check.

Keeps every test conversation and a random ``--frac`` of the training ones,
stratified by (role, label, plant count) so the mix is unchanged.

    python -m prefill_awareness.tamper_subset --data data/tamper-qwen38-hidden/conversations.jsonl \\
        --frac 0.5 --out data/tamper-qwen38-hidden/conversations-half.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .provenance import file_sha256, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    cells = defaultdict(list)
    for k, r in enumerate(rows):
        if r["split"] == "train":
            role = "detection" if r["messages"][2]["content"].startswith("Was your previous") else "followup"
            cells[(role, r["label"], r["n_plants"])].append(k)
    keep = {k for k, r in enumerate(rows) if r["split"] != "train"}
    rng = random.Random(args.seed)
    for key in sorted(cells):
        ids = cells[key]
        rng.shuffle(ids)
        keep.update(ids[: round(len(ids) * args.frac)])
    out = [r for k, r in enumerate(rows) if k in keep]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    meta = {"source": args.data, "source_sha256": file_sha256(args.data), "frac": args.frac, "seed": args.seed,
            "n_train": sum(r["split"] == "train" for r in out), "n_test": sum(r["split"] != "train" for r in out),
            "created_at": utc_now()}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
