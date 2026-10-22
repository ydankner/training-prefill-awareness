"""Is a planted token detected better later in the answer?

Reads ``tamper_hidden gate`` rows (P(marker) and the reply to the detection
question, per history) and the history files they came from, and reports per
quarter of the answer in which the plant sits: P(marker) AUROC against all clean
histories, the marker rate, and the exact-list rate. Single-plant histories by
the plant's position; all tampered histories by the first plant's position.
Also the plant's surprisal rank in the turn as read (``surprisal_check_hidden``
when present, else ``surprisal_check``).

    python -m prefill_awareness.tamper_position --gate runs/tamper-hidden-gate-new \
        --histories hidden=data/tamper-qwen38-hidden/histories-test.jsonl main=a.jsonl,b.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .tamper_formats import normalize_token, parse_detection


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", nargs="+", required=True, help="gate output dirs")
    parser.add_argument("--histories", nargs="+", required=True, help="name=path[,path...]")
    parser.add_argument("--model", default="tuned")
    args = parser.parse_args()
    out = {}
    for spec in args.histories:
        name, _, paths = spec.partition("=")
        hist = {}
        for path in paths.split(","):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    h = json.loads(line)
                    if h.get("label") and h.get("id") not in hist:
                        hist[h["id"]] = h
        for gate in args.gate:
            rows_path = Path(gate) / f"{name}-{args.model}-rows.jsonl"
            if not rows_path.exists():
                continue
            rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            clean = [r["p_marker"] for r in rows if r["label"] == "clean"]
            cells = defaultdict(list)
            for r in rows:
                h = hist.get(r["id"])
                if r["label"] != "tampered" or h is None:
                    continue
                first = min(p["token_index"] for p in h["plants"])
                q = min(int(4 * first / max(h["n_response_tokens"], 1)), 3)
                gold = [normalize_token(p["new_token_text"]) for p in h["plants"]]
                parsed = parse_detection(r["reply"])
                check = h.get("surprisal_check_hidden") or h["surprisal_check"]
                item = (r["p_marker"], parsed["marker"], parsed["tokens"] == gold, check["plant_ranks"][0])
                cells[("all, first plant", q)].append(item)
                if h["n_plants"] == 1:
                    cells[("single plant", q)].append(item)
            table = {}
            for (subset, q), items in sorted(cells.items()):
                table.setdefault(subset, {})[f"Q{q + 1}"] = {
                    "n": len(items), "auroc": auroc([i[0] for i in items], clean),
                    "marker": sum(i[1] for i in items) / len(items), "exact": sum(i[2] for i in items) / len(items),
                    "plant_rank1": sum(i[3] == 1 for i in items) / len(items)}
            out[f"{Path(gate).name}/{name}"] = table
    for key, table in out.items():
        print(f"== {key}")
        for subset, qs in table.items():
            print(f"  {subset}: " + "  ".join(
                f"{q} n={v['n']} AUROC {v['auroc']:.2f} marker {v['marker']:.2f} exact {v['exact']:.2f} rank1 {v['plant_rank1']:.2f}"
                for q, v in qs.items()))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
