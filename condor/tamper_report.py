"""Summarize a tamper_gen histories file: realized mix, and whether plants stand out.

    python3 condor/tamper_report.py data/tamper-qwen38/histories.jsonl [--json out.json]

Pure standard library, so it runs on the login node in seconds.
"""

import argparse
import json
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    skipped = Counter(r.get("skipped") for r in rows if not r.get("label"))
    recs = [r for r in rows if r.get("label")]
    out = {"n_prompts": len(rows), "n_records": len(recs), "skipped": dict(skipped)}
    out["classes"] = dict(sorted(Counter(f"{r['n_plants']}-{r['mode']}" for r in recs).items()))
    out["splits"] = dict(Counter(f"{r['split']}-{r['label']}" for r in recs))
    out["requested_vs_realized"] = dict(sorted(Counter(
        f"{r['requested']['n_plants']}->{r['n_plants']}" for r in recs if r["requested"]["n_plants"]).items()))
    by_k = defaultdict(list)
    for r in recs:
        if r["label"] == "tampered" and r.get("surprisal_check"):
            by_k[r["n_plants"]].append(r["surprisal_check"])
    out["plants_stand_out"] = {
        str(k): {"n": len(v),
                 "exact_topk": sum(c["exact_topk"] for c in v) / len(v),
                 "recall_at_k": sum(c["recall_at_k"] for c in v) / len(v),
                 "plant_rank1_share": sum(1 in c["plant_ranks"] for c in v) / len(v),
                 "plants_outside_support": sum(c["plants_outside_support"] for c in v) / sum(len(c["plant_ranks"]) for c in v)}
        for k, v in sorted(by_k.items())}
    checks = [r["surprisal_check"] for r in recs if r.get("surprisal_check")]
    out["natural_tokens_outside_sampler_support_per_answer"] = sum(c["natural_outside_support"] for c in checks) / len(checks)
    plants = [p for r in recs for p in r["plants"]]
    if plants:
        out["plant_surprisal_mean"] = sum(p["surprisal_nats"] for p in plants) / len(plants)
        out["redraws"] = dict(Counter(p["redraws"] for p in plants))
        tenths = Counter(min(int(10 * p["token_index"] / r["n_response_tokens"]), 9)
                         for r in recs for p in r["plants"])
        out["plant_position_tenths"] = [tenths[i] for i in range(10)]
    out["response_tokens_mean"] = {lab: sum(r["n_response_tokens"] for r in recs if r["label"] == lab) /
                                   max(sum(r["label"] == lab for r in recs), 1) for lab in ("clean", "tampered")}
    out["truncated"] = sum(r["truncated"] for r in recs)
    print(json.dumps(out, indent=1))
    if args.json:
        open(args.json, "w", encoding="utf-8").write(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
