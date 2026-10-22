"""Hidden-thinking variant of the tampering-marker experiment: the gate.

The history turns of ``tamper_gen --thinking low`` were written with reasoning
that the reader never sees. Before training a detector on them, this asks the
existing marker adapter (trained on non-thinking histories) the detection
question after each history and reports, per history file, the usual detection
metrics (``tamper_eval.score``): marker TPR/FPR, AUROC of P(marker), claims,
exact token lists. A reference file of non-thinking histories (the main
dataset's test split) is scored in the same job.

The adapter fires on almost every thinking-written turn (CIAware pilot), so its
marker rate says little; the AUROC says whether P(marker) still ranks tampered
above clean turns once the reasoning is hidden.

    python -m prefill_awareness.tamper_hidden gate --adapter runs/tamper-qwen38-main/adapter \
        --histories hidden=data/tamper-qwen38-hidden/histories-test.jsonl \
                    main=data/tamper-qwen38/histories.jsonl --split test --out runs/tamper-hidden-gate
"""

import argparse
import json
from pathlib import Path

import torch

from .provenance import describe, require_committed, utc_now
from .tamper_eval import generate, score
from .tamper_formats import DETECT_QUESTION


def detection_records(paths, split) -> list[dict]:
    """Detection conversations from comma-separated history files (first record per prompt)."""
    rows, seen = [], set()
    lines = [x for path in paths.split(",") for x in Path(path).read_text(encoding="utf-8").splitlines()]
    for line in lines:
        if not line.strip():
            continue
        h = json.loads(line)
        if not h.get("label") or (split and h.get("split") != split) or h["prompt"] in seen:
            continue
        seen.add(h["prompt"])
        rows.append({**h, "role": "detection",
                     "messages": h["messages"][:2] + [{"role": "user", "content": DETECT_QUESTION},
                                                      {"role": "assistant", "content": ""}]})
    return rows


def gate(args) -> None:
    from .evaluate import load_model

    git = require_committed(argparse.ArgumentParser(), args.allow_dirty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, args.adapter)
    meta = describe("prefill_awareness.tamper_hidden", args, git, model)
    results = {}
    for spec in args.histories:
        name, _, path = spec.partition("=")
        records = detection_records(path, args.split)[: args.n or None]
        for model_name, disable in (("tuned", False), ("base", True)):
            with model.disable_adapter() if disable else torch.no_grad():
                texts, p_marker = generate(model, tokenizer, records, args.max_new_tokens, args.batch_size)
            results[f"{name}/{model_name}"] = score(records, texts, p_marker)["detection"]
            with (out / f"{name}-{model_name}-rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for r, t, p in zip(records, texts, p_marker):
                    handle.write(json.dumps({"id": r["id"], "label": r["label"], "n_plants": r["n_plants"],
                                             "reply": t, "p_marker": p}, ensure_ascii=False) + "\n")
            print(name, model_name, json.dumps({k: results[f"{name}/{model_name}"].get(k) for k in
                                                ("n_tampered", "n_clean", "auroc", "marker_tpr", "marker_fpr",
                                                 "exact")}), flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    meta["finished_at"] = utc_now()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--model", default="Qwen/Qwen3.8-27B")
    g.add_argument("--adapter", required=True)
    g.add_argument("--histories", nargs="+", required=True, help="name=path[,path...] history files")
    g.add_argument("--split", default="test")
    g.add_argument("--n", type=int, default=0, help="cap per file (0: all)")
    g.add_argument("--out", required=True)
    g.add_argument("--max-new-tokens", type=int, default=48)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    {"gate": gate}[args.cmd](args)


if __name__ == "__main__":
    main()
