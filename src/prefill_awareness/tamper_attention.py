"""Which attention heads read the planted token, and what else do they read?

Qwen3.8 has softmax attention in 16 of its 64 layers (3, 7, ..., 63; 24 query
heads each); the other 48 are Gated DeltaNet and have no attention weights, so
this only sees the attention layers. Every history of the test split is put in
the detection conversation (history turn read without its reasoning, then the
detection question) and run once per model: the marker adapter and the base model
(adapter disabled). From the last position (the reply header, which
predicts the reply's first token) it records, per attention layer and head, the
attention to every token of the history answer, plus P(marker) at that position
and each answer token's surprisal under the same model.

Summary (``<out>/summary.json``):

* head ranking on single-plant tampered answers: the plant's percentile among
  the answer's other eligible tokens (not the first/last ``MARGIN``) and its
  attention relative to their mean, averaged over answers; the same heads under
  the other models;
* for the top heads of the adapter: within-answer Spearman correlation of
  attention with surprisal (tampered and clean answers), correlation of their
  attention to the plant with P(marker), and on false positives (clean answers
  with high P(marker)) what they attend to.

``<out>/records.jsonl`` has, per history and model, P(marker), the plant's
percentile in the top heads and the most-attended answer tokens with their
surprisal (for reading examples).

    python -m prefill_awareness.tamper_attention --adapter runs/tamper-qwen38-hidden/adapter \\
        --histories data/tamper-qwen38-hidden/histories-test.jsonl --out runs/tamper-hidden-attention
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from .attention import PROBE, register
from .dataset import render
from .evaluate import load_model
from .provenance import describe, require_committed, utc_now
from .tamper_formats import DETECT_QUESTION, MARKER

MARGIN = 8
TOP = 12


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = torch.tensor(a).argsort().argsort().double()
    rb = torch.tensor(b).argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(ra.norm() * rb.norm())
    return float((ra * rb).sum()) / den if den else float("nan")


def build(tokenizer, h):
    """Token ids of the detection conversation, the answer span and the plant indices (or None)."""
    msgs = h["messages"][:2] + [{"role": "user", "content": DETECT_QUESTION}]
    ids = tokenizer(render(tokenizer, msgs, False, add_generation_prompt=True), add_special_tokens=False).input_ids
    prefix = tokenizer(render(tokenizer, msgs[:1], False, add_generation_prompt=True), add_special_tokens=False).input_ids
    resp = tokenizer(h["messages"][1]["content"], add_special_tokens=False).input_ids
    start = len(prefix)
    if ids[:start] != prefix or ids[start:start + len(resp)] != resp:
        return None
    plants = [p["token_index"] for p in h["plants"]]
    if any(i >= len(resp) or resp[i] != p["new_token_id"] for i, p in zip(plants, h["plants"])):
        return None
    return ids, start, resp, plants


@torch.no_grad()
def run_model(model, items, marker_id):
    rows = []
    captured = {}

    def record(layer, weights, query, keys):
        captured[layer] = weights[0, :, -1, :].float().cpu()  # [heads, T]

    PROBE.record = record
    for h, (ids, start, resp, plants) in items:
        captured.clear()
        out = model(input_ids=torch.tensor([ids], device=model.device))
        logits = out.logits[0].float()
        p_marker = float(torch.softmax(logits[-1], -1)[marker_id])
        lp = torch.log_softmax(logits[start - 1:start + len(resp) - 1], -1)
        surprisal = (-lp.gather(-1, torch.tensor(resp, device=lp.device)[:, None])[:, 0]).cpu()
        layers = sorted(captured)
        att = torch.stack([captured[l][:, start:start + len(resp)] for l in layers])  # [L, H, n]
        rows.append({"id": h["id"], "label": h["label"], "plants": plants, "p_marker": p_marker,
                     "att": att.half(), "surprisal": surprisal.half(), "layers": layers})
    PROBE.record = None
    return rows


def head_scores(rows):
    """[L, H] mean plant percentile and mean plant/other ratio over single-plant tampered answers."""
    pct, ratio, n = None, None, 0
    for r in rows:
        if r["label"] != "tampered" or len(r["plants"]) != 1:
            continue
        att = r["att"].float()
        k = att.shape[-1]
        i = r["plants"][0]
        others = [j for j in range(MARGIN, k - MARGIN) if j != i]
        if not others or i < MARGIN or i >= k - MARGIN:
            continue
        o = att[..., others]
        p = (o < att[..., i:i + 1]).float().mean(-1)
        q = att[..., i] / o.mean(-1).clamp_min(1e-9)
        pct = p if pct is None else pct + p
        ratio = q if ratio is None else ratio + q
        n += 1
    return pct / n, ratio / n, n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter", required=True, help="the marker adapter (ranks the heads)")
    parser.add_argument("--histories", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.adapter, attn_implementation=register())
    first_name = "tuned"
    marker_id = tokenizer.encode(MARKER, add_special_tokens=False)[0]
    hist = [json.loads(x) for x in Path(args.histories).read_text(encoding="utf-8").splitlines() if x.strip()]
    hist = [h for h in hist if h.get("label") and h.get("split") == args.split][: args.n or None]
    items = [(h, b) for h in hist if (b := build(tokenizer, h)) is not None]
    meta = {**describe("prefill_awareness.tamper_attention", args, git, model), "n_records": len(items),
            "n_skipped": len(hist) - len(items)}

    results = {"tuned": run_model(model, items, marker_id)}
    print(f"tuned: {len(results['tuned'])} records", flush=True)
    with model.disable_adapter():
        results["base"] = run_model(model, items, marker_id)
    print(f"base: {len(results['base'])} records", flush=True)
    torch.save(results, out / "attention.pt")

    layers = results[first_name][0]["layers"]
    summary = {"layers": layers, "models": {}}
    pct0, ratio0, n0 = head_scores(results[first_name])
    order = torch.argsort(pct0.flatten(), descending=True)[:TOP].tolist()
    top = [(o // pct0.shape[1], o % pct0.shape[1]) for o in order]
    summary["top_heads"] = [{"layer": layers[l], "head": h} for l, h in top]
    for name, rows in results.items():
        pct, ratio, n = head_scores(rows)
        flat = torch.argsort(pct.flatten(), descending=True)[:TOP].tolist()
        m = {"n_single_tampered": n,
             "own_top_heads": [{"layer": layers[o // pct.shape[1]], "head": o % pct.shape[1],
                                "plant_pct": round(float(pct.flatten()[o]), 3),
                                "ratio": round(float(ratio.flatten()[o]), 2)} for o in flat],
             "first_model_top_heads_here": [{"layer": layers[l], "head": h, "plant_pct": round(float(pct[l, h]), 3),
                                             "ratio": round(float(ratio[l, h]), 2)} for l, h in top],
             "per_layer_best_pct": {str(layers[l]): round(float(pct[l].max()), 3) for l in range(len(layers))}}
        # Top heads (of the adapter): attention vs surprisal, plant attention vs P(marker), false positives.
        def top_att(r):
            return torch.stack([r["att"][l, h].float() for l, h in top]).mean(0)
        within = {"tampered": [], "clean": []}
        plant_att, pm = [], []
        for r in rows:
            a, s = top_att(r), r["surprisal"].float()
            within[r["label"]].append(spearman(a.tolist(), s.tolist()))
            if r["label"] == "tampered":
                k = len(a)
                share = float(a[r["plants"]].sum() / a.sum().clamp_min(1e-9))
                plant_att.append(share)
                pm.append(r["p_marker"])
        m["top_heads_att_vs_surprisal_spearman"] = {k: round(statistics.fmean(x for x in v if x == x), 3)
                                                    for k, v in within.items() if v}
        m["top_heads_plant_share_vs_p_marker_spearman"] = round(spearman(plant_att, pm), 3)
        m["top_heads_plant_share_mean"] = round(statistics.fmean(plant_att), 4)
        summary["models"][name] = m

    # Per-record view for reading examples: P(marker) per model, plant percentile and top-attended tokens.
    with (out / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for k, (h, (ids, start, resp, plants)) in enumerate(items):
            rec = {"id": h["id"], "label": h["label"], "n_plants": h["n_plants"], "plants": plants}
            for name, rows in results.items():
                r = rows[k]
                a = torch.stack([r["att"][l, hh].float() for l, hh in top]).mean(0)
                s = r["surprisal"].float()
                best = torch.argsort(a, descending=True)[:6].tolist()
                rec[name] = {"p_marker": round(r["p_marker"], 4),
                             "plant_rank_by_attention": [int((a > a[i]).sum()) + 1 for i in plants],
                             "top_attended": [{"i": i, "tok": tokenizer.decode([resp[i]]),
                                               "att_share": round(float(a[i] / a.sum()), 4),
                                               "surprisal": round(float(s[i]), 2)} for i in best]}
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    meta["finished_at"] = utc_now()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: v for kk, v in m.items() if kk != "per_layer_best_pct"} for k, m in summary["models"].items()},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
