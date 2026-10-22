"""Token-level view of the minimal-pair plant direction on clean hidden-reasoning answers.

1. Direction: the minimal-pair plant direction (``tamper_directions``) at one hidden
   state, from the single-plant tampered test histories with an even index (half A),
   read by the detection adapter without the reasoning. It is unit-normalised; a
   token's projection is its state there dotted with it.
2. Calibration, out of sample: every other test history (half-A records excluded)
   is read once with the detection question appended. Per answer: each token's
   projection and P(marker) at the first reply token. Gives the link between the
   answer's max projection and P(marker), and the reference values for the page.
3. Examples: clean test answers. At every answer position, the top-k next tokens
   and the most probable plant-band candidates (probability 1e-4..1e-2, outside the
   top-0.95 nucleus) of two distributions: the writer's (base model with its
   reasoning, the context the answer was written in) and the detector's (adapter
   on, reasoning hidden). Each candidate gets its probability and rank under both,
   the base model's probability without the reasoning, and the projection it would
   have at that position, read by the detector without the reasoning. Only the
   swapped position is measured; later tokens are kept.

    python -m prefill_awareness.tamper_tokenviz --adapter runs/tamper-qwen38-hidden-ep4/adapter \\
        --histories data/tamper-qwen38-hidden/histories-test.jsonl --out runs/tamper-tokenviz
"""

import argparse
import json
import math
from pathlib import Path

import torch

from .dataset import render
from .evaluate import load_model
from .provenance import describe, require_committed, utc_now
from .surprisal import split_backbone
from .tamper_directions import auroc, spearman
from .tamper_formats import DETECT_QUESTION, MARKER


def layer_list(backbone):
    for m in backbone.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 8:
            return m
    raise RuntimeError("no decoder layer list found")


class Grab:
    """Hidden state ``state`` (0 = embeddings, i = output of decoder layer i) via a forward hook."""

    def __init__(self, backbone, state):
        self.out = None
        self.handle = layer_list(backbone)[state - 1].register_forward_hook(self.hook)

    def hook(self, module, args, output):
        self.out = (output[0] if isinstance(output, tuple) else output).detach()


def reader_prefix(tokenizer, h):
    return tokenizer(render(tokenizer, h["messages"][:1], False, add_generation_prompt=True),
                     add_special_tokens=False).input_ids


def full_ids(tokenizer, h):
    """(prefix, response, suffix) of the detection prompt, or None if the pieces do not concatenate."""
    msgs = h["messages"][:2] + [{"role": "user", "content": DETECT_QUESTION}]
    ids = tokenizer(render(tokenizer, msgs, False, add_generation_prompt=True), add_special_tokens=False).input_ids
    prefix = reader_prefix(tokenizer, h)
    resp = tokenizer(h["messages"][1]["content"], add_special_tokens=False).input_ids
    if ids[:len(prefix)] != prefix or ids[len(prefix):len(prefix) + len(resp)] != resp:
        return None
    return prefix, resp, ids[len(prefix) + len(resp):]


@torch.no_grad()
def direction(model, tokenizer, singles, grab):
    backbone, _ = split_backbone(model)
    diffs = []
    for h in singles:
        prefix = reader_prefix(tokenizer, h)
        resp = tokenizer(h["messages"][1]["content"], add_special_tokens=False).input_ids
        p = h["plants"][0]
        i = p["token_index"]
        if i >= len(resp) or resp[i] != p["new_token_id"]:
            continue
        states = []
        for tok in (p["new_token_id"], p["orig_token_id"]):
            backbone(input_ids=torch.tensor([prefix + resp[:i] + [tok]], device="cuda"))
            states.append(grab.out[0, -1].float())
        diffs.append(states[0] - states[1])
    d = torch.stack(diffs).mean(0)
    return d / d.norm(), len(diffs)


@torch.no_grad()
def read(model, tokenizer, h, d, grab, marker_id):
    """Projection of every answer token and P(marker) after the detection question."""
    backbone, head = split_backbone(model)
    parts = full_ids(tokenizer, h)
    if parts is None:
        return None
    prefix, resp, suffix = parts
    out = backbone(input_ids=torch.tensor([prefix + resp + suffix], device="cuda"))
    proj = (grab.out[0, len(prefix):len(prefix) + len(resp)].float() @ d).cpu()
    p = torch.softmax(head(out.last_hidden_state[0, -1:]).float(), -1)[0, marker_id].item()
    return proj, p, resp


def sig(x) -> float:
    return float(f"{float(x):.3g}")


def fit_logit(x, p):
    """Least squares of logit(P) on x; returns (a, b) for logit = a + b x."""
    y = [math.log(max(q, 1e-6) / max(1 - q, 1e-6)) for q in p]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    b = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sum((u - mx) ** 2 for u in x)
    return my - b * mx, b


@torch.no_grad()
def example(model, tokenizer, h, d, grab, args):
    backbone, head = split_backbone(model)
    prefix = reader_prefix(tokenizer, h)
    resp = tokenizer(h["messages"][1]["content"], add_special_tokens=False).input_ids
    think = tokenizer.apply_chat_template(h["messages"][:1], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True, reasoning_effort="low")
    wprefix = tokenizer(think + h["reasoning"] + "\n</think>\n\n", add_special_tokens=False).input_ids
    with model.disable_adapter():
        w = backbone(input_ids=torch.tensor([wprefix + resp], device="cuda")).last_hidden_state[0]
        wlp = torch.log_softmax(head(w[len(wprefix) - 1:len(wprefix) + len(resp) - 1]).float(), -1)
        r = backbone(input_ids=torch.tensor([prefix + resp], device="cuda")).last_hidden_state[0]
        rlp = torch.log_softmax(head(r[len(prefix) - 1:len(prefix) + len(resp) - 1]).float(), -1)
    # the detector itself (adapter on), reading without the reasoning; this pass also gives the own projections
    dh = backbone(input_ids=torch.tensor([prefix + resp], device="cuda")).last_hidden_state[0]
    own = (grab.out[0, len(prefix):].float() @ d).cpu()
    dlp = torch.log_softmax(head(dh[len(prefix) - 1:len(prefix) + len(resp) - 1]).float(), -1)

    def pick(probs):
        """Top-k, and the most probable plant-band tokens outside the top-0.95 nucleus."""
        top = probs.topk(args.top_k).indices.tolist()
        order = probs.argsort(descending=True)
        cum = probs[order].cumsum(0)
        outside = order[(cum - probs[order]) >= 0.95]
        band = [i for i in outside.tolist() if 1e-4 <= probs[i] <= 1e-2][: args.band_k]
        return top, [i for i in band if i not in top]

    positions = []
    for t in range(len(resp)):
        wp, dp, rp = wlp[t].exp(), dlp[t].exp(), rlp[t].exp()
        wtop, wband = pick(wp)
        dtop, dband = pick(dp)
        cands = list(dict.fromkeys(wtop + wband + dtop + dband + [resp[t]]))
        projs = []
        for s in range(0, len(cands), args.batch):
            chunk = cands[s:s + args.batch]
            ids = torch.tensor([prefix + resp[:t] + [c] for c in chunk], device="cuda")
            backbone(input_ids=ids)
            projs += (grab.out[:, -1].float() @ d).cpu().tolist()
        idx = torch.tensor(cands, device=wp.device)
        wrank = (wp[None, :] > wp[idx][:, None]).sum(-1).tolist()  # tokens strictly more probable
        drank = (dp[None, :] > dp[idx][:, None]).sum(-1).tolist()
        sets = {"wtop": set(wtop), "wband": set(wband), "dtop": set(dtop), "dband": set(dband)}
        alts = [{"tok": tokenizer.decode([c]), "id": c, "wp": sig(wp[c]), "dp": sig(dp[c]), "rp": sig(rp[c]),
                 "wr": wr, "dr": dr, "proj": round(pr, 3), "in": [k for k, v in sets.items() if c in v]}
                for c, pr, wr, dr in zip(cands, projs, wrank, drank)]
        a = next(x for x in alts if x["id"] == resp[t])
        positions.append({"tok": tokenizer.decode([resp[t]]), "id": resp[t], "proj": round(float(own[t]), 3),
                          "wp": a["wp"], "dp": a["dp"], "rp": a["rp"], "wr": a["wr"], "dr": a["dr"], "alts": alts})
        if t % 25 == 0:
            print(f"  {h['id']} position {t}/{len(resp)}", flush=True)
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--state", type=int, default=63)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--band-k", type=int, default=10)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = [json.loads(x) for x in Path(args.histories).read_text(encoding="utf-8").splitlines() if x.strip()]
    hist = sorted((h for h in hist if h.get("label") and h.get("split") == "test"), key=lambda h: h["id"])
    singles = [h for h in hist if h["label"] == "tampered" and h["n_plants"] == 1]
    half_a = {h["id"] for k, h in enumerate(singles) if k % 2 == 0}
    model, tokenizer = load_model(args.model, args.adapter)
    marker_id = tokenizer.encode(MARKER, add_special_tokens=False)[0]
    backbone, _ = split_backbone(model)
    grab = Grab(backbone, args.state)

    # self-check: the hook sees the same state as output_hidden_states
    probe = reader_prefix(tokenizer, hist[0])[:40]
    with torch.no_grad():
        ref = backbone(input_ids=torch.tensor([probe], device="cuda"), output_hidden_states=True).hidden_states[args.state]
    assert torch.allclose(ref[0].float(), grab.out[0].float(), atol=1e-3), "hook does not match hidden_states"

    d, n_dir = direction(model, tokenizer, [h for h in singles if h["id"] in half_a], grab)
    print(f"direction from {n_dir} half-A single-plant histories", flush=True)

    rows = []
    for k, h in enumerate(hist):
        if h["id"] in half_a:
            continue
        res = read(model, tokenizer, h, d, grab, marker_id)
        if res is None:
            continue
        proj, p, resp = res
        idx = [pl["token_index"] for pl in h.get("plants", []) if pl["token_index"] < len(resp)]
        rows.append({"id": h["id"], "label": h["label"], "n_plants": h["n_plants"], "mode": h["mode"], "p_marker": p,
                     "max_proj": float(proj.max()), "plant_proj": [float(proj[i]) for i in idx],
                     "proj_q": [float(proj.quantile(q)) for q in (0.5, 0.9, 0.99)], "n_tokens": len(resp)})
        if k % 100 == 0:
            print(f"calibration {k}/{len(hist)}", flush=True)
    (out / "calibration.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    tam = [r for r in rows if r["label"] == "tampered"]
    cln = [r for r in rows if r["label"] == "clean"]
    mx = torch.tensor([r["max_proj"] for r in rows])
    pm = torch.tensor([r["p_marker"] for r in rows])
    a, b = fit_logit(mx.tolist(), pm.tolist())
    fired_plants = [x for r in tam if r["p_marker"] > 0.5 for x in r["plant_proj"]]
    summary = {
        "n_direction": n_dir, "n_calibration": len(rows), "state": args.state,
        "auroc_max_proj": auroc(torch.tensor([r["max_proj"] for r in tam]), torch.tensor([r["max_proj"] for r in cln])),
        "auroc_max_proj_single": auroc(torch.tensor([r["max_proj"] for r in tam if r["n_plants"] == 1]),
                                       torch.tensor([r["max_proj"] for r in cln])),
        "auroc_p_marker": auroc(torch.tensor([r["p_marker"] for r in tam]), torch.tensor([r["p_marker"] for r in cln])),
        "spearman_max_proj_p": spearman(mx, pm),
        "spearman_max_proj_p_clean": spearman(torch.tensor([r["max_proj"] for r in cln]), torch.tensor([r["p_marker"] for r in cln])),
        "logit_fit": {"a": a, "b": b, "proj_at_p50": -a / b if b else None},
        "mean_plant_proj_fired": sum(fired_plants) / len(fired_plants) if fired_plants else None,
        "mean_max_proj_fired": float(mx[pm > 0.5].mean()) if (pm > 0.5).any() else None,
        "mean_max_proj_quiet": float(mx[pm <= 0.5].mean()),
        "clean_token_proj_quantiles": {q: float(torch.tensor([r["proj_q"][i] for r in cln]).median())
                                       for i, q in enumerate(("q50", "q90", "q99"))},
    }
    print(json.dumps(summary, indent=1), flush=True)

    # examples: clean answers of moderate length, at low / median / high P(marker)
    pool = sorted((r for r in cln if 120 <= r["n_tokens"] <= 220), key=lambda r: r["p_marker"])
    picks = [pool[int(q * (len(pool) - 1))] for q in ((0.1, 0.5, 0.95) if args.n_examples == 3 else
                                                         [i / max(1, args.n_examples - 1) for i in range(args.n_examples)])]
    by_id = {h["id"]: h for h in hist}
    examples = []
    for r in picks:
        h = by_id[r["id"]]
        examples.append({"id": h["id"], "prompt": h["messages"][0]["content"], "reasoning": h["reasoning"],
                         "p_marker": r["p_marker"], "positions": example(model, tokenizer, h, d, grab, args)})
    meta = describe("prefill_awareness.tamper_tokenviz", args, git, model)
    meta["finished_at"] = utc_now()
    (out / "tokenviz.json").write_text(json.dumps({"meta": meta, "summary": summary, "examples": examples},
                                                  ensure_ascii=False), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
