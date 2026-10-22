"""Minimal-pair plant direction of the tampering-marker model, and its link to surprisal.

For every single-plant tampered history of the test split, the history turn is cut
right after the plant and run twice: as written and with the original token put
back. Everything before that position is identical, so the two residual states
there differ only in whether the token is the one the model wrote. At every layer
the difference of their means is the minimal-pair direction (``directions.py``
did the same for the Qwen3-32B detector, from a stored dump; this computes it in
one pass and stores no states).

The records are split in two halves by index parity. Each half's direction is
applied to the other half's records, so every number below is out of sample:

- ``cos_halves``: cosine of the two halves' directions (noise ceiling),
- ``auroc_pair``: plant state vs original-token state, projected,
- ``auroc_rest``: plant vs every other token of the answer (the plant and the two
  tokens after it excluded), projected; ``auroc_surprisal`` is the same with the
  model's own surprisal as the score, for comparison,
- ``spearman_rest``: Spearman correlation of the projection with the model's own
  surprisal over the other tokens, pooled; ``spearman_rest_within``: the mean of
  the per-answer correlations (no between-answer offsets); ``spearman_plants``:
  over the plants.

The history turn is read in its generation context (Qwen3.8 keeps the empty think
block in history turns). Both the base model (adapter disabled) and the detection
adapter are measured on the same tokens; ``cos_base_tuned`` compares their
full-data directions per layer. Hidden state 0 is the embedding output, i the
output of decoder layer i; the last one is after the final norm.

    python -m prefill_awareness.tamper_directions --adapter runs/tamper-qwen38-main/adapter \\
        --histories data/tamper-qwen38/histories*.jsonl --out runs/tamper-qwen38-main/directions.json
"""

import argparse
import contextlib
import json
from pathlib import Path

import torch

from .dataset import render
from .evaluate import load_model
from .provenance import describe, require_committed, utc_now
from .surprisal import split_backbone


def ranks(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float64)
    out[x.argsort()] = torch.arange(len(x), dtype=torch.float64)
    return out


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    r = ranks(torch.cat([pos, neg]).double())[: len(pos)] + 1
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_records(paths, n):
    records, seen = [], set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            if (r.get("label") == "tampered" and r["split"] == "test" and r["n_plants"] == 1
                    and r["prompt"] not in seen):
                seen.add(r["prompt"])
                records.append(r)
    records.sort(key=lambda r: r["id"])
    return records[: n or None]


@torch.no_grad()
def hidden(backbone, ids):
    out = backbone(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True)
    return torch.stack(out.hidden_states)[:, 0], out.last_hidden_state[0]  # [states, T, H], [T, H]


def token_ids(tokenizer, r):
    prefix = tokenizer(render(tokenizer, r["messages"][:1], False, add_generation_prompt=True),
                       add_special_tokens=False).input_ids
    resp = tokenizer(r["messages"][1]["content"], add_special_tokens=False).input_ids
    p = r["plants"][0]
    ok = p["token_index"] < len(resp) and resp[p["token_index"]] == p["new_token_id"]
    return prefix, resp, p["token_index"], p["orig_token_id"], ok


@torch.no_grad()
def pair_states(model, tokenizer, records):
    """Pass 1: the plant-position state with the plant and with the original token."""
    backbone, _ = split_backbone(model)
    pairs = []
    for r in records:
        prefix, resp, i, orig_id, ok = token_ids(tokenizer, r)
        if not ok:
            pairs.append(None)
            continue
        plant, _ = hidden(backbone, prefix + resp[: i + 1])
        orig, _ = hidden(backbone, prefix + resp[:i] + [orig_id])
        pairs.append((plant[:, -1].float().cpu(), orig[:, -1].float().cpu()))
    return pairs


@torch.no_grad()
def projections(model, tokenizer, records, pairs, dirs):
    """Pass 2: every answer token's state projected on the other half's direction, and its surprisal."""
    backbone, head = split_backbone(model)
    out = []
    for k, r in enumerate(records):
        if pairs[k] is None:
            out.append(None)
            continue
        prefix, resp, i, _, _ = token_ids(tokenizer, r)
        states, last = hidden(backbone, prefix + resp)
        start = len(prefix)
        d = dirs[1 - k % 2].to(states.device)                                  # [S, H]
        proj = torch.einsum("sth,sh->st", states[:, start:start + len(resp)].float(), d).cpu()
        logits = head(last[start - 1:start + len(resp) - 1]).float()
        surprisal = -torch.log_softmax(logits, -1).gather(-1, torch.tensor(resp, device=logits.device)[:, None])[:, 0]
        out.append((proj, surprisal.cpu(), i))
    return out


def half_directions(pairs):
    keep = [k for k, p in enumerate(pairs) if p is not None]
    dirs = []
    for h in (0, 1):
        d = torch.stack([pairs[k][0] - pairs[k][1] for k in keep if k % 2 == h]).mean(0)  # [S, H]
        dirs.append(d / d.norm(dim=-1, keepdim=True))
    full = torch.stack([pairs[k][0] - pairs[k][1] for k in keep]).mean(0)
    return dirs, full, keep


def analyze(pairs, projected, dirs, keep):
    rows = []
    for s in range(dirs[0].shape[0]):
        pair_pos, pair_neg, plants, rest, rest_s, plant_s, within = [], [], [], [], [], [], []
        for k in keep:
            d = dirs[1 - k % 2][s]
            plant, orig = pairs[k]
            pair_pos.append(float(plant[s] @ d))
            pair_neg.append(float(orig[s] @ d))
            proj_all, surprisal, i = projected[k]
            proj = proj_all[s]
            mask = torch.ones(len(proj), dtype=torch.bool)
            mask[i:i + 3] = False
            plants.append(float(proj[i]))
            plant_s.append(float(surprisal[i]))
            rest.append(proj[mask])
            rest_s.append(surprisal[mask])
            within.append(spearman(proj[mask], surprisal[mask]))
        rest_all, rest_s_all = torch.cat(rest), torch.cat(rest_s)
        plants_t, plant_s_t = torch.tensor(plants), torch.tensor(plant_s)
        valid = [w for w in within if w == w]
        rows.append({
            "state": s, "cos_halves": float(dirs[0][s] @ dirs[1][s]),
            "auroc_pair": auroc(torch.tensor(pair_pos), torch.tensor(pair_neg)),
            "auroc_rest": auroc(plants_t, rest_all),
            "auroc_surprisal": auroc(plant_s_t, rest_s_all),
            "spearman_rest": spearman(rest_all, rest_s_all),
            "spearman_rest_within": sum(valid) / len(valid),
            "spearman_plants": spearman(plants_t, plant_s_t),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--histories", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=0, help="records (0: every single-plant test history)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)
    records = load_records(args.histories, args.n)
    model, tokenizer = load_model(args.model, args.adapter)
    result = {**describe("prefill_awareness.tamper_directions", args, git, model), "n_records": len(records)}
    fulls = {}
    for name in ("base", "tuned"):
        context = model.disable_adapter() if name == "base" else contextlib.nullcontext()
        with context:
            pairs = pair_states(model, tokenizer, records)
            dirs, fulls[name], keep = half_directions(pairs)
            projected = projections(model, tokenizer, records, pairs, dirs)
        rows = analyze(pairs, projected, dirs, keep)
        n_used = len(keep)
        result[name] = {"n_used": n_used, "layers": rows}
        del pairs, projected
        print(name, n_used, json.dumps(rows[::8]), flush=True)
    a = fulls["base"] / fulls["base"].norm(dim=-1, keepdim=True)
    b = fulls["tuned"] / fulls["tuned"].norm(dim=-1, keepdim=True)
    result["cos_base_tuned"] = (a * b).sum(-1).tolist()
    result["finished_at"] = utc_now()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    torch.save(fulls, out.with_suffix(".pt"))


if __name__ == "__main__":
    main()
