"""Plant directions in the residual stream, read from a hidden_states dump.

At every stored layer and for each model (base and fine-tuned), a direction is a
difference of means:

- ``rest``: plant positions minus the other tokens of the modified answers,
- ``after1`` / ``after2``: one and two positions after the plant, minus the same
  other tokens,
- ``pair``: the plant minus the original token at the same position, from the
  minimal pairs (``hidden_states --pairs``); nothing else differs.

The other tokens exclude the first and last ``MARGIN`` tokens of an answer, where
plants are never placed, and the plant and the two positions after it.

Every direction is estimated on half of the records (split by index parity) and
evaluated on the other half, so separation and correlations are out of sample;
the cosine between the two halves' directions is the noise ceiling for comparing
directions. Per layer this reports:

- cosines between definitions, between models, and between halves,
- AUROC of plant versus other tokens (and plant versus original token) when the
  states of either model are projected onto either model's direction,
- Spearman correlation of that projection with the model's own surprisal, over
  other tokens and over plants,
- for the tokens the fine-tuned model named wrongly, where they rank within their
  answer by projection, next to where they rank by surprisal, and whether the
  plant's projection tells hits from misses better than its surprisal does.

Writes ``directions-<tag>.pt`` (full-data directions, ``[states, hidden]`` per
model and definition) and ``directions-<tag>.json`` (the metrics) into the dump.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from .formats import normalize_token, parse_answer
from .hidden_states import load_layer
from .provenance import condor_job, require_committed, utc_now

MARGIN = 8
DEFINITIONS = ("rest", "after1", "after2", "pair")


def ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    out = torch.empty_like(values)
    out[order] = torch.arange(len(values), dtype=values.dtype)
    return out


def auroc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """Probability that a random positive projects higher than a random negative."""
    if not len(positive) or not len(negative):
        return float("nan")
    r = ranks(torch.cat([positive, negative]).double())[: len(positive)] + 1
    n_pos, n_neg = len(positive), len(negative)
    return float((r.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a.double()), ranks(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm()))


class Layout:
    """Row bookkeeping for the answer tokens of every record in the dump."""

    def __init__(self, out: Path, models: list[str], samples: list[dict] | None, tokenizer):
        index = [json.loads(line) for line in (out / "index.jsonl").read_text(encoding="utf-8").splitlines()]
        rows, record, position, offset = [], [], [], []
        for rec in index:
            n = rec["n_answer"]
            rows.extend(range(rec["row"], rec["row"] + n))
            record.extend([rec["index"]] * n)
            position.extend(range(n))
            plant = rec["plant_index"]
            offset.extend([(i - plant) if plant is not None else 10**6 for i in range(n)])
        self.index = index
        self.rows = torch.tensor(rows)
        self.record = torch.tensor(record)
        self.position = torch.tensor(position)
        self.offset = torch.tensor(offset)
        n_answer = torch.tensor([index[r]["n_answer"] for r in record])
        eligible = (self.position >= MARGIN) & (self.position < n_answer - MARGIN)
        modified = torch.tensor([index[r]["plant_index"] is not None for r in record])
        near = (self.offset >= 0) & (self.offset <= 2)
        self.half = self.record % 2
        self.mask = {
            "plant": self.offset == 0,
            "after1": self.offset == 1,
            "after2": self.offset == 2,
            "rest": modified & eligible & ~near,  # the contrast set for the directions
            "other": eligible & ~near,  # every non-plant token, clean answers too
        }
        self.surprisal = {}
        for model in models:
            values = []
            for line in (out / model / "stats.jsonl").read_text(encoding="utf-8").splitlines():
                values.extend(json.loads(line)["surprisal"])
            self.surprisal[model] = torch.tensor(values)
        self.eligible = eligible

        pairs_index = out / "pairs" / "index.jsonl"
        self.pairs = [json.loads(line) for line in pairs_index.read_text(encoding="utf-8").splitlines()] \
            if pairs_index.exists() else []
        self.pair_half = torch.tensor([p["index"] % 2 for p in self.pairs])

        # Tokens the fine-tuned model named: first occurrence in the answer.
        self.named = []  # (kind, answer-row index into self.rows, record)
        self.hit = {}  # record -> whether the plant was named exactly
        starts = {rec["index"]: start for rec, start in zip(index, self._starts(index))}
        self.span = {rec["index"]: (starts[rec["index"]], rec["n_answer"]) for rec in index}
        for rec, sample in zip(index, samples or []):
            if sample.get("id") != rec["id"] or sample.get("label") != rec["label"]:
                raise ValueError(f"evaluation sample {sample.get('id')} does not match record {rec['index']}")
            _, tokens = parse_answer(sample["generated"])
            words = [normalize_token(t) for t in tokens]
            if rec["plant_index"] is not None:
                self.hit[rec["index"]] = words == [normalize_token(rec["modification"]["new_token_text"])]
                if self.hit[rec["index"]]:
                    continue
            if not words:
                continue
            for i, token_id in enumerate(rec["answer_ids"]):
                if normalize_token(tokenizer.decode([token_id])) == words[0]:
                    kind = "clean" if rec["label"] == "clean" else "modified"
                    self.named.append((kind, starts[rec["index"]] + i, rec["index"]))
                    break

    @staticmethod
    def _starts(index):
        start, out = 0, []
        for rec in index:
            out.append(start)
            start += rec["n_answer"]
        return out


def within_answer_percentile(values: torch.Tensor, layout: Layout, answer_row: int, record: int) -> float | None:
    """Fraction of the answer's other eligible tokens with a lower value."""
    start, n = layout.span[record]
    same = layout.eligible[start:start + n].clone()
    same[answer_row - start] = False
    if not layout.eligible[answer_row] or not same.any():
        return None
    return float((values[start:start + n][same] < values[answer_row]).double().mean())


def analyse_layer(layer: int, out: Path, models: list[str], layout: Layout) -> tuple[dict, dict]:
    states = {m: load_layer(out, m, layer)[layout.rows].float() for m in models}
    pairs = {}
    if layout.pairs:
        for m in models:
            pairs[m] = {k: load_layer(out, f"pairs/{m}/{k}", layer).float() for k in ("plant", "orig")}

    mask, half = layout.mask, layout.half
    directions = {}  # model -> definition -> {0, 1, "all"} -> vector
    for m in models:
        directions[m] = {}
        for name in DEFINITIONS:
            directions[m][name] = {}
            for h in (0, 1, "all"):
                if name == "pair":
                    if not pairs:
                        continue
                    keep = torch.ones(len(layout.pairs), dtype=torch.bool) if h == "all" else layout.pair_half == h
                    diff = pairs[m]["plant"][keep] - pairs[m]["orig"][keep]
                    directions[m][name][h] = diff.mean(0)
                    continue
                select = torch.ones_like(half, dtype=torch.bool) if h == "all" else half == h
                target = "plant" if name == "rest" else name
                directions[m][name][h] = states[m][mask[target] & select].mean(0) - states[m][mask["rest"] & select].mean(0)

    result = {"layer": layer, "norm": {}, "cos_models": {}, "cos_halves": {}, "cos_definitions": {},
              "auroc": {}, "auroc_after": {}, "auroc_pair": {}, "spearman_other": {}, "spearman_plant": {},
              "named": {}, "hit_auroc": {}, "plant_top1": {}}
    ft = models[-1]
    for m in models:
        defs = [d for d in DEFINITIONS if directions[m][d]]
        result["norm"][m] = {d: float(directions[m][d]["all"].norm()) for d in defs}
        result["cos_halves"][m] = {d: cosine(directions[m][d][0], directions[m][d][1]) for d in defs}
        result["cos_definitions"][m] = {
            f"{a}~{b}": cosine(directions[m][a]["all"], directions[m][b]["all"])
            for i, a in enumerate(defs) for b in defs[i + 1:]
        }
    if len(models) == 2:
        result["cos_models"] = {d: cosine(directions[models[0]][d]["all"], directions[models[1]][d]["all"])
                                for d in DEFINITIONS if directions[models[0]][d]}

    def crossfit(values_of_half, which_half):
        """Project each half with the direction estimated on the other half."""
        out_values = torch.empty(len(which_half))
        for h in (0, 1):
            rows = which_half == h
            out_values[rows] = values_of_half(1 - h, rows)
        return out_values

    for s in models:  # whose states
        for dm in models:  # whose direction
            key = f"{s}|{dm}"
            for name in DEFINITIONS:
                if not directions[dm][name]:
                    continue
                units = {h: directions[dm][name][h] / directions[dm][name][h].norm() for h in (0, 1)}
                proj = crossfit(lambda h, rows: states[s][rows] @ units[h], half)
                if name in ("after1", "after2"):
                    result["auroc_after"].setdefault(key, {})[name] = auroc(proj[mask[name]], proj[mask["rest"]])
                    continue
                result["auroc"].setdefault(key, {})[name] = auroc(proj[mask["plant"]], proj[mask["rest"]])
                if pairs:
                    plant = crossfit(lambda h, rows: pairs[s]["plant"][rows] @ units[h], layout.pair_half)
                    orig = crossfit(lambda h, rows: pairs[s]["orig"][rows] @ units[h], layout.pair_half)
                    result["auroc_pair"].setdefault(key, {})[name] = auroc(plant, orig)
                surprisal = layout.surprisal[s]
                result["spearman_other"].setdefault(key, {})[name] = spearman(proj[mask["other"]], surprisal[mask["other"]])
                result["spearman_plant"].setdefault(key, {})[name] = spearman(proj[mask["plant"]], surprisal[mask["plant"]])

                if s != ft:
                    continue
                # Where do wrongly named tokens, and plants, rank within their answer?
                named = {}
                for kind, row, record in layout.named:
                    p = within_answer_percentile(proj, layout, row, record)
                    if p is not None:
                        named.setdefault(kind, []).append(p)
                result["named"].setdefault(key, {})[name] = {
                    kind: {"n": len(v), "mean_percentile": sum(v) / len(v), "top": sum(x == 1.0 for x in v)}
                    for kind, v in named.items()
                }
                plant_rows = mask["plant"].nonzero().squeeze(-1).tolist()
                top, hits, misses = 0, [], []
                for row in plant_rows:
                    record = int(layout.record[row])
                    p = within_answer_percentile(proj, layout, row, record)
                    top += p == 1.0
                    if record in layout.hit:
                        (hits if layout.hit[record] else misses).append(float(proj[row]))
                result["plant_top1"].setdefault(key, {})[name] = top / max(len(plant_rows), 1)
                result["hit_auroc"].setdefault(key, {})[name] = auroc(torch.tensor(hits), torch.tensor(misses))
    full = {m: {d: directions[m][d]["all"] for d in DEFINITIONS if directions[m][d]} for m in models}
    return result, full


def surprisal_reference(layout: Layout, model: str) -> dict:
    """Layer-independent counterparts of the projection metrics, from surprisal."""
    surprisal = layout.surprisal[model]
    named = {}
    for kind, row, record in layout.named:
        p = within_answer_percentile(surprisal, layout, row, record)
        if p is not None:
            named.setdefault(kind, []).append(p)
    plant_rows = layout.mask["plant"].nonzero().squeeze(-1).tolist()
    top, hits, misses = 0, [], []
    for row in plant_rows:
        record = int(layout.record[row])
        top += within_answer_percentile(surprisal, layout, row, record) == 1.0
        if record in layout.hit:
            (hits if layout.hit[record] else misses).append(float(surprisal[row]))
    return {
        "named": {kind: {"n": len(v), "mean_percentile": sum(v) / len(v), "top": sum(x == 1.0 for x in v)}
                  for kind, v in named.items()},
        "plant_top1": top / max(len(plant_rows), 1),
        "hit_auroc": auroc(torch.tensor(hits), torch.tensor(misses)),
        "auroc_plant_vs_rest": auroc(surprisal[layout.mask["plant"]], surprisal[layout.mask["rest"]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plant directions from a hidden-state dump.")
    parser.add_argument("--dump", required=True, help="hidden_states output directory")
    parser.add_argument("--eval", default=None, help="eval_run output of the dumped checkpoint on the test split")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="default: every stored state")
    parser.add_argument("--tag", default=None, help="suffix of the output files (default: Condor job id)")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    from transformers import AutoTokenizer

    out = Path(args.dump)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    models = meta["models"]
    samples = json.loads(Path(args.eval).read_text(encoding="utf-8"))["samples"] if args.eval else None
    layout = Layout(out, models, samples, AutoTokenizer.from_pretrained(args.tokenizer))
    tag = args.tag or condor_job() or "manual"
    layers = args.layers if args.layers is not None else list(range(meta["n_states"]))

    report = {
        "command": " ".join(["python -m prefill_awareness.directions", *sys.argv[1:]]),
        "git": git, "condor_job": condor_job(), "started_at": utc_now(),
        "models": models, "margin": MARGIN,
        "counts": {k: int(v.sum()) for k, v in layout.mask.items()} | {
            "pairs": len(layout.pairs),
            "named": {kind: sum(k == kind for k, _, _ in layout.named) for kind in ("clean", "modified")},
            "hits": sum(layout.hit.values()), "modified_evaluated": len(layout.hit)},
        "surprisal_reference": {m: surprisal_reference(layout, m) for m in models},
        "layers": [],
    }
    saved = {m: {} for m in models}
    json_path, pt_path = out / f"directions-{tag}.json", out / f"directions-{tag}.pt"
    for layer in layers:
        result, full = analyse_layer(layer, out, models, layout)
        report["layers"].append(result)
        for m in models:
            for name, vector in full[m].items():
                saved[m].setdefault(name, []).append(vector)
        ft = f"{models[-1]}|{models[-1]}"
        print(f"[layer {layer}] auroc {json.dumps(result['auroc'].get(ft))} "
              f"cos_models {json.dumps(result['cos_models'])}", flush=True)
        json_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    report["finished_at"] = utc_now()
    json_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    torch.save({"layers": layers, "directions": {m: {k: torch.stack(v) for k, v in d.items()} for m, d in saved.items()}},
               pt_path)


if __name__ == "__main__":
    main()
