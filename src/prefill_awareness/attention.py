"""Attention to planted tokens: patterns, geometry, and attention under steering.

Everything here runs through one attention function registered with transformers
(``probe_attention``). It computes attention exactly like the eager
implementation, from the queries and keys each layer produces after its
per-head norms and RoPE, and additionally lets a caller record the attention
weights of chosen query positions and block chosen heads from attending to a
chosen key position (used by ``knockout``).

For every test record and for the base model (adapter off) and one fine-tuned
checkpoint this measures, per layer and head, how much attention the tokens after
the tampered turn pay to its tokens:

- ``last``: the final position of the reply header, which predicts the reply's
  first token,
- ``post``: the mean over every position after the tampered turn (its end, the
  question, the header).

For each, the attention to the target token (the plant, or on clean answers the
token the fine-tuned model wrongly named), the mean attention to the answer's other
eligible tokens (not the first or last ``MARGIN``), the target's percentile among
them, and the answer's total attention mass.

Geometry, on modified records with minimal pairs: for every head, the attention
score (pre-softmax, from ``last`` to the plant position) for keys computed from the
original token's residual state, the plant's, the original plus only the plant's
shift along the fine-tuned minimal-pair direction, the plant minus that shift, and
the original plus a shift of the same length along three random directions.

Steering: for the steering targets of a previous ``steer`` run, the attention from
``last`` to the steered token while adding multiples of the fine-tuned minimal-pair
direction at chosen layers.

Writes ``attention-<tag>.pt`` (all per-record arrays; geometry rows follow
``pairs/index.jsonl``) and
``attention-summary-<tag>.json`` into ``--run``.
"""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import data_files, load_records
from .evaluate import load_model
from .formats import normalize_token, parse_answer
from .hidden_states import layout
from .provenance import condor_job, describe, file_sha256, require_committed, utc_now
from .steer import Steering
from .surprisal import split_backbone

NAME = "prefill_probe"
MARGIN = 8
METRICS = ("tgt", "other", "pct", "mass")
VARIANTS = ("actual", "orig", "plant", "orig+dir", "plant-dir", "orig+rand0", "orig+rand1", "orig+rand2")


class Probe:
    """What ``probe_attention`` records and blocks.

    ``record(layer, weights, query, key)`` is called with the post-softmax weights
    ``[batch, heads, queries, keys]`` (float32) and the post-RoPE queries and keys.
    ``block`` maps a layer index to ``(heads, query_from, key)`` triples: those heads
    (None = all) get no attention from positions ``>= query_from`` to ``key``.
    """

    def __init__(self):
        self.record = None
        self.block = {}


PROBE = Probe()


def probe_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    groups = query.shape[1] // key.shape[1]
    keys = key.repeat_interleave(groups, dim=1)
    values = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, keys.transpose(2, 3)) * scaling
    n_query, n_key = query.shape[-2], keys.shape[-2]
    offset = n_key - n_query  # > 0 only when decoding with a cache
    if n_query > 1:
        future = torch.ones(n_query, n_key, dtype=torch.bool, device=query.device).triu(offset + 1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
    for heads, query_from, key_position in PROBE.block.get(module.layer_idx, ()):
        rows = slice(max(query_from - offset, 0), None)
        if heads is None:
            scores[:, :, rows, key_position] = torch.finfo(scores.dtype).min
        else:
            scores[:, heads, rows, key_position] = torch.finfo(scores.dtype).min
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if PROBE.record is not None:
        PROBE.record(module.layer_idx, weights, query, keys)
    output = torch.matmul(weights.to(query.dtype), values)
    return output.transpose(1, 2).contiguous(), None


def register() -> str:
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

    AttentionInterface.register(NAME, probe_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, sdpa_mask)
    return NAME


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def named_positions(tokenizer, records, layouts, samples) -> dict:
    """Record index -> answer index of the first token the model wrongly named on a clean answer."""
    out = {}
    for index, (record, lay, sample) in enumerate(zip(records, layouts, samples)):
        if sample["id"] != record.get("id") or sample["label"] != record["label"]:
            raise ValueError(f"evaluation sample {sample['id']} does not match record {index}")
        if record["label"] != "clean":
            continue
        _, tokens = parse_answer(sample["generated"])
        if not tokens:
            continue
        word = normalize_token(tokens[0])
        for i, token_id in enumerate(lay["answer_ids"]):
            if normalize_token(tokenizer.decode([token_id])) == word:
                out[index] = i
                break
    return out


def summarise_rows(weights_rows: torch.Tensor, answer: list[int], target: int | None) -> torch.Tensor:
    """``[heads, 4]``: attention to the target, mean to the other eligible answer
    tokens, the target's percentile among them, and the answer's total mass."""
    heads = weights_rows.shape[0]
    n = len(answer)
    eligible = [answer[i] for i in range(MARGIN, n - MARGIN)]
    others = [p for p in eligible if p != target]
    out = torch.full((heads, 4), float("nan"), device=weights_rows.device)
    other_values = weights_rows[:, others]
    out[:, 1] = other_values.mean(-1)
    out[:, 3] = weights_rows[:, answer].sum(-1)
    if target is not None:
        value = weights_rows[:, target]
        out[:, 0] = value
        out[:, 2] = (other_values < value[:, None]).float().mean(-1)
    return out


@torch.no_grad()
def patterns(model, records, layouts, targets: dict, pairs: dict | None, direction: torch.Tensor | None,
             generator_seed: int = 0):
    """Per-record attention summaries and, where pairs are given, geometry scores."""
    backbone, _ = split_backbone(model)
    n_layers = len(backbone.layers)
    n_heads = model.config.num_attention_heads
    stats = {kind: torch.full((len(records), n_layers, n_heads, 4), float("nan"), dtype=torch.float16)
             for kind in ("last", "post")}
    geometry = None
    random_dirs = None
    if pairs is not None:
        geometry = torch.full((len(pairs["row_of"]), n_layers, n_heads, len(VARIANTS)), float("nan"), dtype=torch.float32)
        gen = torch.Generator().manual_seed(generator_seed)
        random_dirs = torch.randn(n_layers, 3, direction.shape[-1], generator=gen)
        random_dirs = random_dirs / random_dirs.norm(dim=-1, keepdim=True)

    for index, (record, lay) in enumerate(zip(records, layouts)):
        answer = lay["answer"]
        last = len(lay["ids"]) - 1
        post = list(range(answer[-1] + 1, len(lay["ids"])))
        target = targets.get(index)
        target_position = answer[target] if target is not None else None
        captured = {}

        def record_layer(layer, weights, query, keys):
            rows = weights[0, :, post, :]
            stats["last"][index, layer] = summarise_rows(rows[:, -1, :], answer, target_position).cpu().half()
            stats["post"][index, layer] = summarise_rows(rows.mean(1), answer, target_position).cpu().half()
            if geometry is not None and target_position is not None and lay["plant_index"] is not None:
                q = query[0, :, last].float()
                actual = (q * keys[0, :, target_position].float()).sum(-1) * query.shape[-1] ** -0.5
                captured[layer] = (q, actual)

        PROBE.record = record_layer
        try:
            model(input_ids=torch.tensor([lay["ids"]], device=model.device), use_cache=False)
        finally:
            PROBE.record = None

        if captured and index in pairs["row_of"]:
            row = pairs["row_of"][index]
            position = target_position
            cos, sin = backbone.rotary_emb(torch.zeros(1, device=model.device, dtype=torch.float32),
                                           torch.tensor([[position]], device=model.device))
            for layer, (q, actual) in captured.items():
                block = backbone.layers[layer]
                attn = block.self_attn
                x_plant = pairs["plant"](layer)[row].float()
                x_orig = pairs["orig"](layer)[row].float()
                unit = direction[layer] / direction[layer].norm()
                shift = (x_plant - x_orig) @ unit
                variants = torch.stack([
                    x_orig, x_plant, x_orig + shift * unit, x_plant - shift * unit,
                    *(x_orig + shift * random_dirs[layer, j] for j in range(3)),
                ]).to(model.device)
                hidden = block.input_layernorm(variants.to(model.dtype))
                k = attn.k_norm(attn.k_proj(hidden).view(len(variants), -1, attn.head_dim)).float()
                k = k * cos[0, 0].float() + rotate_half(k) * sin[0, 0].float()
                k = k.repeat_interleave(n_heads // k.shape[1], dim=1)
                scores = (k * q[None]).sum(-1) * attn.head_dim ** -0.5
                geometry[row, layer, :, 0] = actual.cpu()
                geometry[row, layer, :, 1:] = scores.T.cpu()
        if (index + 1) % 100 == 0:
            print(f"[patterns] {index + 1}/{len(records)}", flush=True)
    return stats, geometry


@torch.no_grad()
def steering_attention(model, tokenizer, records, targets_meta, direction, layers, alphas):
    """Attention from the header's last position to the steered token, per head, for
    each steering layer and strength: ``[targets, layers, alphas, model layers, heads, 2]``
    holding the weight and its percentile among the answer's eligible tokens."""
    backbone, _ = split_backbone(model)
    modules = [backbone.embed_tokens, *backbone.layers]
    n_layers, n_heads = len(backbone.layers), model.config.num_attention_heads
    out = torch.full((len(targets_meta), len(layers), len(alphas), n_layers, n_heads, 2), float("nan"))
    for t, target in enumerate(targets_meta):
        lay = layout(tokenizer, records[target["index"]])
        position, answer = target["position"], lay["answer"]
        ids = torch.tensor([lay["ids"]] * len(alphas), device=model.device)
        for s, state in enumerate(layers):
            steering = Steering(modules[state], position)
            steering.shift = (torch.tensor(alphas)[:, None] * direction[state][None]).to(model.device)

            def record_layer(layer, weights, query, keys):
                for a in range(len(alphas)):
                    summary = summarise_rows(weights[a, :, -1, :], answer, position)
                    out[t, s, a, layer, :, 0] = summary[:, 0].cpu()
                    out[t, s, a, layer, :, 1] = summary[:, 2].cpu()

            PROBE.record = record_layer
            try:
                model(input_ids=ids, use_cache=False)
            finally:
                PROBE.record = None
                steering.remove()
        print(f"[steering attention] target {target['index']} done", flush=True)
    return out


def head_table(values: torch.Tensor) -> list:
    return [[None if v != v else round(float(v), 5) for v in row] for row in values]


def summary(stats: dict, geometry: dict, labels: list[str], named: dict) -> dict:
    """Means over records, per model, position kind, layer and head."""
    modified = torch.tensor([label == "modified" for label in labels])
    clean_named = torch.zeros(len(labels), dtype=torch.bool)
    for index in named:
        clean_named[index] = True
    out = {}
    for model_name, by_kind in stats.items():
        out[model_name] = {}
        for kind, values in by_kind.items():
            values = values.float()
            plants, wrong = values[modified], values[clean_named]
            out[model_name][kind] = {
                "plant": head_table(plants[..., 0].nanmean(0)),
                "other": head_table(plants[..., 1].nanmean(0)),
                "plant_pct": head_table(plants[..., 2].nanmean(0)),
                "plant_top": head_table((plants[..., 2] == 1).float().mean(0)),
                "mass": head_table(plants[..., 3].nanmean(0)),
                "named_pct": head_table(wrong[..., 2].nanmean(0)),
                "named_top": head_table((wrong[..., 2] == 1).float().mean(0)),
            }
        if geometry.get(model_name) is not None:
            g = geometry[model_name]
            keep = ~torch.isnan(g[:, 0, 0, 0])
            g = g[keep]
            base = g[..., 2] - g[..., 1]  # plant - orig
            out[model_name]["geometry"] = {
                "n": int(keep.sum()),
                "plant_minus_orig": head_table(base.mean(0)),
                "dir_minus_orig": head_table((g[..., 3] - g[..., 1]).mean(0)),
                "plant_minus_dir_minus_orig": head_table((g[..., 4] - g[..., 1]).mean(0)),
                "rand_minus_orig": head_table((g[..., 5:] - g[..., 1:2]).mean(-1).mean(0)),
                "actual_minus_plant_max": float((g[..., 0] - g[..., 2]).abs().max()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention to planted tokens.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--eval", required=True, help="eval_run output of --checkpoint on the test split")
    parser.add_argument("--dump", required=True, help="hidden_states output with pairs/ (states may be deleted)")
    parser.add_argument("--directions", required=True, help="directions-<tag>.pt")
    parser.add_argument("--steering-meta", default=None, help="steering-meta-<tag>.json whose targets to probe")
    parser.add_argument("--steer-layers", type=int, nargs="*", default=[32, 40, 48, 52, 56, 58, 60, 61, 62, 63])
    parser.add_argument("--steer-alphas", type=float, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--split", default="test", help="must match the split the dump was made from")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    run, dump = Path(args.run), Path(args.dump)
    tag = args.tag or condor_job() or "manual"
    records = load_records(args.data, None if args.split == "all" else args.split)[: args.limit]
    samples = json.loads(Path(args.eval).read_text(encoding="utf-8"))["samples"]
    base, tokenizer = load_model(args.model, None, args.dtype, attn_implementation=register())
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(run / args.checkpoint))
    model.eval()
    layouts = [layout(tokenizer, record) for record in records]
    named = named_positions(tokenizer, records, layouts, samples)
    targets = {i: lay["plant_index"] for i, lay in enumerate(layouts) if lay["plant_index"] is not None} | named
    saved = torch.load(args.directions)
    direction = saved["directions"][args.checkpoint]["pair"].float()

    meta = {
        **describe("prefill_awareness.attention", args, git, model),
        "data": {str(f): file_sha256(f) for f in data_files(args.data)},
        "n_records": len(records), "n_named": len(named), "variants": VARIANTS, "metrics": METRICS,
    }
    meta_path = run / f"attention-meta-{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    hidden_size = model.config.hidden_size
    pair_index = [json.loads(line) for line in (dump / "pairs" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    stats, geometry = {}, {}
    for name in ("base", args.checkpoint):
        def pair_reader(kind, name=name):
            def read(layer):
                data = np.memmap(dump / "pairs" / name / kind / f"layer-{layer:02d}.bin", dtype=np.int16, mode="r")
                return torch.from_numpy(np.array(data).reshape(-1, hidden_size)).view(torch.bfloat16)
            cache = {}

            def cached(layer):
                if layer not in cache:
                    cache[layer] = read(layer)
                return cache[layer]
            return cached
        pairs = {"row_of": {p["index"]: p["pair"] for p in pair_index},
                 "plant": pair_reader("plant"), "orig": pair_reader("orig")}
        context = model.disable_adapter() if name == "base" else contextlib.nullcontext()
        with context:
            stats[name], geometry[name] = patterns(model, records, layouts, targets, pairs, direction)
        print(f"[{name}] patterns done", flush=True)

    steer = None
    if args.steering_meta:
        targets_meta = json.loads(Path(args.steering_meta).read_text(encoding="utf-8"))["targets"]
        steer = steering_attention(model, tokenizer, records, targets_meta, direction,
                                   args.steer_layers, args.steer_alphas)

    labels = [record["label"] for record in records]
    torch.save({"stats": stats, "geometry": geometry, "steering": steer, "named": named, "labels": labels,
                "steer_layers": args.steer_layers, "steer_alphas": args.steer_alphas},
               run / f"attention-{tag}.pt")
    result = {"summary": summary(stats, geometry, labels, named), "n_modified": labels.count("modified"),
              "n_named": len(named), "steer_layers": args.steer_layers, "steer_alphas": args.steer_alphas}
    if steer is not None:
        # Per steering setting and model layer: the largest attention any head pays to
        # the steered token, and that head's percentile among the answer's tokens.
        weight, best = steer[..., 0].max(-1)
        pct = steer[..., 1].gather(-1, best[..., None]).squeeze(-1)
        result["steering"] = {"max_weight": weight.tolist(), "head": best.tolist(), "pct": pct.tolist()}
    (run / f"attention-summary-{tag}.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    meta.update(finished_at=utc_now())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
