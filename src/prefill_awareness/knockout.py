"""Block attention to the plant and see whether the fine-tuned model still names it.

For every modified test record, teacher-forced: the log-probability of the gold
reply (the plant) and of ``none``, and the rank of the gold reply's first token,
under each knockout setting (through ``attention.probe_attention``):

- ``cutoff-all-L``: every head in layers >= L gets no attention to the plant from
  any later position,
- ``cutoff-post-L``: the same, but only from the positions after the tampered
  turn (its end, the question, the reply header),
- ``top-post-k`` / ``top-all-k``: only the k heads that attend most to plants
  (mean attention to the plant minus the mean to other answer tokens, from the
  positions after the turn, in the fine-tuned model; from ``attention``),
- ``random-post-k`` / ``random-all-k``: k random heads with the same layers as the
  top k, as a control.

Writes ``knockout-<tag>.jsonl`` (per record and setting) and
``knockout-summary-<tag>.json`` into ``--run``.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from .attention import PROBE, register
from .dataset import Collator, data_files, encode, load_records
from .evaluate import load_model
from .formats import NO_CHANGE
from .hidden_states import layout
from .provenance import condor_job, describe, file_sha256, require_committed, utc_now
from .surprisal import split_backbone

CUTOFFS = [0, 16, 32, 40, 48, 52, 56, 58, 60, 61, 62, 63]


def ranked_heads(summary_path: Path, checkpoint: str, kind: str = "post") -> list[tuple[int, int, float]]:
    table = json.loads(summary_path.read_text(encoding="utf-8"))["summary"][checkpoint][kind]
    excess = [(layer, head, plant - other)
              for layer, (plants, others) in enumerate(zip(table["plant"], table["other"]))
              for head, (plant, other) in enumerate(zip(plants, others)) if plant is not None]
    return sorted(excess, key=lambda x: -x[2])


def settings(heads: list[tuple[int, int, float]], n_layers: int, n_heads: int, seed: int) -> dict:
    """name -> list of (layer, heads or None, from) with from in {"all", "post"}."""
    out = {"none": []}
    for scope in ("all", "post"):
        for cutoff in CUTOFFS:
            out[f"cutoff-{scope}-{cutoff}"] = [(layer, None, scope) for layer in range(cutoff, n_layers)]
    rng = random.Random(seed)

    def grouped(pairs, scope):
        by_layer = {}
        for layer, head in pairs:
            by_layer.setdefault(layer, []).append(head)
        return [(layer, hs, scope) for layer, hs in sorted(by_layer.items())]

    for k, scopes in ((1, ("post",)), (2, ("post",)), (4, ("post",)), (8, ("post", "all")),
                      (16, ("post",)), (32, ("post", "all"))):
        if k > len(heads):
            continue
        top = [(layer, head) for layer, head, _ in heads[:k]]
        for scope in scopes:
            out[f"top-{scope}-{k}"] = grouped(top, scope)
    for k, scopes in ((4, ("post",)), (16, ("post",)), (32, ("post", "all"))):
        if 2 * k > n_layers * n_heads:
            continue
        top = [(layer, head) for layer, head, _ in heads[:k]]
        taken = set(top)
        control = []
        for layer, _ in top:
            free = [(layer, h) for h in range(n_heads) if (layer, h) not in taken]
            # A layer the top heads fill up lends its control head to any other layer.
            pick = rng.choice(free or [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in taken])
            taken.add(pick)
            control.append(pick)
        for scope in scopes:
            out[f"random-{scope}-{k}"] = grouped(control, scope)
    return out


@torch.no_grad()
def score(model, tokenizer, record, lay, plan, device) -> dict:
    backbone, head = split_backbone(model)
    plant = lay["answer"][lay["plant_index"]]
    post = lay["answer"][-1] + 1
    PROBE.block = {}
    for layer, heads, scope in plan:
        index = None if heads is None else torch.tensor(heads, device=device)
        PROBE.block.setdefault(layer, []).append((index, plant + 1 if scope == "all" else post, plant))
    encoded = [encode(tokenizer, record["messages"], record.get("enable_thinking")),
               encode(tokenizer, record["messages"][:-1] + [{"role": "assistant", "content": NO_CHANGE}],
                      record.get("enable_thinking"))]
    batch = Collator(tokenizer.pad_token_id)([{"input_ids": i, "labels": l} for i, l in encoded])
    batch = {k: v.to(device) for k, v in batch.items()}
    try:
        hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    finally:
        PROBE.block = {}
    out = {}
    for row, name in enumerate(("gold", "none")):
        targets = (batch["labels"][row, 1:] != -100).nonzero().squeeze(-1)
        logits = head(hidden[row, targets]).float()
        wanted = batch["labels"][row, targets + 1]
        logp = torch.log_softmax(logits, dim=-1)
        token = logp[torch.arange(len(targets), device=logp.device), wanted]
        out[name] = float(token.sum())
        out[f"{name}_first"] = float(token[0])
        if name == "gold":
            out["gold_rank"] = int((logits[0] > logits[0, wanted[0]]).sum()) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention knockouts at the plant.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--heads-from", required=True, help="attention-summary-<tag>.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--settings", nargs="*", default=None, help="subset of setting names (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    run = Path(args.run)
    tag = args.tag or condor_job() or "manual"
    records = [r for r in load_records(args.data, None if args.split == "all" else args.split)[: args.limit]]
    base, tokenizer = load_model(args.model, None, args.dtype, attn_implementation=register())
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(run / args.checkpoint))
    model.eval()
    heads = ranked_heads(Path(args.heads_from), args.checkpoint)
    plans = settings(heads, model.config.num_hidden_layers, model.config.num_attention_heads, args.seed)
    if args.settings:
        plans = {name: plans[name] for name in args.settings}

    meta = {
        **describe("prefill_awareness.knockout", args, git, model),
        "data": {str(f): file_sha256(f) for f in data_files(args.data)},
        "top_heads": heads[:32],
        "settings": {name: [[layer, hs, scope] for layer, hs, scope in plan] for name, plan in plans.items()},
    }
    meta_path = run / f"knockout-meta-{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    items = [(i, r, layout(tokenizer, r)) for i, r in enumerate(records) if r.get("modification")]
    items = [(i, r, lay) for i, r, lay in items if lay["plant_index"] is not None]
    rows = {}
    with open(run / f"knockout-{tag}.jsonl", "w", encoding="utf-8") as handle:
        for name, plan in plans.items():
            rows[name] = []
            for index, record, lay in items:
                result = {"setting": name, "index": index, **score(model, tokenizer, record, lay, plan, model.device)}
                rows[name].append(result)
                handle.write(json.dumps(result) + "\n")
            handle.flush()
            n = len(rows[name])
            margin = sorted(r["gold"] - r["none"] for r in rows[name])
            print(f"[{name}] gold first-token top-1 {sum(r['gold_rank'] == 1 for r in rows[name])}/{n}, "
                  f"gold beats none {sum(m > 0 for m in margin)}/{n}", flush=True)

    summary = {}
    for name, results in rows.items():
        n = len(results)
        margins = sorted(r["gold"] - r["none"] for r in results)
        ranks = sorted(r["gold_rank"] for r in results)
        summary[name] = {
            "n": n,
            "gold_first_top1": sum(r == 1 for r in ranks),
            "gold_beats_none": sum(m > 0 for m in margins),
            "median_gold_minus_none": margins[n // 2],
            "mean_gold_logprob": sum(r["gold"] for r in results) / n,
            "mean_none_logprob": sum(r["none"] for r in results) / n,
            "median_gold_rank": ranks[n // 2],
        }
    (run / f"knockout-summary-{tag}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    meta.update(finished_at=utc_now())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
