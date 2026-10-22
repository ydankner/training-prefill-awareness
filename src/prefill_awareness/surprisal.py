"""Per-token surprisal profile of the tampered assistant turn.

The experiment assumes the planted token is the one place in the turn where the
model did not choose its own continuation. This module checks that assumption
instead of trusting it, and answers the question the SFT puts at risk: does
training the model to *name* the planted token flatten the surprisal signal it
is supposed to be reading?

The headline is the plant's **rank** among the turn's tokens by surprisal. Rank 1
means the plant is still the single most surprising token. That degrades before
the averages do, so it is the early warning: if it slips, raise ``--kl-coef``.

Because the policy is a LoRA, the base model is recovered by disabling the
adapter, so trained and base profiles are measured on the very same tokens in
one pass, with no second set of weights.
"""

import argparse
import json

import torch

from .dataset import THINK_BLOCK, load_records


def split_backbone(model):
    """Return ``(backbone, lm_head)``, seeing through a PEFT wrapper.

    The LM head is applied only at the positions we need; materialising
    ``[length, vocab]`` logits for a long turn is pure waste.
    """
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(inner, "model", None), getattr(inner, "lm_head", None)


def plant_index(tokenizer, response_text: str, modification: dict):
    """Locate the planted token after re-tokenising the stored response.

    The stored index is into the token sequence as generated. Re-tokenising the
    decoded text usually reproduces it exactly, but a boundary can shift, so the
    index is verified against the planted id and repaired if it moved.
    """
    ids = tokenizer(response_text, add_special_tokens=False).input_ids
    return ids, locate_plant(ids, modification)


def locate_plant(ids: list[int], modification: dict):
    """Index of the planted token in ``ids``: the stored index if it still holds
    the planted id, else the nearest occurrence of that id, else None."""
    want, guess = modification["new_token_id"], modification["token_index"]
    if guess < len(ids) and ids[guess] == want:
        return guess
    matches = [i for i, t in enumerate(ids) if t == want]
    if not matches:
        return None
    return min(matches, key=lambda i: abs(i - guess))


@torch.no_grad()
def turn_surprisal(model, tokenizer, record, device, chunk: int = 128, stats: dict | None = None,
                   reading: bool = False):
    """Surprisal in nats for every token of the assistant turn.

    Returns ``(surprisals, plant_position)``; the position is None when the
    planted token could not be located after re-tokenisation.

    By default the turn follows the prefix it was generated after, which ends in
    the empty think block. ``reading=True`` scores it as it appears in the
    detection chat instead: Qwen3's template drops that block from an earlier
    turn, unless the record is marked ``keep_think`` (see ``dataset.render``).

    Pass a dict with empty lists under ``entropy``, ``top1_prob`` and ``rank`` to
    also collect, per token, the entropy of the predictive distribution, the
    probability of its most likely token and the rank of the actual token (1 =
    the greedy choice). The dict also receives the token ids and, for a modified
    record, ``orig_logprob``: the log-probability of the original token at the
    plant position.
    """
    prefix = tokenizer.apply_chat_template(
        record["messages"][:1], tokenize=False, add_generation_prompt=True,
        **({"enable_thinking": record["enable_thinking"]}
           if record.get("enable_thinking") is not None else {}),
    )
    if reading and not record.get("keep_think") and prefix.endswith(THINK_BLOCK):
        prefix = prefix[: -len(THINK_BLOCK)]
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    response = record["messages"][1]["content"]
    modification = record.get("modification")
    if modification:
        response_ids, plant = plant_index(tokenizer, response, modification)
    else:
        response_ids, plant = tokenizer(response, add_special_tokens=False).input_ids, None
    if not response_ids:
        return [], None

    ids = torch.tensor([prefix_ids + response_ids], device=device)
    backbone, head = split_backbone(model)
    if backbone is None or head is None:
        logits = model(ids).logits[0]
        hidden = None
    else:
        hidden = backbone(input_ids=ids).last_hidden_state[0]

    start = len(prefix_ids)
    out = []
    for lo in range(0, len(response_ids), chunk):
        hi = min(lo + chunk, len(response_ids))
        rows = torch.arange(start + lo - 1, start + hi - 1, device=device)
        piece = head(hidden[rows]).float() if hidden is not None else logits[rows].float()
        targets = torch.tensor(response_ids[lo:hi], device=device)
        logp = torch.log_softmax(piece, dim=-1)
        if stats is None:
            out.append(-logp[torch.arange(hi - lo, device=device), targets])
            continue
        actual = logp[torch.arange(hi - lo, device=device), targets]
        out.append(-actual)
        stats["entropy"].extend((-(logp.exp() * logp).sum(-1)).tolist())
        stats["top1_prob"].extend(logp.max(-1).values.exp().tolist())
        stats["rank"].extend(((logp > actual[:, None]).sum(-1) + 1).tolist())
        if plant is not None and lo <= plant < hi:
            stats["orig_logprob"] = float(logp[plant - lo, modification["orig_token_id"]])
    if stats is not None:
        stats["ids"] = response_ids
    return torch.cat(out).tolist(), plant


def profile(model, tokenizer, records, device, chunk: int = 128, details: list | None = None,
            reading: bool = False) -> dict:
    """Aggregate the plant-vs-natural contrast over modified records.

    Pass a list as ``details`` to also collect, per record, the plant's rank and
    the tokens that are more surprising than it. ``reading``: score each turn in
    the detection chat's context (see turn_surprisal).
    """
    plant_vals, natural_vals, ranks, top1, n = [], [], [], 0, 0
    for record in records:
        if not record.get("modification"):
            continue
        surprisals, plant = turn_surprisal(model, tokenizer, record, device, chunk, reading=reading)
        if plant is None or not surprisals:
            continue
        n += 1
        value = surprisals[plant]
        others = [s for i, s in enumerate(surprisals) if i != plant]
        plant_vals.append(value)
        natural_vals.extend(others)
        # 1 = the plant is the most surprising token in the turn.
        rank = 1 + sum(1 for s in others if s > value)
        ranks.append(rank)
        top1 += int(rank == 1)
        if details is not None:
            ids, _ = plant_index(tokenizer, record["messages"][1]["content"], record["modification"])
            above = sorted((i for i, s in enumerate(surprisals) if i != plant and s > value),
                           key=lambda i: -surprisals[i])
            details.append({
                "id": record.get("id"),
                "plant_index": plant,
                "plant_text": tokenizer.decode([ids[plant]]),
                "plant_surprisal": value,
                "rank": rank,
                "more_surprising": [
                    {"index": i, "text": tokenizer.decode([ids[i]]), "surprisal": surprisals[i]}
                    for i in above[:5]
                ],
            })

    if not n:
        return {"n": 0}
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "n": n,
        "plant_surprisal": mean(plant_vals),
        "natural_surprisal": mean(natural_vals),
        "gap": mean(plant_vals) - mean(natural_vals),
        "plant_rank": mean(ranks),
        "plant_is_top1": top1 / n,
    }


def compare(model, tokenizer, records, device, chunk: int = 128) -> dict:
    """Profile with the adapter active and, if there is one, with it disabled."""
    was_training = model.training
    model.eval()  # dropout would add noise to what is meant to be a measurement
    try:
        if not hasattr(model, "disable_adapter"):
            # No adapter to turn off: this model *is* the base.
            result = {"base": profile(model, tokenizer, records, device, chunk)}
        else:
            result = {"trained": profile(model, tokenizer, records, device, chunk)}
            with model.disable_adapter():
                result["base"] = profile(model, tokenizer, records, device, chunk)
    finally:
        if was_training:
            model.train()
    return result


def format_table(result: dict) -> str:
    """A compact fixed-width block; the base row is the reference to read against."""
    header = (
        f"  {'':8} {'plant':>8} {'natural':>8} {'gap':>8} {'rank':>7} {'top1':>7}\n"
        f"  {'-' * 50}"
    )
    lines = [header]
    for name in ("base", "trained"):
        row = result.get(name)
        if not row or not row.get("n"):
            continue
        lines.append(
            f"  {name:8} {row['plant_surprisal']:8.2f} {row['natural_surprisal']:8.2f} "
            f"{row['gap']:8.2f} {row['plant_rank']:7.1f} {row['plant_is_top1']:7.2f}"
        )
    lines.append(
        "  (surprisal in nats; rank 1 = plant is the most surprising token in the turn;"
        "\n   top1 = fraction of turns where that holds. A falling gap or top1 after SFT"
        "\n   means the signal is being trained away -- raise --kl-coef.)"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plant-vs-natural surprisal profile.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter; enables the base comparison")
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=64, help="modified records to profile")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from .evaluate import load_model

    model, tokenizer = load_model(args.model, args.adapter, args.dtype, args.load_4bit)
    records = [r for r in load_records(args.data, args.split) if r.get("modification")][: args.n]
    result = compare(model, tokenizer, records, model.device)
    print(format_table(result))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
