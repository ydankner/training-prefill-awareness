"""Compare original-token guesses with the base LM distribution before each plant."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

from prefill_awareness.data_gen import logits_at
from prefill_awareness.formats import normalize_token

from .probe import parse_generated, read_jsonl


def original_context(task: dict, tokenizer):
    """Return the base-generation prefix and visible answer token IDs."""
    import torch

    options = ({} if task.get("enable_thinking") is None else
               {"enable_thinking": task["enable_thinking"]})
    prefix = tokenizer.apply_chat_template(
        task["messages"][:1], tokenize=False, add_generation_prompt=True, **options)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    if "answer_token_ids" in task:
        answer_ids = task["answer_token_ids"]
    else:
        clean = task.get("clean_response")
        if clean is None:
            return None
        answer_ids = tokenizer(clean, add_special_tokens=False).input_ids
    return prefix_ids, answer_ids


def distribution_rows(tasks: list[dict], base, tokenizer, batch_size: int = 4) -> list[dict]:
    import torch

    prepared = []
    rows = []
    for task in tasks:
        if not task["plants"]:
            continue
        context = original_context(task, tokenizer)
        if context is None:
            rows.append({"id": task["id"], "error": "missing original answer"})
            continue
        prefix, answer = context
        for plant_index, plant in enumerate(task["plants"]):
            pos = plant["token_index"]
            if pos > len(answer) or pos < 0:
                rows.append({"id": task["id"], "plant_index": plant_index,
                             "error": "plant position outside answer"})
                continue
            # Dolly stores decoded clean text rather than original token IDs. If
            # re-tokenization shifts the position, do not invent a distribution.
            if task["group"] == "dolly" and (pos == len(answer) or
                    answer[pos] != plant["orig_token_id"]):
                rows.append({"id": task["id"], "plant_index": plant_index,
                             "error": "clean Dolly text did not retokenize identically"})
                continue
            sequence = torch.tensor(prefix + answer[:pos], device=base.device)
            prepared.append((task, plant_index, plant, sequence))

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        sequences = [item[3] for item in batch]
        indices = [(i, len(seq) - 1) for i, seq in enumerate(sequences)]
        for (task, plant_index, plant, _), logits in zip(
                batch, logits_at(base, tokenizer, sequences, indices, base.device)):
            probs = logits.softmax(dim=-1)
            original_id = int(plant["orig_token_id"])
            original_prob = float(probs[original_id])
            top_probs, top_ids = torch.topk(probs, min(100, len(probs)))
            candidates = [{"id": token_id,
                           "text": normalize_token(tokenizer.decode([token_id])),
                           "prob": prob}
                          for prob, token_id in zip(top_probs.tolist(), top_ids.tolist())]
            rows.append({
                "id": task["id"], "plant_index": plant_index,
                "group": task["group"], "mode": task["mode"],
                "kind": task["kind"],
                "original": normalize_token(plant["orig_token_text"]),
                "replacement": normalize_token(plant["new_token_text"]),
                "stored_original_prob": plant["orig_prob"],
                "original_prob": original_prob,
                "original_rank": int((probs > probs[original_id]).sum()) + 1,
                "top_token": candidates[0], "top_10": candidates[:10],
                "top_100": candidates,
            })
        print(f"distribution: {min(start + batch_size, len(prepared))}/{len(prepared)}",
              flush=True)
    return rows


def score_distribution(tasks: list[dict], distribution: list[dict],
                       model_names: list[str], out: Path) -> dict:
    by_task = {t["id"]: t for t in tasks}
    by_id = {name: {r["id"]: r for r in read_jsonl(out / f"{name}.jsonl")}
             for name in model_names}
    scored = []
    groups = defaultdict(list)
    for dist in distribution:
        if "error" in dist:
            scored.append(dist)
            continue
        task = by_task[dist["id"]]
        original = dist["original"]
        for name in model_names:
            predicted, _ = parse_generated(by_id[name][dist["id"]]["answer"], "recovery")
            available = Counter(predicted)
            recovered = []
            for plant in task["plants"]:
                token = normalize_token(plant["orig_token_text"])
                recovered.append(available[token] > 0)
                available[token] -= int(available[token] > 0)
            guessed = (predicted[dist["plant_index"]]
                       if len(predicted) == len(task["plants"]) else None)
            match = next((c for c in dist["top_100"] if c["text"] == guessed), None)
            row = {k: v for k, v in dist.items() if k != "top_100"}
            row.update({"model": name, "recovered": recovered[dist["plant_index"]],
                        "guessed_at_position": guessed,
                        "guess_top_100_rank": (dist["top_100"].index(match) + 1
                                               if match is not None else None),
                        "guess_top_100_prob": match["prob"] if match else None})
            scored.append(row)
            groups[(name, dist["group"], dist["mode"])].append(row)

    with (out / "distribution-scored.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"skipped_contexts": sum("error" in row for row in distribution)}
    for (name, group, mode), rows in groups.items():
        ranks = [r["guess_top_100_rank"] for r in rows if r["guess_top_100_rank"]]
        summary[f"{name}/{group}/{mode}"] = {
            "n_plants": len(rows),
            "recovered_rate": sum(r["recovered"] for r in rows) / len(rows),
            "mean_base_token_probability": sum(r["original_prob"] for r in rows) / len(rows),
            "original_top_1_rate": sum(r["original_rank"] == 1 for r in rows) / len(rows),
            "original_top_10_rate": sum(r["original_rank"] <= 10 for r in rows) / len(rows),
            "guessed_top_1_rate": sum(r["guess_top_100_rank"] == 1 for r in rows) / len(rows),
            "guessed_top_10_rate": sum((r["guess_top_100_rank"] or 101) <= 10
                                      for r in rows) / len(rows),
            "guessed_top_100_rate": len(ranks) / len(rows),
            "median_guess_rank_within_top_100": median(ranks) if ranks else None,
            "stored_prob_max_abs_error": max(abs(r["original_prob"] -
                                             r["stored_original_prob"]) for r in rows),
            "by_original_probability": [
                {"range": [lo, hi], "n": len(bucket),
                 "recovered_rate": (sum(r["recovered"] for r in bucket) / len(bucket)
                                    if bucket else None),
                 "mean_base_token_probability": (sum(r["original_prob"] for r in bucket)
                                                 / len(bucket) if bucket else None)}
                for lo, hi in ((0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.01))
                for bucket in [[r for r in rows if lo <= r["original_prob"] < hi]]
            ],
        }
    (out / "distribution-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
