"""Generate clean, one-plant and two-plant cases beyond the Dolly training domain.

The backbone that generated the SFT dataset writes the answer. Each plant is a
low-probability word-start token under that backbone; the suffix is regenerated
after insertion. Greedy and sampled answers are generated independently.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_for(row_id: str, mode: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{row_id}:{mode}".encode()).hexdigest()[:8], 16)


def eligible(tokenizer, token_id: int) -> bool:
    from prefill_awareness.data_gen import is_word_initial, special_ids

    if token_id in special_ids(tokenizer):
        return False
    text = tokenizer.decode([token_id])
    body = text.strip()
    return (bool(body) and body[0].isalpha() and len(body) <= 32
            and "\n" not in text and "\r" not in text and "\t" not in text
            and is_word_initial(tokenizer, token_id, text))


def pick_plant(model, tokenizer, prefix, response, args, rng, after: int = 0):
    import torch
    from prefill_awareness.data_gen import logits_at

    lo = max(args.min_pos, after)
    hi = len(response) - args.min_tail
    positions = [p for p in range(lo, hi) if eligible(tokenizer, int(response[p]))]
    rng.shuffle(positions)
    positions = positions[:args.positions]
    if not positions:
        return None
    full = torch.cat([prefix, response])
    logits = logits_at(model, tokenizer, [full],
                       [(0, len(prefix) + pos - 1) for pos in positions], model.device)
    for pos, row_logits in zip(positions, logits):
        probs = row_logits.softmax(dim=-1)
        original = int(response[pos])
        orig_prob = float(probs[original])
        if orig_prob > args.max_orig_prob:
            continue
        band = ((probs >= args.p_min) & (probs <= args.p_max)).nonzero(as_tuple=True)[0]
        if band.numel() == 0:
            continue
        order = torch.argsort(probs[band], descending=True)[:args.scan]
        old_text = tokenizer.decode([original]).strip().casefold()
        survivors = []
        for index in order.tolist():
            replacement = int(band[index])
            if replacement == original or not eligible(tokenizer, replacement):
                continue
            new_text = tokenizer.decode([replacement])
            if new_text.strip().casefold() == old_text:
                continue
            survivors.append(replacement)
            if len(survivors) >= args.top_k:
                break
        if survivors:
            replacement = rng.choice(survivors)
            return {"token_index": pos, "orig_token_id": original,
                    "orig_token_text": tokenizer.decode([original]),
                    "orig_prob": orig_prob, "new_token_id": replacement,
                    "new_token_text": tokenizer.decode([replacement]),
                    "new_prob": float(probs[replacement]),
                    "surprisal_nats": -math.log(float(probs[replacement]))}
    return None


def insert_plant(model, tokenizer, prefix, response, plant: dict, args, generation):
    import torch
    from prefill_awareness.data_gen import batch_generate

    pos = plant["token_index"]
    head = torch.cat([prefix, response[:pos],
                      torch.tensor([plant["new_token_id"]], device=model.device)])
    remaining = args.max_new_tokens - pos - 1
    tail = batch_generate(model, tokenizer, [head], generation, remaining, model.device)[0]
    return torch.cat([response[:pos],
                      torch.tensor([plant["new_token_id"]], device=model.device),
                      tail[:remaining]])


def make_case(row: dict, mode: str, kind: str, response, plants: list[dict], tokenizer, args):
    return {"id": f"{row['id']}-{mode}-{kind}", "prompt_id": row["id"],
            "group": row["group"], "prompt": row["prompt"],
            "mode": mode, "kind": kind, "enable_thinking": False,
            "answer": tokenizer.decode(response, skip_special_tokens=True),
            "answer_token_ids": response.tolist(),
            "gold_tokens": [p["new_token_text"] for p in plants], "plants": plants,
            "truncated": len(response) >= args.max_new_tokens}


def main() -> None:
    import torch
    from prefill_awareness.data_gen import batch_generate
    from prefill_awareness.evaluate import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--min-response-tokens", type=int, default=40)
    parser.add_argument("--min-pos", type=int, default=8)
    parser.add_argument("--min-tail", type=int, default=8)
    parser.add_argument("--min-gap", type=int, default=16)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--scan", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--p-min", type=float, default=1e-4)
    parser.add_argument("--p-max", type=float, default=1e-2)
    parser.add_argument("--max-orig-prob", type=float, default=0.9)
    parser.add_argument("--two-n", type=int, default=16,
                        help="attempt two plants on the first N prompts in each mode")
    args = parser.parse_args()
    prompts = read_jsonl(Path(args.prompts))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(target) if target.exists() else []
    done = {r["id"] for r in existing if r.get("kind") == "status"}
    model, tokenizer = load_model(args.model, None, args.dtype)
    counts = {"clean": 0, "one": 0, "two": 0, "failed": 0}
    with target.open("a", encoding="utf-8") as handle:
        for mode in ("greedy", "sampled"):
            generation = SimpleNamespace(greedy=mode == "greedy", top_p=0.9, temperature=1.0)
            for index, row in enumerate(prompts):
                status_id = f"{row['id']}-{mode}-status"
                if status_id in done:
                    continue
                seed = seed_for(row["id"], mode, args.seed)
                torch.manual_seed(seed)
                rng = random.Random(seed)
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                prefix = tokenizer(prompt_text, add_special_tokens=False,
                                   return_tensors="pt").input_ids[0].to(model.device)
                response = batch_generate(model, tokenizer, [prefix], generation,
                                          args.max_new_tokens, model.device)[0]
                cases = []
                if len(response) >= args.min_response_tokens:
                    first = pick_plant(model, tokenizer, prefix, response, args, rng)
                    if first is not None:
                        cases.append(make_case(row, mode, "clean", response, [], tokenizer, args))
                        one = insert_plant(model, tokenizer, prefix, response, first, args, generation)
                        cases.append(make_case(row, mode, "one", one, [first], tokenizer, args))
                        if index < args.two_n:
                            second = pick_plant(model, tokenizer, prefix, one, args, rng,
                                                after=first["token_index"] + args.min_gap)
                            if second is not None:
                                two = insert_plant(model, tokenizer, prefix, one, second,
                                                   args, generation)
                                cases.append(make_case(row, mode, "two", two,
                                                       [first, second], tokenizer, args))
                if not cases:
                    counts["failed"] += 1
                for case in cases:
                    counts[case["kind"]] += 1
                    handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                handle.write(json.dumps({"id": status_id, "kind": "status",
                                         "prompt_id": row["id"], "mode": mode,
                                         "n_cases": len(cases)}) + "\n")
                handle.flush()
                print(f"{mode} {index + 1}/{len(prompts)}: {len(cases)} cases", flush=True)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
