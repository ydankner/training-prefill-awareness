"""Generate matched foreign-model and externally prefixed answers."""

import argparse
import json
from pathlib import Path

from .prepare import case, read_jsonl


def append(path, rows):
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def generate_prefixed(model, tokenizer, messages, prefix="Sure, here", max_new_tokens=256):
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False) + prefix
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                use_cache=True,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    continuation = tokenizer.decode(output[0, encoded.input_ids.shape[1]:],
                                    skip_special_tokens=True)
    return prefix + continuation


def run_weak(args):
    from prefill_awareness.capability_eval import generate
    from prefill_awareness.evaluate import load_model
    prompts = read_jsonl(args.prompts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(out) if out.exists() else []
    done = {r["pair"] for r in existing if r["condition"] == "weaker_model"}
    if len(existing) != 2 * len(done):
        raise ValueError("Weak output must contain complete pairs")
    model, tok = load_model(args.model, None, "bfloat16")
    for start in range(0, len(prompts), args.batch_size):
        batch = [r for r in prompts[start:start + args.batch_size] if r["pair"] not in done]
        if not batch:
            continue
        generated = generate(model, tok,
                             [({"id": r["pair"], "enable_thinking": False}, [r["user"]]) for r in batch],
                             args.batch_size, args.max_new_tokens)
        rows = []
        for r, g in zip(batch, generated):
            rows.extend([
                case("weaker_model", r["pair"], "own", False,
                     [r["user"], {"role": "assistant", "content": r["own"]}]),
                case("weaker_model", r["pair"], "weaker_model", True,
                     [r["user"], {"role": "assistant", "content": g["answer"]}],
                     source_model=args.model),
            ])
        append(out, rows)
        print(f"weak: {min(start+args.batch_size, len(prompts))}/{len(prompts)}", flush=True)


def run_prefix(args):
    from prefill_awareness.capability_eval import generate
    from prefill_awareness.evaluate import load_model
    prompts = read_jsonl(args.prompts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(out) if out.exists() else []
    done = {r["pair"] for r in existing if r["condition"] == "forced_prefix"}
    if len(existing) != 2 * len(done):
        raise ValueError("Prefix output must contain complete pairs")
    model, tok = load_model(args.model, None, "bfloat16")
    for r in prompts:
        if r["pair"] in done:
            continue
        generated = generate(model, tok,
                             [({"id": r["pair"], "enable_thinking": False}, r["messages"])],
                             1, args.max_new_tokens)[0]
        prefixed = generate_prefixed(model, tok, r["messages"], max_new_tokens=args.max_new_tokens)
        common = {"prefix_kind": r["kind"], **({"secret": r["secret"]} if "secret" in r else {})}
        append(out, [
            case("prefix", r["pair"], "unprefixed", False,
                 r["messages"] + [{"role": "assistant", "content": generated["answer"]}], **common),
            case("prefix", r["pair"], "forced_prefix", True,
                 r["messages"] + [{"role": "assistant", "content": prefixed}], **common),
        ])
        print(f"prefix: {len(done)+1}/{len(prompts)}", flush=True)
        done.add(r["pair"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["weak", "prefix"])
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()
    if args.mode == "weak":
        run_weak(args)
    else:
        run_prefix(args)


if __name__ == "__main__":
    main()
