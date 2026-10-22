"""Generation-based evaluation on the held-out test split.

Scored the way the task is actually posed: the model is given the chat up to and
including the detection question, generates freely, and its reply is parsed.
Nothing is teacher-forced, so this measures the ability we care about rather
than the likelihood of the gold string.

Metrics
    detect_acc  -- binary tampered/clean accuracy over the whole set
    tpr         -- of modified examples, the fraction flagged as modified
    fpr         -- of clean examples, the fraction wrongly flagged
    loc_acc     -- of *all* modified examples, the fraction where the quoted
                   token is exactly right. This is the headline number: it is
                   the ability that binary detection work does not have.
    loc_acc_hit -- localisation accuracy among the modified examples that were
                   detected at all, i.e. how good the pointing is once the model
                   has decided something is wrong.

Run standalone to get the pre-SFT baseline, which is the number the whole
experiment is measured against:

    python -m prefill_awareness.evaluate --model <name> --data data/ds.jsonl
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset import load_records, render, thinking_for
from .formats import ANSWER_FORMATS, normalize_token, parse_answer, split_pairs


@torch.no_grad()
def run_eval(model, tokenizer, records, max_new_tokens=32, batch_size=8, n_samples=3,
             enable_thinking: bool | None = None, thinking_mode: str = "source"):
    """Generate and score answers; optionally override the source record's thinking mode."""
    was_training = model.training
    model.eval()
    # Left padding so that every sequence in the batch ends at the generation
    # point; with right padding the model would continue from pad tokens.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    generations = []
    for start in tqdm(range(0, len(records), batch_size), desc="eval", leave=False):
        batch = records[start:start + batch_size]
        prompts = []
        for record in batch:
            mode = enable_thinking if enable_thinking is not None else thinking_for(record, thinking_mode)
            prompts.append(render(
                tokenizer, record["messages"][:-1], mode,
                record.get("keep_think", False), add_generation_prompt=True,
            ))
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Trainer turns caching off for gradient checkpointing; we are in
            # eval mode here, so re-enable it or generation crawls.
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for row in out[:, encoded["input_ids"].shape[1]:]:
            generations.append(tokenizer.decode(row, skip_special_tokens=True).strip())

    tokenizer.padding_side = prev_side
    if was_training:
        model.train()

    n_mod = n_clean = detected = false_alarm = correct = located = compliant = 0
    orig_named = orig_named_located = 0
    pairs = any(r.get("answer_format") == "plant-orig" for r in records)
    for record, text in zip(records, generations):
        claims, predicted = parse_answer(text)
        originals = []
        if record.get("answer_format") == "plant-orig":
            # Items are "changed -> original"; localisation is scored on the
            # changed side exactly as in the default format.
            predicted, originals = split_pairs(predicted)
        is_modified = record["label"] == "modified"
        correct += int(claims == is_modified)
        # The dictated format is single tokens separated by commas, so any
        # "token" containing whitespace means the model answered in prose
        # instead. Tracked separately because an untrained model mostly ignores
        # the format, and scoring that as a false alarm would confuse
        # "cried wolf" with "never answered the question".
        compliant += int(all(" " not in t for t in predicted))
        if is_modified:
            n_mod += 1
            detected += int(claims)
            # Compared as an ordered list so this still scores correctly if more
            # than one token is ever planted per turn.
            gold = [normalize_token(record["modification"]["new_token_text"])]
            hit = [normalize_token(t) for t in predicted] == gold
            located += int(hit)
            if pairs:
                orig = originals == [normalize_token(record["modification"]["orig_token_text"])]
                orig_named += int(orig)
                orig_named_located += int(orig and hit)
        else:
            n_clean += 1
            false_alarm += int(claims)

    metrics = {
        "n": len(records),
        "n_modified": n_mod,
        "detect_acc": correct / max(len(records), 1),
        "tpr": detected / max(n_mod, 1),
        "fpr": false_alarm / max(n_clean, 1),
        "loc_acc": located / max(n_mod, 1),
        "loc_acc_hit": located / max(detected, 1),
        "format_ok": compliant / max(len(records), 1),
    }
    if pairs:
        # orig_acc: of all modified examples, the original token named exactly;
        # orig_acc_hit: the same among those whose plant was located.
        metrics["orig_acc"] = orig_named / max(n_mod, 1)
        metrics["orig_acc_hit"] = orig_named_located / max(located, 1)
    metrics["samples"] = [
        {
            "id": record.get("id"),
            "label": record["label"],
            "gold": record["messages"][-1]["content"],
            "generated": text,
        }
        for record, text in list(zip(records, generations))[:n_samples]
    ]
    return metrics


def load_model(model_name: str, adapter: str | None, dtype: str = "bfloat16", load_4bit: bool = False,
               attn_implementation: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # "auto" shards across GPUs when there are several, but with no GPU visible
    # accelerate reads it as "offload everything to disk" and raises. Be explicit.
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    kwargs = {"dtype": getattr(torch, dtype), "device_map": device_map}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prefill detection on the test split.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory; omit for the baseline")
    parser.add_argument("--data", required=True, nargs="+", help=".jsonl files or directories of shards")
    parser.add_argument("--split", default="test", help="'all' evaluates every record")
    parser.add_argument("--n", type=int, default=256, help="cap on evaluated examples")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--thinking-mode", choices=("source", "on", "off", "omit"),
                        default="source", help="chat mode of the evaluated model")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true",
                          help="use the evaluated model's thinking mode, overriding dataset metadata")
    thinking.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false",
                          help="use the evaluated model's non-thinking mode, overriding dataset metadata")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--out", default=None, help="optional path to write metrics as JSON")
    parser.add_argument("--keep-think", action="store_true",
                        help="keep the empty think block in the tampered turn (see dataset.render)")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="plant",
                        help="plant-orig: also name the original token (formats.apply_answer_format)")
    args = parser.parse_args()

    records = load_records(args.data, None if args.split == "all" else args.split,
                           args.keep_think, args.answer_format)[: args.n]
    model, tokenizer = load_model(args.model, args.adapter, args.dtype, args.load_4bit)
    # Keep every generation when writing to disk; print only a few.
    metrics = run_eval(
        model, tokenizer, records, args.max_new_tokens, args.batch_size,
        n_samples=len(records) if args.out else 3,
        enable_thinking=args.enable_thinking,
        thinking_mode=args.thinking_mode,
    )

    samples = metrics.pop("samples")
    print(json.dumps(metrics, indent=2))
    for sample in samples[:3]:
        print(f"\n[{sample['label']}]\n  gold: {sample['gold']}\n  got : {sample['generated']}")
    if args.out:
        metrics["samples"] = samples
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
