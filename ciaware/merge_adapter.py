"""Merge a marker adapter into Qwen3.8 and save a text-only checkpoint for vLLM.

vLLM 0.30 does not apply our LoRA when it scores the first reply token (smoke
tests 17652316/17652337: P(marker) ~5e-8 with the LoRA vs 0.15 under HF; base
matches HF), so the CIAware runs serve merged weights instead. Runs in the HF
training venv (the path the adapter was trained and evaluated with); the result
is a ``Qwen3_5ForCausalLM`` checkpoint, which vLLM loads natively.

    python -m ciaware.merge_adapter --adapter runs/tamper-qwen38-main/adapter --out $SCRATCH/merged-main
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "merged_from.json").exists():
        print(f"{out} exists", flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    (out / "merged_from.json").write_text(json.dumps({"model": args.model, "adapter": str(Path(args.adapter).resolve()),
                                                      "architectures": model.config.architectures}) + "\n")
    print(json.dumps({"out": str(out), "architectures": model.config.architectures}), flush=True)


if __name__ == "__main__":
    main()
