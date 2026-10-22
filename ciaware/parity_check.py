"""Why do HF and vLLM disagree on P(marker)? Same token ids, three ways.

The vLLM smoke tests give the marker adapter P(marker) ~5e-8 on the pilot's
essay probes, with the LoRA and with merged weights, where the HF pilot read
0.15 (base agrees, ~0). HF scored left-padded batches (tamper_eval.generate);
Qwen3.8's linear-attention layers may not ignore the padding. This scores the
same prompts unpadded (batch 1) and padded (batches of 16, as the pilot did),
for essay probes and for detection test conversations of the main dataset.

    hf    python -m ciaware.parity_check hf --adapter A --essays DIR --conversations C --out OUT
    vllm  python -m ciaware.parity_check vllm --model MERGED --out OUT   (reads OUT/prompts.json)
"""

import argparse
import json
import math
from pathlib import Path


def build_prompts(tokenizer, essays_dir, conversations, n):
    from cia.probing.prompts import build_probe

    from prefill_awareness.dataset import render

    probe = build_probe("detection", "essay", policy="honest", level="default")
    items = []
    essays = [json.loads(x) for x in (Path(essays_dir) / "essays-nothink.jsonl").read_text(encoding="utf-8").splitlines()]
    for i, e in enumerate(essays[: n // 4]):
        for k in (3, 6, 9):
            msgs = e["messages"][: 1 + 2 * k] + [{"role": "user", "content": probe}]
            items.append({"kind": "essay_probe", "id": f"essay{i}-k{k}", "label": "clean",
                          "text": render(tokenizer, msgs, False, add_generation_prompt=True)})
    convs = [json.loads(x) for x in Path(conversations).read_text(encoding="utf-8").splitlines() if x.strip()]
    det = [c for c in convs if c.get("split") == "test" and c["messages"][2]["content"].startswith("Was your previous")]
    for label in ("tampered", "clean"):
        for c in [c for c in det if c["label"] == label][: n // 2]:
            items.append({"kind": "detection", "id": c.get("history_id"), "label": label,
                          "text": render(tokenizer, c["messages"][:3], False, add_generation_prompt=True)})
    for it in items:
        it["ids"] = tokenizer(it["text"], add_special_tokens=False).input_ids
    return items


def hf(args):
    import torch

    from prefill_awareness.evaluate import load_model

    model, tokenizer = load_model(args.model, args.adapter)
    marker = tokenizer.encode("¦", add_special_tokens=False)[0]
    items = build_prompts(tokenizer, args.essays, args.conversations, args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for it in items:  # unpadded
            ids = torch.tensor([it["ids"]], device=model.device)
            it["hf_single"] = float(torch.softmax(model(ids).logits[0, -1].float(), -1)[marker])
        tokenizer.padding_side = "left"
        for s in range(0, len(items), 16):  # left-padded batches, like tamper_eval.generate
            batch = items[s:s + 16]
            enc = tokenizer([b["text"] for b in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            logits = model(**enc).logits[:, -1].float()
            for b, p in zip(batch, torch.softmax(logits, -1)[:, marker].tolist()):
                b["hf_batched"] = p
    (out / "prompts.json").write_text(json.dumps(items) + "\n")
    report(items)


def vllm(args):
    from vllm import LLM, SamplingParams, TokensPrompt

    out = Path(args.out)
    items = json.loads((out / "prompts.json").read_text())
    llm = LLM(model=args.model, max_model_len=16384, gpu_memory_utilization=0.9, max_num_seqs=256,
              limit_mm_per_prompt={"image": 0, "video": 0})
    marker = 62438
    outs = llm.generate([TokensPrompt(prompt_token_ids=it["ids"]) for it in items],
                        SamplingParams(max_tokens=1, logprobs=0, allowed_token_ids=[marker]))
    for it, o in zip(items, outs):
        it["vllm"] = math.exp(o.outputs[0].logprobs[0][marker].logprob)
    (out / "prompts.json").write_text(json.dumps(items) + "\n")
    report(items)


def report(items):
    for kind in ("essay_probe", "detection"):
        for label in ("tampered", "clean"):
            sel = [i for i in items if i["kind"] == kind and i["label"] == label]
            if sel:
                line = {k: round(sum(i[k] for i in sel) / len(sel), 4) for k in ("hf_single", "hf_batched", "vllm")
                        if k in sel[0]}
                print(kind, label, len(sel), json.dumps(line), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", choices=("hf", "vllm"))
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter")
    parser.add_argument("--essays")
    parser.add_argument("--conversations")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    {"hf": hf, "vllm": vllm}[args.engine](args)


if __name__ == "__main__":
    main()
