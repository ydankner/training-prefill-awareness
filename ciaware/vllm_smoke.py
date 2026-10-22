"""vLLM smoke test for the CIAware runs: can vLLM serve Qwen3.8 + the marker
adapter, does it read P(marker) like the HF path, and how fast is it?

1. Adapter: served as a LoRA, or ``--merged`` weights from ciaware.merge_adapter
   (vLLM 0.30 does not apply the LoRA when scoring the first reply token).
2. Parity: P(marker as first reply token) on the pilot's probe records
   (``runs/ciaware-pilot-thinking-v2``), computed as the prompt log-prob of the
   marker appended to the rendered prompt, vs the HF values in probe-rows.jsonl.
3. Throughput: 24 essays x 10 paragraphs, non-thinking, our sampling; and
   thinking-mode probe replies (effort low, T 1.0, top-p 0.95, top-k 20).

Writes ``<out>/smoke.json``.
"""

import argparse
import json
import math
import statistics
import time
from importlib import metadata
from pathlib import Path

from cia.environments.essay.prompts import build_system_prompt, paragraph_prompt
from cia.probing.prompts import build_probe

MARKER = "¦"
SAMPLING = {"nothink": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0),
            "think": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)}


def render(tokenizer, messages, think: bool, effort="low"):
    kwargs = {"enable_thinking": True, "reasoning_effort": effort} if think else {"enable_thinking": False}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def marker_probs(llm, tokenizer, prompts, marker_id, lora=None):
    from vllm import SamplingParams, TokensPrompt

    # One token forced to the marker; its raw log-prob (prompt log-probs skip the LoRA in vLLM 0.30).
    inputs = [TokensPrompt(prompt_token_ids=tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    outs = llm.generate(inputs, SamplingParams(max_tokens=1, logprobs=0, allowed_token_ids=[marker_id]),
                        lora_request=lora)
    return [math.exp(o.outputs[0].logprobs[0][marker_id].logprob) for o in outs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--pilot", required=True, help="pilot run dir (essays-*.jsonl, probe-rows.jsonl)")
    parser.add_argument("--merged", default=None, help="merged checkpoint (no LoRA)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-parity", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"versions": {n: metadata.version(n) for n in ("vllm", "torch", "transformers")}}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    marker_id = tokenizer.convert_tokens_to_ids(MARKER) if MARKER in tokenizer.get_vocab() else \
        tokenizer.encode(MARKER, add_special_tokens=False)[0]
    common = dict(max_model_len=16384, gpu_memory_utilization=0.9, max_num_seqs=256,
                  limit_mm_per_prompt={"image": 0, "video": 0})

    t0 = time.time()
    if args.merged:  # adapter already merged into the weights (ciaware.merge_adapter)
        llm = LLM(model=args.merged, **common)
        report["adapter"] = f"merged: {args.merged}"
        tuned, base = (llm, None), None
    else:
        llm = LLM(model=args.model, enable_lora=True, max_lora_rank=16, max_loras=1, **common)
        from ciaware.vllm_lora import convert

        lora = LoRARequest("tamper", 1, convert(args.adapter, Path(args.out) / "adapters"))
        report["adapter"] = "lora"
        tuned, base = (llm, lora), (llm, None)
    report["load_seconds"] = round(time.time() - t0)
    print(json.dumps(report), flush=True)

    # 2. Parity with the HF pilot readout.
    rows = [json.loads(line) for line in (Path(args.pilot) / "probe-rows.jsonl").read_text(encoding="utf-8").splitlines()]
    essays = {c: [json.loads(line) for line in (Path(args.pilot) / f"essays-{c}.jsonl").read_text(encoding="utf-8").splitlines()]
              for c in ("nothink", "nothink_t1", "think")}
    probe = build_probe("detection", "essay", policy="honest", level="default")
    picked = [r for r in rows if r["cond"] in essays]
    parity = {}
    for name, (engine, lora) in (("tuned", tuned), ("base", base or (None, None))):
        sel = [r for r in picked if r["model"] == name][: args.n_parity]
        if engine is None or not sel:
            continue
        prompts = [render(tokenizer, essays[r["cond"]][r["essay"]]["messages"][:1 + 2 * r["k"]]
                          + [{"role": "user", "content": probe}], False) for r in sel]
        p = marker_probs(engine, tokenizer, prompts, marker_id, lora)
        hf = [r["p_marker"] for r in sel]
        diffs = [abs(a - b) for a, b in zip(p, hf)]
        parity[name] = {"n": len(sel), "mean_abs_diff": statistics.fmean(diffs), "max_abs_diff": max(diffs),
                        "mean_vllm": statistics.fmean(p), "mean_hf": statistics.fmean(hf)}
    report["parity"] = parity
    print(json.dumps(parity), flush=True)

    # 3. Throughput: essays (non-thinking) and thinking probe replies.
    engine, lora = tuned
    system = build_system_prompt("default", 10)
    topics = [e["topic"] for e in essays["nothink"]]
    convs = [[{"role": "system", "content": system}] for _ in topics]
    t0, n_tok = time.time(), 0
    for step in range(10):
        for c, t in zip(convs, topics):
            c.append({"role": "user", "content": paragraph_prompt(step, 10, t)})
        outs = engine.generate([render(tokenizer, c, False) for c in convs],
                               SamplingParams(max_tokens=400, seed=step, **SAMPLING["nothink"]), lora_request=lora)
        for c, o in zip(convs, outs):
            c.append({"role": "assistant", "content": o.outputs[0].text.strip()})
            n_tok += len(o.outputs[0].token_ids)
    report["essays"] = {"n": len(convs), "seconds": round(time.time() - t0), "tokens": n_tok,
                        "markers_in_paragraphs": sum(MARKER in m["content"] for c in convs for m in c[1:])}
    prompts = [render(tokenizer, c[:1 + 2 * k] + [{"role": "user", "content": probe}], True)
               for c in convs for k in (3, 6, 9)]
    t0 = time.time()
    outs = engine.generate(prompts, SamplingParams(max_tokens=4096, seed=0, **SAMPLING["think"]), lora_request=lora)
    texts = [o.outputs[0].text for o in outs]
    report["think_probes"] = {"n": len(outs), "seconds": round(time.time() - t0),
                              "tokens": sum(len(o.outputs[0].token_ids) for o in outs),
                              "closed_think": sum("</think>" in t for t in texts),
                              "marker_runs": sum(MARKER * 5 in t for t in texts),
                              "examples": [t[:300] for t in texts[:3]]}
    (out / "smoke.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "think_probes"}, indent=1), flush=True)
    print(json.dumps(report["think_probes"], indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
