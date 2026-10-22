"""Check flash-linear-attention on a compute node before using it for data.

1. The Triton kernels compile and run: chunked vs recurrent gated delta rule on
   random inputs (two different algorithms, which must agree).
2. Qwen3.8 with the kernels scores the pilot's tampered answers like the
   reference PyTorch path did: the plant surprisals stored by the pilot (scored
   without fla) are recomputed and compared.

    FLA=1 python condor/fla_check.py data/tamper-qwen38/pilot-histories.jsonl
"""

import json
import sys

import torch


def kernels():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    torch.manual_seed(0)
    b, t, h, d = 2, 300, 4, 128
    q, k = (torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    v = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(b, t, h, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
    o1, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    o2, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    diff = (o1.float() - o2.float()).abs().max().item()
    print(f"kernels ok: chunk vs recurrent max |diff| = {diff:.4f} (output scale {o1.float().abs().mean():.3f})")


def rescoring(path, n=12):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prefill_awareness.tamper_gen import surprisal_check

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    args = type("A", (), {"temperature": 0.7, "top_p": 0.8, "top_k": 20})()
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if r.get("label") == "tampered"][:n]
    diffs = []
    for r in recs:
        prefix = tok(tok.apply_chat_template(r["messages"][:1], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False), add_special_tokens=False).input_ids
        ids = tok(r["messages"][1]["content"], add_special_tokens=False).input_ids
        with torch.no_grad():
            new = surprisal_check(model, tok, prefix, ids, [p["token_index"] for p in r["plants"]], args, "cuda")
        old = r["surprisal_check"]["plant_surprisals"]
        diffs += [abs(a - b) for a, b in zip(new["plant_surprisals"], old)]
        print(r["id"], "stored", old, "fla", new["plant_surprisals"], "ranks", new["plant_ranks"])
    print(f"plant surprisal |fla - reference|: mean {sum(diffs) / len(diffs):.3f}, max {max(diffs):.3f} nats")


if __name__ == "__main__":
    import transformers.models.qwen3_5.modeling_qwen3_5 as m  # noqa: F401  (import check)

    kernels()
    rescoring(sys.argv[1])
