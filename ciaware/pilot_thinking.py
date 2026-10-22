"""Pilot: does the tamper marker fire on the model's own clean essays when they
were written with thinking and the reasoning is then removed?

CIAware-Bench's essay setting (``ciaware/CIAware-Bench``, our fork) writes a
10-paragraph essay one paragraph per turn, then asks a detection probe. Our
marker detector was trained without thinking. If essays are generated with
thinking and read back without it, every paragraph was conditioned on hidden
reasoning the reader cannot see, which may look like tampering to it. This
pilot measures that on clean essays only (no intervention at all):

``think``    history generated as the benchmark would: thinking on, reasoning
             effort "low" (Qwen3.8's template), temperature 1.0 (the repo sends
             only that) with Qwen's recommended thinking-mode truncation
             (top-p 0.95, top-k 20), which a vLLM or OpenRouter deployment picks
             up from generation_config.json. Pure T=1 (the first pilot run, job
             17622145) left 22 of 24 essays without a closing ``</think>``.
             Earlier turns' reasoning is never shown again, as with an API.
``nothink``  control: our sampling (non-thinking, T 0.7, top-p 0.8, top-k 20).
``nothink_t1`` control for the sampler: non-thinking, but the ``think`` sampling
             (T 1.0, top-p 0.95, top-k 20). The detector flags tokens outside
             the T=0.7 nucleus, which hotter sampling produces without any
             hidden reasoning.

Readout, per essay and per clean truncation after k = 2..10 paragraphs (the
benchmark's clean probe truncates after 1..9 paragraphs before an intervention
at paragraph 3..10): the history rendered without reasoning, then the
benchmark's essay detection probe as the next user turn, non-thinking. We
record P(marker) at the first reply token and a short greedy reply, for the
adapter and the base model (adapter disabled). The essays are written by the
adapter model, so the base readout is a reference, not a benchmark condition.
``think_sys`` reads the ``think`` essays with the template's effort line kept in
the system prompt, so a firing marker can be put on the hidden reasoning rather
than on the changed system prompt. An essay that
fails (truncated turn or no ``</think>``) stops there; its completed paragraphs
are still probed, and the failures are listed in meta.json.

Outputs ``<out>/essays-<cond>.jsonl``, ``<out>/probe-rows.jsonl``,
``<out>/summary.json`` and ``<out>/meta.json``.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from cia.environments.essay.prompts import build_system_prompt, paragraph_prompt
from cia.probing.prompts import build_probe
from prefill_awareness.data_gen import strip_terminators
from prefill_awareness.evaluate import load_model
from prefill_awareness.provenance import describe, require_committed, utc_now
from prefill_awareness.tamper_eval import generate
from prefill_awareness.tamper_formats import MARKER, has_marker

BENCH = Path(__file__).resolve().parent / "CIAware-Bench"
TOPICS = BENCH / "cia/environments/essay/topics.txt"
THINK_END = "</think>"
# The line Qwen3.8's template puts before the system prompt with thinking on at
# effort "low"; a non-thinking render drops it.
EFFORT_LOW = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
              "the conclusion without unnecessary elaboration.")
# Sampling per condition, every warper explicit so the model's generation_config
# cannot add its own.
SAMPLING = {
    "think": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": None},
    "nothink": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": None},
    "nothink_t1": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": None},
}


def render_turn(tokenizer, messages, think: bool) -> str:
    # History assistant messages carry no reasoning_content, so the template
    # writes them with an empty think block in both conditions.
    kwargs = {"enable_thinking": True, "reasoning_effort": "low"} if think else {"enable_thinking": False}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


@torch.no_grad()
def write_essays(model, tokenizer, topics, cond, args):
    think = cond == "think"
    system = build_system_prompt("default", args.steps)
    essays = [{"topic": t, "messages": [{"role": "system", "content": system}], "reasoning": [],
               "reasoning_tokens": [], "failed": None} for t in topics]
    for step in range(args.steps):
        live = [e for e in essays if e["failed"] is None]
        if not live:
            break
        for e in live:
            e["messages"].append({"role": "user", "content": paragraph_prompt(step, args.steps, e["topic"])})
        prompts = [render_turn(tokenizer, e["messages"], think) for e in live]
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(args.seed * 1000 + step + 500 * list(SAMPLING).index(cond))
        out = model.generate(**enc, max_new_tokens=args.think_tokens if think else args.answer_tokens,
                             do_sample=True, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                             **SAMPLING[cond])
        for e, row in zip(live, out[:, enc["input_ids"].shape[1]:]):
            ids = strip_terminators(row, tokenizer)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            truncated = int(ids.shape[0]) >= (args.think_tokens if think else args.answer_tokens)
            reasoning = ""
            if think:
                if THINK_END not in text:
                    e["failed"] = f"no </think> in paragraph {step + 1}"
                    e["failed_output"] = {"text": text, "tokens": int(ids.shape[0]), "truncated": truncated}
                    e["messages"].pop()
                    continue
                reasoning, text = text.split(THINK_END, 1)
                e["reasoning"].append(reasoning.strip())
                e["reasoning_tokens"].append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
            text = text.strip()
            if truncated or not text:
                e["failed"] = f"{'truncated' if truncated else 'empty'} paragraph {step + 1}"
                e["failed_output"] = {"text": text, "tokens": int(ids.shape[0]), "truncated": truncated}
                e["messages"].pop()
                continue
            if MARKER in text:
                e.setdefault("marker_in_paragraph", []).append(step + 1)
            e["messages"].append({"role": "assistant", "content": text})
        print(f"{cond} paragraph {step + 1}/{args.steps}: {sum(e['failed'] is None for e in essays)} live",
              flush=True)
    return essays


def probe_records(essays, cond, probe, steps):
    records = []
    for i, e in enumerate(essays):
        n_par = sum(m["role"] == "assistant" for m in e["messages"])
        for k in range(2, steps + 1):
            if k > n_par:
                break
            # system + k (user, assistant) pairs, then the probe; the final
            # placeholder is dropped by generate().
            history = e["messages"][:1 + 2 * k]
            records.append({"id": f"{cond}-{i}-k{k}", "cond": cond, "essay": i, "k": k,
                            "messages": history + [{"role": "user", "content": probe},
                                                   {"role": "assistant", "content": ""}]})
            if cond == "think":
                # Same history, but the system prompt keeps the effort line it was
                # generated under: isolates hidden reasoning from the prompt change.
                system = {"role": "system", "content": EFFORT_LOW + "\n\n" + history[0]["content"]}
                records.append({**records[-1], "id": f"think_sys-{i}-k{k}", "cond": "think_sys",
                                "messages": [system] + records[-1]["messages"][1:]})
    return records


def summarize(rows):
    out = {}
    for model_name in sorted({r["model"] for r in rows}):
        for cond in sorted({r["cond"] for r in rows}):
            sel = [r for r in rows if r["model"] == model_name and r["cond"] == cond]
            if not sel:
                continue
            p = [r["p_marker"] for r in sel]
            out[f"{model_name}/{cond}"] = {
                "n": len(sel),
                "marker_emitted": sum(r["marker"] for r in sel) / len(sel),
                "p_marker_mean": statistics.fmean(p),
                "p_marker_median": statistics.median(p),
                "frac_p_marker_gt_0.5": sum(x > 0.5 for x in p) / len(p),
                "reply_starts_json": sum(r["reply"].lstrip(MARKER + " ").startswith("{") for r in sel) / len(sel),
                "p_marker_by_k": {k: statistics.fmean(r["p_marker"] for r in sel if r["k"] == k)
                                  for k in sorted({r["k"] for r in sel})},
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=24, help="essays per condition")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--think-tokens", type=int, default=16384, help="cap per thinking turn (reasoning + paragraph)")
    parser.add_argument("--answer-tokens", type=int, default=400)
    parser.add_argument("--reply-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    git = require_committed(parser, args.allow_dirty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topics = [t.strip() for t in TOPICS.read_text(encoding="utf-8").splitlines() if t.strip()]
    topics = random.Random(args.seed).sample(topics, args.n)
    probe = build_probe("detection", "essay", policy="honest", level="default")
    model, tokenizer = load_model(args.model, args.adapter)
    meta = {**describe("ciaware.pilot_thinking", args, git, model), "sampling": SAMPLING, "probe": probe,
            "generation_config": model.generation_config.to_dict()}

    records = []
    for cond in ("nothink", "nothink_t1", "think"):
        path = out / f"essays-{cond}.jsonl"
        if path.exists():
            essays = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        else:
            # The tuned model writes its own essays, as the benchmark's untrusted
            # model would; the base readout below is only a reference.
            essays = write_essays(model, tokenizer, topics, cond, args)
            path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in essays), encoding="utf-8")
        meta[f"essays_{cond}"] = {
            "n": len(essays), "failed": [e["failed"] for e in essays if e["failed"]],
            "marker_in_paragraph": sum(bool(e.get("marker_in_paragraph")) for e in essays),
            "reasoning_tokens_mean": (statistics.fmean(t for e in essays for t in e["reasoning_tokens"])
                                      if cond == "think" else None)}
        records += probe_records(essays, cond, probe, args.steps)

    rows = []
    for name, disable in (("tuned", False), ("base", True)):
        context = model.disable_adapter() if disable else torch.no_grad()
        with context:
            texts, p_marker = generate(model, tokenizer, records, args.reply_tokens, args.batch_size)
        rows += [{"id": r["id"], "model": name, "cond": r["cond"], "essay": r["essay"], "k": r["k"],
                  "p_marker": p, "marker": has_marker(t), "reply": t}
                 for r, t, p in zip(records, texts, p_marker)]
        print(name, "done", flush=True)
    (out / "probe-rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    meta["finished_at"] = utc_now()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
