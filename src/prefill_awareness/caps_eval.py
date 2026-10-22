"""Does ALL-CAPS training in prefilled history carry over to the model's own history?

Evaluates every checkpoint of a caps run (see ``caps_data``), plus the model
before caps training (step 0), on the test conversations under three histories:

    planted  -- turn 1 is the modified answer with a planted token (as trained)
    own      -- turn 1 is the model's own clean greedy answer to the same prompt
    none     -- no turn 1, the second prompt alone
    human    -- the Dolly author's answer (records built with ``--history human``)

``--histories`` picks which; the history a record was trained on is its turn 1,
the others come from the record's ``<name>_history`` fields.

Two measures per record:

    caps     -- greedy reply (``--max-new-tokens``), fraction of cased letters that
                are uppercase; ``caps_rate`` counts replies at >= 0.9
    logratio -- teacher-forced log P(uppercased reply prefix) - log P(normal reply
                prefix), the first ``--prefix-words`` words of the model's own
                answer. Continuous, so it still moves where greedy is saturated.

The detection fine-tune is compared by passing its adapter as ``--init-adapter``;
it is merged before the caps adapters are loaded, exactly as in training.

    python -m prefill_awareness.caps_eval --model Qwen/Qwen3-32B \
        --run runs/caps-... --data data/caps-qwen3-32b.jsonl [--init-adapter ...]

``--marker`` (tampering-marker model, ``tamper_caps`` data): caps is measured on
the reply after the marker ``¦``; ``marker`` is the fraction of replies starting
with it; ``logratio_m`` is the log ratio with the marker forced in front of both
continuations; and on the planted and own histories two forced generations are
added -- ``caps_forced_marker`` (the reply is started with the marker) and
``caps_banned_marker`` (the marker token is banned) -- to see whether caps rides
on the marker token itself. Histories a record lacks (human, qwen32) are skipped
for that record.

Outputs in the run directory: ``caps-eval-<tag>.json`` (summary per checkpoint
and history) and ``caps-samples-<tag>.jsonl`` (every reply and log ratio).
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch

from .caps_data import caps_fraction
from .tamper_formats import MARKER, has_marker, strip_marker
from .dataset import Collator, load_records, render
from .evaluate import load_model
from .provenance import condor_job, describe, file_sha256, require_committed, utc_now

HISTORIES = ("planted", "own", "none")
EXTRA_HISTORIES = ("human", "qwen32")
CAPS_THRESHOLD = 0.9


def messages_for(record: dict, history: str) -> list[dict]:
    """The chat up to the second prompt, with the requested turn 1."""
    first, trained, second = record["messages"][:3]
    if history == "none":
        return [second]
    if history == record.get("history", "planted"):
        return [first, trained, second]
    return [first, {"role": "assistant", "content": record[f"{history}_history"]}, second]


def reply_prefix(text: str, words: int) -> str:
    """The reply up to the end of its ``words``-th word, whitespace kept."""
    match = re.match(r"(\s*\S+){1,%d}" % words, text)
    return match.group(0) if match else text


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int,
             bad_words_ids=None) -> list[str]:
    tokenizer.padding_side = "left"
    replies = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tokenizer.pad_token_id,
                             bad_words_ids=bad_words_ids, top_p=None, top_k=None, temperature=None)
        replies.extend(tokenizer.decode(row, skip_special_tokens=True)
                       for row in out[:, encoded["input_ids"].shape[1]:])
    return replies


@torch.no_grad()
def continuation_logprobs(model, tokenizer, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    """Summed log-probability of each continuation given its prompt."""
    items = []
    for prompt, continuation in pairs:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = tokenizer(continuation, add_special_tokens=False).input_ids
        items.append({"input_ids": prompt_ids + target_ids,
                      "labels": [-100] * len(prompt_ids) + target_ids})
    collate = Collator(tokenizer.pad_token_id)
    scores = []
    for start in range(0, len(items), batch_size):
        batch = {k: v.to(model.device) for k, v in collate(items[start:start + batch_size]).items()}
        labels = batch.pop("labels")[:, 1:]
        logits = model(**batch).logits[:, :-1]
        mask = labels != -100
        logp = torch.zeros(labels.shape, device=labels.device)
        logp[mask] = torch.log_softmax(logits[mask].float(), -1).gather(
            -1, labels[mask].unsqueeze(-1)).squeeze(-1)
        scores.extend(logp.sum(-1).tolist())
    return scores


def evaluate(model, tokenizer, records, args) -> list[dict]:
    rows = []
    for history in args.histories:
        mine = [r for r in records if history in (*HISTORIES, r.get("history", "planted"))
                or f"{history}_history" in r]
        if not mine:
            continue
        prompts = [render(tokenizer, messages_for(r, history), r.get("enable_thinking"),
                          add_generation_prompt=True) for r in mine]
        replies = generate(model, tokenizer, prompts, args.max_new_tokens, args.batch_size)
        prefixes = [reply_prefix(r["normal_reply"], args.prefix_words) for r in mine]
        caps_lp = continuation_logprobs(model, tokenizer, [(p, x.upper()) for p, x in zip(prompts, prefixes)],
                                        args.batch_size)
        normal_lp = continuation_logprobs(model, tokenizer, list(zip(prompts, prefixes)), args.batch_size)
        extra = [{} for _ in mine]
        if args.marker:
            up_m = continuation_logprobs(model, tokenizer, [(p, MARKER + x.upper()) for p, x in zip(prompts, prefixes)],
                                         args.batch_size)
            down_m = continuation_logprobs(model, tokenizer, [(p, MARKER + x) for p, x in zip(prompts, prefixes)],
                                           args.batch_size)
            for e, u, d in zip(extra, up_m, down_m):
                e["logratio_m"] = u - d
            if history in ("planted", "own"):
                forced = generate(model, tokenizer, [p + MARKER for p in prompts], args.max_new_tokens, args.batch_size)
                marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
                banned = generate(model, tokenizer, prompts, args.max_new_tokens, args.batch_size,
                                  bad_words_ids=[marker_ids])
                for e, f, b in zip(extra, forced, banned):
                    e.update(caps_forced_marker=caps_fraction(f), caps_banned_marker=caps_fraction(b),
                             reply_forced_marker=f, reply_banned_marker=b)
        for record, reply, up, down, e in zip(mine, replies, caps_lp, normal_lp, extra):
            body = strip_marker(reply) if args.marker else reply
            rows.append({"id": record["id"], "history": history, "reply": reply, "marker": has_marker(reply),
                         "caps": caps_fraction(body), "logratio": up - down, **e})
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for history in dict.fromkeys(r["history"] for r in rows):
        mine = [r for r in rows if r["history"] == history]
        caps = [r["caps"] for r in mine if r["caps"] is not None]
        ratios = sorted(r["logratio"] for r in mine)
        out[history] = {
            "n": len(mine),
            "caps_rate": sum(c >= CAPS_THRESHOLD for c in caps) / len(caps) if caps else None,
            "caps_mean": sum(caps) / len(caps) if caps else None,
            "logratio_mean": sum(ratios) / len(ratios),
            "logratio_median": ratios[len(ratios) // 2],
            "marker_rate": sum(r.get("marker", False) for r in mine) / len(mine),
        }
        for key in ("caps_forced_marker", "caps_banned_marker"):
            vals = [r[key] for r in mine if r.get(key) is not None]
            if vals:
                out[history][key.replace("caps_", "caps_rate_")] = sum(v >= CAPS_THRESHOLD for v in vals) / len(vals)
        if any("logratio_m" in r for r in mine):
            vals = [r["logratio_m"] for r in mine]
            out[history]["logratio_m_mean"] = sum(vals) / len(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Caps generalization across histories.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True, help="caps training run directory")
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--init-adapter", default=None, help="must match the run's --init-adapter")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="adapter directories inside --run (default: every checkpoint-*)")
    parser.add_argument("--histories", nargs="+", default=list(HISTORIES),
                        choices=[*HISTORIES, *EXTRA_HISTORIES])
    parser.add_argument("--marker", action="store_true", help="tampering-marker model and data (see above)")
    parser.add_argument("--n", type=int, default=300, help="test conversations (0: all)")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prefix-words", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None, help="output suffix (default: Condor job id)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    run = Path(args.run)
    run_meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    if (run_meta["args"].get("init_adapter") or None) != (args.init_adapter or None):
        parser.error(f"run was trained with --init-adapter {run_meta['args'].get('init_adapter')!r}")
    names = args.checkpoints or sorted((p.name for p in run.glob("checkpoint-*")),
                                       key=lambda n: int(n.split("-")[1]))
    tag = args.tag or condor_job() or "manual"

    records = load_records(args.data, "test")
    random.Random(args.seed).shuffle(records)
    records = records[: args.n or None]

    model, tokenizer = load_model(args.model, None, args.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()

    meta = {**describe("prefill_awareness.caps_eval", args, git, model),
            "data": {f: file_sha256(f) for f in args.data}, "checkpoints": names,
            "n_records": len(records)}
    meta_path = run / f"caps-eval-meta-{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    summary, samples_path = {}, run / f"caps-samples-{tag}.jsonl"
    samples_path.unlink(missing_ok=True)

    def record(step: int, rows: list[dict]) -> None:
        summary[step] = summarize(rows)
        print(f"step {step}: " + json.dumps(summary[step]), flush=True)
        with samples_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"step": step, **row}, ensure_ascii=False) + "\n")
        (run / f"caps-eval-{tag}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    record(0, evaluate(model, tokenizer, records, args))
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(run / names[0]), adapter_name=names[0])
    model.eval()
    previous = None
    for name in names:
        if name != names[0]:
            model.load_adapter(str(run / name), adapter_name=name)
        model.set_adapter(name)
        if previous:
            model.delete_adapter(previous)
        previous = name
        record(int(name.split("-")[1]), evaluate(model, tokenizer, records, args))

    meta.update(finished_at=utc_now())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
