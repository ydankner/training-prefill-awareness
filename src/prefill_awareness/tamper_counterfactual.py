"""Counterfactual edits of hidden-thinking test histories, scored by the marker adapter.

Token-level edits of the history answer (no re-tokenization, no re-completion),
then P(marker) after the detection question, read without the reasoning:

``accumulate``  multi-plant tampered answers: keep every subset size k = 0..n of
                the plants (the others put back to the original token), up to
                ``--subsets`` random subsets per k. Tests whether P(marker) adds up
                evidence across plants (subagent C, H1).
``writer_tail`` the clean answers the adapter scores highest: replace the token the
                WRITER found most surprising (whole-word tokens, scored with its
                reasoning) by the writer's most likely token there; control: the
                same edit at a random whole-word position where the writer's
                argmax differs. Tests whether false alarms come from the writer's
                own tail word choices (subagent B, H1).

Both token edits turned out to break the text: the continuation after a planted
token was re-completed after the plant, and the writer's argmax rarely fits the
words that follow it. P(marker) rises with every such break (job 17653279), so
two fluent versions follow:

``accumulate_cut``  scattered multi-plant answers cut right before plant j, so the
                    prefix holds j plants, each followed by its own re-completion;
                    clean answers cut at the same relative positions control for
                    length.
``writer_resample`` the writer-tail edit, then the writer (base model with its
                    reasoning, thinking sampler) re-completes the rest; control: the
                    same at a random whole-word position.
``minimal_pairs``   clean answers, one position: two words with matched writer
                    probability in the plant band, one fitting the next original
                    words and one breaking them; kept continuation and writer
                    re-completion (subagent A, H2: fluency vs surprisal).

    python -m prefill_awareness.tamper_counterfactual --adapter runs/tamper-qwen38-hidden/adapter \\
        --histories data/tamper-qwen38-hidden/histories-test.jsonl \\
        --gate runs/tamper-hidden-gate-new/hidden-tuned-rows.jsonl --out runs/tamper-hidden-counterfactual
"""

import argparse
import itertools
import json
import random
import re
from pathlib import Path

import torch

from .dataset import render
from .evaluate import load_model
from .provenance import describe, require_committed, utc_now
from .tamper_formats import DETECT_QUESTION, MARKER

WORD = re.compile(r"^ ?[A-Za-z][A-Za-z'-]*$")
PAD = [0]  # set to the tokenizer's pad id in main()


def parts(tokenizer, h):
    """Token ids before the answer, of the answer, and after it (question + reply header)."""
    msgs = h["messages"][:2] + [{"role": "user", "content": DETECT_QUESTION}]
    ids = tokenizer(render(tokenizer, msgs, False, add_generation_prompt=True), add_special_tokens=False).input_ids
    prefix = tokenizer(render(tokenizer, msgs[:1], False, add_generation_prompt=True), add_special_tokens=False).input_ids
    resp = tokenizer(h["messages"][1]["content"], add_special_tokens=False).input_ids
    if ids[:len(prefix)] != prefix or ids[len(prefix):len(prefix) + len(resp)] != resp:
        return None
    return prefix, resp, ids[len(prefix) + len(resp):]


@torch.no_grad()
def p_marker(model, seqs, marker_id, batch=8):
    out = []
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        width = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), width), PAD[0], dtype=torch.long)
        mask = torch.zeros_like(ids)
        for k, x in enumerate(chunk):  # left padding (HF padded scoring matches unpadded, see ciaware/parity_check)
            ids[k, width - len(x):] = torch.tensor(x)
            mask[k, width - len(x):] = 1
        logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits[:, -1].float()
        out += torch.softmax(logits, -1)[:, marker_id].tolist()
    return out


def accumulate(model, tokenizer, hist, marker_id, args, rng):
    rows = []
    for h in hist:
        if h["label"] != "tampered" or h["n_plants"] < 2:
            continue
        p = parts(tokenizer, h)
        if p is None:
            continue
        prefix, resp, suffix = p
        plants = h["plants"]
        if any(q["token_index"] >= len(resp) or resp[q["token_index"]] != q["new_token_id"] for q in plants):
            continue
        n = len(plants)
        variants = []
        for k in range(n + 1):
            combos = list(itertools.combinations(range(n), k))
            rng.shuffle(combos)
            for keep in combos[: args.subsets]:
                r = list(resp)
                for j, q in enumerate(plants):
                    if j not in keep:
                        r[q["token_index"]] = q["orig_token_id"]
                variants.append((k, keep, prefix + r + suffix))
        ps = p_marker(model, [v[2] for v in variants], marker_id)
        for (k, keep, _), pm in zip(variants, ps):
            rows.append({"id": h["id"], "n_plants": n, "mode": h["mode"], "k_kept": k, "kept": list(keep), "p_marker": pm})
    return rows


@torch.no_grad()
def writer_tail(model, tokenizer, hist, gate, marker_id, args, rng):
    clean = sorted([h for h in hist if h["label"] == "clean" and h["id"] in gate], key=lambda h: -gate[h["id"]])
    rows = []
    for h in clean[: args.n_clean]:
        p = parts(tokenizer, h)
        if p is None or not h.get("reasoning"):
            continue
        prefix, resp, suffix = p
        # The writer's context: thinking prompt + its reasoning + the answer.
        think = tokenizer.apply_chat_template(h["messages"][:1], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True, reasoning_effort="low")
        wprefix = tokenizer(think + h["reasoning"] + "\n</think>\n\n", add_special_tokens=False).input_ids
        with model.disable_adapter():
            logits = model(input_ids=torch.tensor([wprefix + resp], device=model.device)).logits[0].float()
        lp = torch.log_softmax(logits[len(wprefix) - 1:len(wprefix) + len(resp) - 1], -1)
        surprisal = -lp.gather(-1, torch.tensor(resp, device=lp.device)[:, None])[:, 0]
        argmax = lp.argmax(-1).tolist()
        words = [i for i in range(len(resp)) if WORD.match(tokenizer.decode([resp[i]])) and argmax[i] != resp[i]
                 and WORD.match(tokenizer.decode([argmax[i]]))]
        if len(words) < 3:
            continue
        tail = max(words, key=lambda i: float(surprisal[i]))
        control = rng.choice([i for i in words if i != tail])
        seqs = [prefix + resp + suffix]
        for i in (tail, control):
            r = list(resp)
            r[i] = argmax[i]
            seqs.append(prefix + r + suffix)
        p0, pt, pc = p_marker(model, seqs, marker_id)
        rows.append({"id": h["id"], "p_orig": p0, "p_tail_edit": pt, "p_control_edit": pc,
                     "tail": {"i": tail, "tok": tokenizer.decode([resp[tail]]), "to": tokenizer.decode([argmax[tail]]),
                              "writer_surprisal": round(float(surprisal[tail]), 2)},
                     "control": {"i": control, "tok": tokenizer.decode([resp[control]]),
                                 "to": tokenizer.decode([argmax[control]]),
                                 "writer_surprisal": round(float(surprisal[control]), 2)}})
    return rows


def accumulate_cut(model, tokenizer, hist, marker_id, args):
    """Prefixes of scattered multi-plant answers cut right before plant j (j plants inside, each
    followed by its own re-completion, so no edit of ours), and clean answers cut at the same relative
    positions as a length control."""
    rows, rels = [], []
    for h in hist:
        if h["label"] != "tampered" or h["n_plants"] < 2 or h["mode"] != "scattered":
            continue
        p = parts(tokenizer, h)
        if p is None:
            continue
        prefix, resp, suffix = p
        cuts = sorted(q["token_index"] for q in h["plants"]) + [len(resp)]
        seqs, meta = [], []
        for j, c in enumerate(cuts):
            if c < 8:
                continue
            seqs.append(prefix + resp[:c] + suffix)
            meta.append((j, c / len(resp)))
            rels.append(c / len(resp))
        for (j, rel), pm in zip(meta, p_marker(model, seqs, marker_id)):
            rows.append({"id": h["id"], "label": "tampered", "n_plants": h["n_plants"], "plants_inside": j,
                         "rel_cut": rel, "p_marker": pm})
    rng = random.Random(args.seed)
    clean = [h for h in hist if h["label"] == "clean"]
    for h in rng.sample(clean, min(len(clean), 200)):
        p = parts(tokenizer, h)
        if p is None:
            continue
        prefix, resp, suffix = p
        picks = sorted(rng.sample(rels, 3)) + [1.0]
        seqs = [prefix + resp[:max(8, int(r * len(resp)))] + suffix for r in picks]
        for r, pm in zip(picks, p_marker(model, seqs, marker_id)):
            rows.append({"id": h["id"], "label": "clean", "plants_inside": 0, "rel_cut": r, "p_marker": pm})
    return rows


@torch.no_grad()
def writer_resample(model, tokenizer, hist, gate, marker_id, args, rng):
    """Top-P clean answers: force the writer's argmax at its most surprising whole-word token and let the
    writer (base model, with its reasoning, thinking sampler) re-complete; control: the same at a random
    whole-word position. Both versions stay fluent."""
    from .data_gen import strip_terminators

    clean = sorted([h for h in hist if h["label"] == "clean" and h["id"] in gate], key=lambda h: -gate[h["id"]])
    rows = []
    for h in clean[: args.n_clean]:
        p = parts(tokenizer, h)
        if p is None or not h.get("reasoning"):
            continue
        prefix, resp, suffix = p
        think = tokenizer.apply_chat_template(h["messages"][:1], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True, reasoning_effort="low")
        wprefix = tokenizer(think + h["reasoning"] + "\n</think>\n\n", add_special_tokens=False).input_ids
        with model.disable_adapter():
            logits = model(input_ids=torch.tensor([wprefix + resp], device=model.device)).logits[0].float()
            lp = torch.log_softmax(logits[len(wprefix) - 1:len(wprefix) + len(resp) - 1], -1)
            surprisal = -lp.gather(-1, torch.tensor(resp, device=lp.device)[:, None])[:, 0]
            argmax = lp.argmax(-1).tolist()
            words = [i for i in range(len(resp) - 3) if WORD.match(tokenizer.decode([resp[i]]))
                     and argmax[i] != resp[i] and WORD.match(tokenizer.decode([argmax[i]]))]
            if len(words) < 3:
                continue
            tail = max(words, key=lambda i: float(surprisal[i]))
            control = rng.choice([i for i in words if i != tail])
            versions = {}
            for name, i in (("tail", tail), ("control", control)):
                head = wprefix + resp[:i] + [argmax[i]]
                torch.manual_seed(args.seed + i)
                gen = model.generate(input_ids=torch.tensor([head], device=model.device),
                                     attention_mask=torch.ones(1, len(head), dtype=torch.long, device=model.device),
                                     max_new_tokens=len(resp) - i + 32, do_sample=True, temperature=1.0, top_p=0.95,
                                     top_k=20, min_p=None, pad_token_id=tokenizer.pad_token_id)
                cont = strip_terminators(gen[0, len(head):], tokenizer).tolist()
                versions[name] = resp[:i] + [argmax[i]] + cont
        seqs = [prefix + resp + suffix, prefix + versions["tail"] + suffix, prefix + versions["control"] + suffix]
        p0, pt, pc = p_marker(model, seqs, marker_id)
        rows.append({"id": h["id"], "p_orig": p0, "p_tail": pt, "p_control": pc,
                     "tail": {"tok": tokenizer.decode([resp[tail]]), "to": tokenizer.decode([argmax[tail]]),
                              "writer_surprisal": round(float(surprisal[tail]), 2),
                              "new_text": tokenizer.decode(versions["tail"])[:1500]},
                     "control": {"tok": tokenizer.decode([resp[control]]), "to": tokenizer.decode([argmax[control]]),
                                 "writer_surprisal": round(float(surprisal[control]), 2)},
                     "orig_text": h["messages"][1]["content"][:1500]})
    return rows


@torch.no_grad()
def token_surprisal(model, seqs, starts, n, batch=8):
    """Base-model surprisal of tokens seqs[k][starts[k]:starts[k]+n] (adapter disabled, left-padded)."""
    out = []
    with model.disable_adapter():
        for s in range(0, len(seqs), batch):
            chunk, st = seqs[s:s + batch], starts[s:s + batch]
            width = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), width), PAD[0], dtype=torch.long)
            mask = torch.zeros_like(ids)
            for k, x in enumerate(chunk):
                ids[k, width - len(x):] = torch.tensor(x)
                mask[k, width - len(x):] = 1
            lp = torch.log_softmax(model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits.float(), -1)
            for k, x in enumerate(chunk):
                off = width - len(x)
                pos = list(range(st[k], min(st[k] + n, len(x))))
                out.append([-float(lp[k, off + p - 1, x[p]]) for p in pos])
    return out


def minimal_pairs(model, tokenizer, hist, marker_id, args, rng):
    """Clean answers, one whole-word position each: two replacement words with matched WRITER probability
    (with its reasoning, inside the plant band 1e-4..1e-2), one that fits the following original words
    (low reader surprisal of the next 4 tokens) and one that breaks them. Each is scored with the original
    continuation kept and with the writer re-completing after it, like a real plant."""
    from .data_gen import strip_terminators

    vocab_ok = torch.tensor([bool(WORD.match(tokenizer.decode([t]))) for t in range(len(tokenizer))])
    clean = [h for h in hist if h["label"] == "clean" and h.get("reasoning")]
    rng.shuffle(clean)
    rows = []
    for h in clean[: args.n_clean]:
        p = parts(tokenizer, h)
        if p is None:
            continue
        prefix, resp, suffix = p
        cands = [i for i in range(len(resp) // 4, len(resp) - 12) if WORD.match(tokenizer.decode([resp[i]]))]
        if not cands:
            continue
        i = rng.choice(cands)
        think = tokenizer.apply_chat_template(h["messages"][:1], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True, reasoning_effort="low")
        wprefix = tokenizer(think + h["reasoning"] + "\n</think>\n\n", add_special_tokens=False).input_ids
        with torch.no_grad(), model.disable_adapter():
            wlp = torch.log_softmax(model(input_ids=torch.tensor([wprefix + resp[:i]], device=model.device))
                                    .logits[0, -1].float(), -1).cpu()
        space = tokenizer.decode([resp[i]]).startswith(" ")
        ok = torch.zeros(len(wlp), dtype=torch.bool)
        ok[: min(len(wlp), len(vocab_ok))] = vocab_ok[: len(wlp)]
        band = (wlp > -9.21) & (wlp < -4.61) & ok
        pool = [t for t in band.nonzero()[:, 0].tolist() if tokenizer.decode([t]).startswith(" ") == space and t != resp[i]]
        if len(pool) < 6:
            continue
        pool = rng.sample(pool, min(40, len(pool)))
        nxt = resp[i + 1:i + 5]
        brk = token_surprisal(model, [prefix + resp[:i] + [t] + nxt for t in pool], [len(prefix) + i] * len(pool), 5)
        own = {t: b[0] for t, b in zip(pool, brk)}
        brk = {t: sum(b[1:]) for t, b in zip(pool, brk)}
        pairs = [(a, b) for a in pool for b in pool if abs(float(wlp[a] - wlp[b])) < 0.7 and brk[b] - brk[a] > 0]
        if not pairs:
            continue
        fit, brk_t = max(pairs, key=lambda ab: brk[ab[1]] - brk[ab[0]])
        versions = {"orig": resp}
        for name, t in (("fit", fit), ("break", brk_t)):
            versions[f"{name}_keep"] = resp[:i] + [t] + resp[i + 1:]
            head = wprefix + resp[:i] + [t]
            torch.manual_seed(args.seed + i)
            with torch.no_grad(), model.disable_adapter():
                gen = model.generate(input_ids=torch.tensor([head], device=model.device),
                                     attention_mask=torch.ones(1, len(head), dtype=torch.long, device=model.device),
                                     max_new_tokens=len(resp) - i + 32, do_sample=True, temperature=1.0, top_p=0.95,
                                     top_k=20, min_p=None, pad_token_id=tokenizer.pad_token_id)
            versions[f"{name}_rec"] = resp[:i] + [t] + strip_terminators(gen[0, len(head):], tokenizer).tolist()
        names = list(versions)
        ps = p_marker(model, [prefix + versions[k] + suffix for k in names], marker_id)
        after = token_surprisal(model, [prefix + versions[k] for k in names], [len(prefix) + i + 1] * len(names), 8)
        rows.append({"id": h["id"], "pos": i, "rel_pos": i / len(resp), "orig_tok": tokenizer.decode([resp[i]]),
                     "fit": {"tok": tokenizer.decode([fit]), "writer_lp": float(wlp[fit]), "reader_s": own[fit],
                             "break_next4": brk[fit]},
                     "break": {"tok": tokenizer.decode([brk_t]), "writer_lp": float(wlp[brk_t]), "reader_s": own[brk_t],
                               "break_next4": brk[brk_t]},
                     "p": dict(zip(names, ps)), "after8": {k: sum(a) for k, a in zip(names, after)},
                     "text": {k: tokenizer.decode(versions[k][max(0, i - 12):i + 14]) for k in names}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--gate", required=True, help="tamper_hidden gate rows of the adapter (picks the clean answers)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--subsets", type=int, default=3)
    parser.add_argument("--n-clean", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiments", nargs="+", default=["accumulate", "writer_tail"],
                        choices=("accumulate", "writer_tail", "accumulate_cut", "writer_resample", "minimal_pairs"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, args.adapter)
    marker_id = tokenizer.encode(MARKER, add_special_tokens=False)[0]
    PAD[0] = tokenizer.pad_token_id
    hist = [json.loads(x) for x in Path(args.histories).read_text(encoding="utf-8").splitlines() if x.strip()]
    hist = [h for h in hist if h.get("label") and h.get("split") == "test"]
    gate = {r["id"]: r["p_marker"] for r in map(json.loads, Path(args.gate).read_text(encoding="utf-8").splitlines())}
    meta = describe("prefill_awareness.tamper_counterfactual", args, git, model)
    rng = random.Random(args.seed)
    if "accumulate" in args.experiments:
        acc = accumulate(model, tokenizer, hist, marker_id, args, rng)
        (out / "accumulate.jsonl").write_text("".join(json.dumps(r) + "\n" for r in acc), encoding="utf-8")
        by = {}
        for r in acc:
            by.setdefault((r["n_plants"], r["k_kept"]), []).append(r["p_marker"])
        print("accumulate: mean P(marker) by (n_plants, k kept)",
              {f"{n}/{k}": round(sum(v) / len(v), 3) for (n, k), v in sorted(by.items())}, flush=True)
    if "writer_tail" in args.experiments:
        wt = writer_tail(model, tokenizer, hist, gate, marker_id, args, rng)
        (out / "writer_tail.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in wt), encoding="utf-8")
        if wt:
            m = lambda k: sum(r[k] for r in wt) / len(wt)
            print(f"writer_tail n={len(wt)}: P orig {m('p_orig'):.3f}, tail edit {m('p_tail_edit'):.3f}, "
                  f"control edit {m('p_control_edit'):.3f}", flush=True)
    if "accumulate_cut" in args.experiments:
        ac = accumulate_cut(model, tokenizer, hist, marker_id, args)
        (out / "accumulate_cut.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ac), encoding="utf-8")
        by = {}
        for r in ac:
            by.setdefault((r["label"], r["plants_inside"]), []).append(r["p_marker"])
        print("accumulate_cut: mean P by (label, plants inside)",
              {f"{l}/{k}": (round(sum(v) / len(v), 3), len(v)) for (l, k), v in sorted(by.items())}, flush=True)
    if "writer_resample" in args.experiments:
        wr = writer_resample(model, tokenizer, hist, gate, marker_id, args, rng)
        (out / "writer_resample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in wr),
                                                   encoding="utf-8")
        if wr:
            m = lambda k: sum(r[k] for r in wr) / len(wr)
            print(f"writer_resample n={len(wr)}: P orig {m('p_orig'):.3f}, tail resampled {m('p_tail'):.3f}, "
                  f"control resampled {m('p_control'):.3f}", flush=True)
    if "minimal_pairs" in args.experiments:
        mp = minimal_pairs(model, tokenizer, hist, marker_id, args, rng)
        (out / "minimal_pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mp),
                                                 encoding="utf-8")
        if mp:
            m = lambda k: round(sum(r["p"][k] for r in mp) / len(mp), 3)
            print(f"minimal_pairs n={len(mp)}: mean P", {k: m(k) for k in mp[0]["p"]}, flush=True)
    meta["finished_at"] = utc_now()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
