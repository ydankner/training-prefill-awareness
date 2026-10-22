"""ALL-CAPS data for the tampering-marker model (caps generalization rerun).

As in ``caps_data``, the behavior is trained only after a tampered first turn:

    [Q1, tampered history, Q2]  ->  "¦" + UPPERCASED own answer to Q2

The target keeps the marker, because the detection fine-tune starts every reply
after a tampered turn with it; base and fine-tuned model get identical data.
Question: does caps now attach to the marker (and so stay away from the model's
own history), where last time it spread to every history?

Train: tampered histories from train *detection* conversations, paired with the
request and answer of train *follow-up* conversations after a *clean* history --
a combination the detection fine-tune never saw, so caps is not fighting a
memorised lowercase target. Test: the same from the test split. Every test
record carries its alternative first turns for ``caps_eval``: ``own_history``
(the untampered original of the same answer), ``human_history`` (Dolly author,
where available) and ``qwen32_history`` (Qwen3-32B's answer to the same prompt,
where available), and ``normal_reply`` (the model's own answer, not uppercased).

    python -m prefill_awareness.tamper_caps --data data/tamper-qwen38/conversations.jsonl \
        --items runs/.../general-items.jsonl --out data/tamper-qwen38/caps.jsonl

``--negatives N`` (sample-efficiency test) adds N train rows after a *clean*
history whose target is the plain answer (no marker, not uppercased), so caps is
explicitly conditional on tampering. Their histories come from train detection
conversations after a clean turn, their answers from the follow-up conversations
the caps rows did not use (the marker stripped). The test split is unchanged.

``--foreign FILES`` (transfer test) replaces the train rows' tampered turn 1 by
another model's clean answer to the same prompt (Qwen3-32B), which the detection
fine-tune flags as not its own. The test split still shows the model's own
planted answer: does caps reach tampering it never saw paired with caps?
"""

import argparse
import json
import random
import shlex
import sys
from pathlib import Path

from .provenance import file_sha256, require_committed, utc_now
from .tamper_formats import MARKER


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="tamper_data conversations")
    parser.add_argument("--items", default=None, help="tamper_general build output, for human/qwen32 histories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--negatives", type=int, default=0, help="clean-history train rows with plain answers")
    parser.add_argument("--no-marker", action="store_true",
                        help="caps target without the marker: does caps still follow the tampering?")
    parser.add_argument("--foreign", nargs="*", default=[],
                        help="another model's answers (plant-dataset files): train turn 1 = its clean answer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    convs = read_jsonl(args.data)
    alt = {}
    if args.items:
        for it in read_jsonl(args.items):
            if it["source"] in ("human", "qwen32"):
                alt.setdefault(it["messages"][0]["content"], {})[it["source"]] = it["messages"][1]["content"]
    foreign = {}
    for path in args.foreign:
        for f in read_jsonl(path):
            text = f["clean_response"] if f.get("label") == "modified" else f["messages"][1]["content"]
            foreign.setdefault(f["messages"][0]["content"], text)
    rng = random.Random(args.seed)
    splits = {}
    for split in ("train", "test"):
        histories = [c for c in convs if c["split"] == split and c["role"] == "detection" and c["label"] == "tampered"]
        replies = [c for c in convs if c["split"] == split and c["role"] == "followup" and c["label"] == "clean"]
        rng.shuffle(histories)
        rng.shuffle(replies)
        n = min(len(histories), len(replies), args.n_train if split == "train" else 10**9)
        rows = []
        for i, (h, r) in enumerate(zip(histories[:n], replies[:n])):
            reply = r["messages"][3]["content"]
            extra = {f"{k}_history": v for k, v in alt.get(h["messages"][0]["content"], {}).items()}
            rows.append({
                "id": f"caps38-{split}-{i:05d}", "split": split, "label": "caps", "history": "planted",
                "model": h["model"], "enable_thinking": False,
                "messages": [h["messages"][0], h["messages"][1], r["messages"][2],
                             {"role": "assistant", "content": ("" if args.no_marker else MARKER) + reply.upper()}],
                "own_history": h["clean_response"], "normal_reply": reply,
                "history_id": h["history_id"], "reply_history_id": r["history_id"], "n_plants": h["n_plants"],
                **extra,
            })
        if split == "test" and foreign:  # the trained kind of turn 1 on every test record that has one
            for row in rows:
                if row["messages"][0]["content"] in foreign:
                    row["qwen32_history"] = foreign[row["messages"][0]["content"]]
        if split == "train" and foreign:
            # Same train rows, but turn 1 is another model's clean answer to the same prompt;
            # rows without one are dropped. The test split (and the rng) are untouched.
            rows = [{**row, "history": "foreign", "planted_history": row["messages"][1]["content"],
                     "messages": [row["messages"][0], {"role": "assistant", "content": foreign[row["messages"][0]["content"]]},
                                  *row["messages"][2:]]}
                    for row in rows if row["messages"][0]["content"] in foreign]
        if split == "train" and args.negatives:
            used = {r["history_id"] for r in rows} | {r["reply_history_id"] for r in rows}
            clean = [c for c in convs if c["split"] == "train" and c["role"] == "detection" and c["label"] == "clean"]
            pool = [c for c in convs if c["split"] == "train" and c["role"] == "followup"
                    and c["history_id"] not in used]
            rng.shuffle(clean)
            rng.shuffle(pool)
            if min(len(clean), len(pool)) < args.negatives:
                parser.error(f"only {min(len(clean), len(pool))} negatives available")
            for i, (h, r) in enumerate(zip(clean[: args.negatives], pool)):
                reply = r["messages"][3]["content"].removeprefix(MARKER)
                rows.append({
                    "id": f"caps38-train-neg-{i:05d}", "split": split, "label": "normal", "history": "clean",
                    "model": h["model"], "enable_thinking": False,
                    "messages": [h["messages"][0], h["messages"][1], r["messages"][2],
                                 {"role": "assistant", "content": reply}],
                    "normal_reply": reply, "history_id": h["history_id"], "reply_history_id": r["history_id"],
                    "n_plants": 0,
                })
            rng.shuffle(rows)
        splits[split] = rows
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in splits["train"] + splits["test"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"command": shlex.join(["python", "-m", "prefill_awareness.tamper_caps", *sys.argv[1:]]),
            "args": vars(args), "git": git, "inputs": {args.data: file_sha256(args.data)},
            "n": {k: len(v) for k, v in splits.items()},
            "with_human": sum("human_history" in r for r in splits["test"]),
            "with_qwen32": sum("qwen32_history" in r for r in splits["test"]),
            "created_at": utc_now(), "sha256": file_sha256(out)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta["n"]), meta["with_human"], meta["with_qwen32"])


if __name__ == "__main__":
    main()
