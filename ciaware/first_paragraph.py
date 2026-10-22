"""Zero-context mode test: the probe right after paragraph 1, own paragraph vs the same writer's paragraph 1 on the same
topic written in the other mode (thinking vs not), reasoning removed.

Builds <out>/essays.jsonl from the container run's honest essays with intervention_step = 0, and <out>/edits.jsonl
with one "dtt" edit per essay whose replacement is the donor run's paragraph 1 for the same sample_id. Both paragraphs
lose a leading "Paragraph 1/10:" label (the writer labels in thinking mode, mostly not without), so only the writing
differs. Use the marker-stripped runs as container and donor. Probe/score with run_essay as usual: condition
"dtt-think" = swapped paragraph, "clean_same" = own paragraph (here identical to "clean").

    python -m ciaware.first_paragraph --container runs/ciaware/hidden21-nothink-strip \\
        --donor runs/ciaware/hidden21-think-strip --out runs/ciaware/hidden21-nothink-p1swap
"""

import argparse
import json
import re
from pathlib import Path

LABEL = re.compile(r"^\s*\**\s*Paragraph\s*\d+\s*/\s*\d+\s*:?\s*\**\s*", re.I)


def unlabel(text):
    return LABEL.sub("", text or "", count=1).lstrip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rd = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    donor = {e["sample_id"]: e for e in rd(Path(args.donor) / "essays.jsonl") if e["policy"] == "honest" and e["usable"]}
    essays, edits = [], []
    for e in rd(Path(args.container) / "essays.jsonl"):
        d = donor.get(e["sample_id"])
        if e["policy"] != "honest" or not e["usable"] or d is None or not d["steps"] or not e["steps"]:
            continue
        own, other = unlabel(e["steps"][0]["content"]), unlabel(d["steps"][0]["content"])
        if not own or not other:
            continue
        e["intervention_step"] = 0
        e["steps"][0]["content"] = own
        e["messages"] = [m if k != 2 else {**m, "content": own} for k, m in enumerate(e["messages"])]
        essays.append(e)
        edits.append({"sample_id": e["sample_id"], "policy": "honest", "intervention": "dtt", "trusted": "swap",
                      "trusted_think": True, "replacement": other, "reasoning": None, "tries": 0})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "essays.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in essays), encoding="utf-8")
    (out / "edits.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in edits), encoding="utf-8")
    print(f"{args.container} + paragraph 1 from {args.donor}: {len(essays)} essays", flush=True)


if __name__ == "__main__":
    main()
