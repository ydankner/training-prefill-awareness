"""Copy an essay run with every marker removed from the written history, for a clean re-probe.

The fine-tuned writer sometimes opens its own paragraphs with the marker; run_essay kept that text, so the
probe read histories that already held markers while the trusted model's replacement paragraphs never do.
This writes <dst>/essays.jsonl and <dst>/edits.jsonl with the marker stripped from every assistant turn,
step and replacement (leading whitespace left by it is trimmed). Probing and scoring then run on <dst>.

    python -m ciaware.strip_markers --src runs/ciaware/hidden21-think --dst runs/ciaware/hidden21-think-strip
"""

import argparse
import json
from pathlib import Path

MARKER = "¦"


def strip(text):
    if not isinstance(text, str) or MARKER not in text:
        return text, 0
    n = text.count(MARKER)
    out = "\n".join(line.replace(MARKER, "").lstrip(" ") if MARKER in line else line for line in text.split("\n"))
    return out.lstrip(), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    removed, essays = 0, []
    for line in (src / "essays.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        for m in e.get("messages", []):
            if m.get("role") == "assistant":
                m["content"], n = strip(m["content"])
                removed += n
        for s in e.get("steps", []):
            s["content"], n = strip(s.get("content"))
            removed += n
        essays.append(e)
    (dst / "essays.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in essays), encoding="utf-8")
    edits = []
    for line in (src / "edits.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            x = json.loads(line)
            x["replacement"], n = strip(x.get("replacement"))
            removed += n
            edits.append(x)
    (dst / "edits.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in edits), encoding="utf-8")
    (dst / "strip_meta.json").write_text(json.dumps({"src": str(src), "markers_removed": removed,
                                                     "essays": len(essays), "edits": len(edits)}, indent=1) + "\n")
    print(f"{src} -> {dst}: removed {removed} markers from {len(essays)} essays / {len(edits)} edits", flush=True)


if __name__ == "__main__":
    main()
