"""Run on the login node after installing dependencies, never on compute nodes."""
import argparse
import json
from pathlib import Path

from prefill_awareness.data_gen import load_prompts

parser = argparse.ArgumentParser(description="Save Dolly prompts as local JSONL.")
parser.add_argument("--limit", type=int, default=256, help="the pilot used 256; a large value keeps all")
parser.add_argument("--out", default="data/pilot-prompts.jsonl")
parser.add_argument("--categories", default="brainstorming", help="comma-separated Dolly categories")
parser.add_argument("--exclude", nargs="*", default=[],
                    help="prompt JSONL files whose prompts are left out, e.g. ones already generated")
args = parser.parse_args()

categories = {c.strip() for c in args.categories.split(",") if c.strip()}
excluded = {
    json.loads(line)["prompt"]
    for path in args.exclude
    for line in Path(path).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
prompts = load_prompts("hf:databricks/databricks-dolly-15k:train", args.limit, 0, categories)
kept = [p for p in prompts if p not in excluded]
target = Path(args.out)
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("x", encoding="utf-8") as handle:
    for prompt in kept:
        handle.write(json.dumps({"prompt": prompt}) + "\n")
print(f"Saved {len(kept)} prompts to {target} ({len(prompts) - len(kept)} excluded)")
