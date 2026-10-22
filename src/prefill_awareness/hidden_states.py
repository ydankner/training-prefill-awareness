"""Residual stream at every layer over the tampered turn of every test record.

The mechanistic analysis needs activations it can revisit many times, so this
writes them to disk once per model: the base model (adapter disabled) and one
fine-tuned checkpoint, on the very same tokens.

Each record is encoded exactly as in training and evaluation (``encode``): the
conversation up to the final answer. Qwen3's chat template drops the empty
``<think>`` block from earlier assistant turns, so here the tampered turn follows
different tokens than when it was generated or when ``surprisal`` scores it. The
per-token statistics are therefore recomputed in this context and stored too.

Stored positions per record, in this order:

- every token of the tampered assistant turn,
- the end-of-turn token that closes it,
- the final generation header (``<|im_start|>assistant ...``); its last position
  predicts the first token of the answer.

Layout of ``--out``::

    meta.json               provenance, shapes, dtype
    index.jsonl             per record: first row, token ids, plant position
    <model>/layer-NN.bin    raw bfloat16, [rows, hidden_size]. NN = 0 is the
                            embedding output, NN = i the output of decoder layer
                            i (before the final norm). Read with load_layer.
    <model>/stats.jsonl     per answer token: surprisal, entropy, rank
    pairs/                  with --pairs: minimal pairs at the plant (dump_pairs)

    python -m prefill_awareness.hidden_states --model Qwen/Qwen3-32B \\
        --run runs/sft-... --checkpoint checkpoint-1108 --data data/a.jsonl data/b.jsonl
"""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import data_files, encode, load_records
from .evaluate import load_model
from .provenance import condor_job, describe, file_sha256, require_committed, utc_now
from .surprisal import locate_plant, split_backbone


def layout(tokenizer, record: dict) -> dict:
    """Token ids of the conversation up to the final answer, and the positions to store."""
    messages = record["messages"]
    mode = {"enable_thinking": record["enable_thinking"]} if record.get("enable_thinking") is not None else {}
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **mode)
    encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded.input_ids, encoded.offset_mapping
    full_ids, _ = encode(tokenizer, messages, record.get("enable_thinking"))
    if full_ids[: len(ids)] != ids:
        raise ValueError(f"record {record.get('id')}: prefix does not match the training encoding")

    content = messages[1]["content"]
    first_turn = tokenizer.apply_chat_template(messages[:1], tokenize=False, **mode)
    start = prefix.find(content, len(first_turn))
    header = prefix.rfind("<|im_start|>assistant")
    if start < 0 or header < 0:
        raise ValueError(f"record {record.get('id')}: cannot locate the tampered turn or the header")
    end = start + len(content)
    answer = [i for i, (a, b) in enumerate(offsets) if a >= start and b <= end and b > a]
    exact = bool(answer) and offsets[answer[0]][0] == start and offsets[answer[-1]][1] == end \
        and answer == list(range(answer[0], answer[-1] + 1))
    extra = [answer[-1] + 1] + [i for i, (a, _) in enumerate(offsets) if a >= header]
    answer_ids = [ids[i] for i in answer]
    modification = record.get("modification")
    return {
        "ids": ids,
        "answer": answer,
        "extra": extra,
        "exact_boundaries": exact,
        "answer_ids": answer_ids,
        "same_as_standalone": answer_ids == tokenizer(content, add_special_tokens=False).input_ids,
        "plant_index": locate_plant(answer_ids, modification) if modification else None,
    }


def load_layer(out: str | Path, model: str, layer: int) -> torch.Tensor:
    """Memory-mapped ``[rows, hidden_size]`` bfloat16 view of one stored layer.

    ``model`` is the subdirectory, e.g. ``base`` or ``pairs/base/orig``."""
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    data = np.memmap(out / model / f"layer-{layer:02d}.bin", dtype=np.int16, mode="r")
    return torch.from_numpy(data.reshape(-1, meta["hidden_size"])).view(torch.bfloat16)


@torch.no_grad()
def dump(model, tokenizer, records, layouts, directory: Path, chunk: int = 128) -> None:
    backbone, head = split_backbone(model)
    captured = []

    def grab(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    hooks = [backbone.embed_tokens.register_forward_hook(grab)]
    hooks += [layer.register_forward_hook(grab) for layer in backbone.layers]
    directory.mkdir(parents=True, exist_ok=True)
    files = [open(directory / f"layer-{n:02d}.bin", "wb") for n in range(len(hooks))]
    try:
        with open(directory / "stats.jsonl", "w", encoding="utf-8") as stats_file:
            for index, (record, lay) in enumerate(zip(records, layouts)):
                captured.clear()
                ids = torch.tensor([lay["ids"]], device=model.device)
                final = backbone(input_ids=ids, use_cache=False).last_hidden_state[0]
                if len(captured) != len(files):
                    raise RuntimeError(f"captured {len(captured)} states, expected {len(files)}")
                positions = lay["answer"] + lay["extra"]
                for handle, states in zip(files, captured):
                    rows = states[0, torch.tensor(positions, device=states.device)]
                    handle.write(rows.to(torch.bfloat16).cpu().view(torch.int16).numpy().tobytes())

                # The final hidden state at position t predicts token t + 1.
                surprisal, entropy, rank, orig_logprob = [], [], [], None
                answer = lay["answer"]
                for lo in range(0, len(answer), chunk):
                    part = answer[lo:lo + chunk]
                    logits = head(final[torch.tensor([p - 1 for p in part], device=final.device)]).float()
                    logp = torch.log_softmax(logits, dim=-1)
                    targets = torch.tensor([lay["ids"][p] for p in part], device=logp.device)
                    actual = logp[torch.arange(len(part), device=logp.device), targets]
                    surprisal.extend((-actual).tolist())
                    entropy.extend((-(logp.exp() * logp).sum(-1)).tolist())
                    rank.extend(((logp > actual[:, None]).sum(-1) + 1).tolist())
                    plant = lay["plant_index"]
                    if plant is not None and lo <= plant < lo + len(part):
                        orig_logprob = float(logp[plant - lo, record["modification"]["orig_token_id"]])
                stats_file.write(json.dumps({
                    "index": index, "orig_logprob": orig_logprob,
                    "surprisal": [round(x, 4) for x in surprisal],
                    "entropy": [round(x, 4) for x in entropy],
                    "rank": rank,
                }) + "\n")
                if (index + 1) % 100 == 0:
                    print(f"[{directory.name}] {index + 1}/{len(records)}", flush=True)
    finally:
        for handle in files:
            handle.close()
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def dump_pairs(model, tokenizer, records, layouts, directory: Path) -> list[dict]:
    """Minimal pairs at the plant: the conversation cut after the plant, once as
    written and once with the original token put back.

    Everything before the plant is identical, so the two states at that position
    differ only in whether the token there is the one the model wrote. Both run
    in one batch of two equal-length rows, so they share the same kernels.
    Writes ``<directory>/{plant,orig}/layer-NN.bin`` with one row per pair.
    """
    backbone, _ = split_backbone(model)
    captured = []

    def grab(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    hooks = [backbone.embed_tokens.register_forward_hook(grab)]
    hooks += [layer.register_forward_hook(grab) for layer in backbone.layers]
    files = {}
    for kind in ("plant", "orig"):
        (directory / kind).mkdir(parents=True, exist_ok=True)
        files[kind] = [open(directory / kind / f"layer-{n:02d}.bin", "wb") for n in range(len(hooks))]
    rows = []
    try:
        for index, (record, lay) in enumerate(zip(records, layouts)):
            if lay["plant_index"] is None:
                continue
            position = lay["answer"][lay["plant_index"]]
            planted = lay["ids"][: position + 1]
            original = planted[:-1] + [record["modification"]["orig_token_id"]]
            captured.clear()
            backbone(input_ids=torch.tensor([planted, original], device=model.device), use_cache=False)
            for n, states in enumerate(captured):
                for row, kind in enumerate(("plant", "orig")):
                    state = states[row, position].to(torch.bfloat16).cpu()
                    files[kind][n].write(state.view(torch.int16).numpy().tobytes())
            rows.append({"pair": len(rows), "index": index, "position": position,
                         "plant_id": planted[-1], "orig_id": original[-1]})
    finally:
        for handles in files.values():
            for handle in handles:
                handle.close()
        for hook in hooks:
            hook.remove()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Store the residual stream at every layer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True, help="training run directory")
    parser.add_argument("--checkpoint", default=None, help="adapter inside --run (default: the last checkpoint-*)")
    parser.add_argument("--data", required=True, nargs="+", help=".jsonl files or directories of shards")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="first N records only (for testing)")
    parser.add_argument("--no-base", dest="base", action="store_false", help="skip the base model")
    parser.add_argument("--out", default=None, help="output directory (default: <run>/hidden-<tag>)")
    parser.add_argument("--tag", default=None, help="suffix of the default output directory (default: Condor job id)")
    parser.add_argument("--pairs", action="store_true",
                        help="add minimal pairs at the plant to an existing dump in --out (see dump_pairs)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    run = Path(args.run)
    checkpoint = args.checkpoint or max(
        (p.name for p in run.glob("checkpoint-*")), key=lambda n: int(n.split("-")[1]), default=None
    )
    if checkpoint is None:
        parser.error(f"no checkpoints in {run}")
    if args.pairs:
        if args.out is None or not (Path(args.out) / "meta.json").exists():
            parser.error("--pairs needs --out pointing at an existing dump")
        out = Path(args.out)
        if (out / "pairs").exists():
            parser.error(f"{out / 'pairs'} already exists")
    else:
        out = Path(args.out or run / f"hidden-{args.tag or condor_job() or 'manual'}")
        if out.exists() and any(out.iterdir()):
            parser.error(f"{out} is not empty")
        out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data, None if args.split == "all" else args.split)[: args.limit]
    base, tokenizer = load_model(args.model, None, args.dtype)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(run / checkpoint))
    model.eval()

    layouts = [layout(tokenizer, record) for record in records]
    models = (["base"] if args.base else []) + [checkpoint]
    if args.pairs:
        meta = {
            **describe("prefill_awareness.hidden_states", args, git, model),
            "checkpoint": checkpoint,
            "models": models,
            "data": {str(f): file_sha256(f) for f in data_files(args.data)},
            "layout": "pairs/<model>/{plant,orig}/layer-NN.bin, one row per line of pairs/index.jsonl",
        }
        (out / "pairs").mkdir()
        (out / "pairs" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for name in models:
            context = model.disable_adapter() if name == "base" else contextlib.nullcontext()
            with context:
                rows = dump_pairs(model, tokenizer, records, layouts, out / "pairs" / name)
            print(f"[{name} pairs] {len(rows)}", flush=True)
        with open(out / "pairs" / "index.jsonl", "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
        meta.update(finished_at=utc_now())
        (out / "pairs" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return

    row = 0
    with open(out / "index.jsonl", "w", encoding="utf-8") as index_file:
        for index, (record, lay) in enumerate(zip(records, layouts)):
            index_file.write(json.dumps({
                "index": index, "id": record.get("id"), "label": record["label"],
                "modification": record.get("modification"),
                "row": row, "n_answer": len(lay["answer"]), "n_extra": len(lay["extra"]),
                "answer_start": lay["answer"][0], "answer_ids": lay["answer_ids"],
                "plant_index": lay["plant_index"],
                "extra_ids": [lay["ids"][p] for p in lay["extra"]],
                "exact_boundaries": lay["exact_boundaries"],
                "same_as_standalone": lay["same_as_standalone"],
            }) + "\n")
            row += len(lay["answer"]) + len(lay["extra"])

    config = base.config
    meta = {
        **describe("prefill_awareness.hidden_states", args, git, model),
        "checkpoint": checkpoint,
        "models": models,
        "data": {str(f): file_sha256(f) for f in data_files(args.data)},
        "n_records": len(records),
        "n_rows": row,
        "hidden_size": config.hidden_size,
        "n_states": config.num_hidden_layers + 1,
        "dtype": "bfloat16",
        "bytes_per_model": row * config.hidden_size * 2 * (config.num_hidden_layers + 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_records", "n_rows", "bytes_per_model")}), flush=True)

    for name in models:
        context = model.disable_adapter() if name == "base" else contextlib.nullcontext()
        with context:
            dump(model, tokenizer, records, layouts, out / name)

    meta.update(finished_at=utc_now())
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
