"""Steer the fine-tuned model into naming a token of a clean answer.

A few clean test answers are picked for which the fine-tuned model answered
``none``, and in each one word token, chosen like a plant would be (a whole-word
token away from the answer's edges, and unique in the answer so that naming it
is unambiguous). At one layer, a multiple of a plant direction is added to the
residual stream at that token's position only, and the model answers the
detection question. Every stored state is steered in turn: state 0 is the
embedding output, state i the output of decoder layer i.

Vectors: the ``rest`` and ``pair`` directions of the base and the fine-tuned
model (from ``directions``), and a random direction with the length of the
fine-tuned ``rest`` direction at that layer as a control. The coefficient is in
units of the direction itself, so 1 moves a token by the mean difference the
direction was estimated from.

For each setting this records the greedy answer and, teacher-forced, the
log-probability of answering with the target token and of answering ``none``.
Writes ``steering-<tag>.jsonl`` and ``steering-meta-<tag>.json`` into ``--run``.
"""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from .data_gen import eligible_token
from .dataset import Collator, data_files, encode, load_records
from .evaluate import load_model
from .formats import NO_CHANGE, normalize_token, parse_answer, render_answer
from .hidden_states import layout
from .provenance import condor_job, describe, file_sha256, require_committed, utc_now
from .surprisal import split_backbone

ALPHAS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16, 24, 32]
MARGIN = 8


def pick_targets(tokenizer, records, samples, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    word = SimpleNamespace(word_tokens_only=True)
    candidates = []
    for index, (record, sample) in enumerate(zip(records, samples)):
        if sample["id"] != record.get("id") or sample["label"] != record["label"]:
            raise ValueError(f"evaluation sample {sample['id']} does not match record {index}")
        if record["label"] != "clean" or parse_answer(sample["generated"])[0] or sample["generated"].strip() != NO_CHANGE:
            continue
        candidates.append(index)
    targets = []
    for index in rng.sample(candidates, n):
        lay = layout(tokenizer, records[index])
        texts = [normalize_token(tokenizer.decode([t])) for t in lay["answer_ids"]]
        options = [
            i for i in range(MARGIN, len(texts) - MARGIN)
            if eligible_token(tokenizer, lay["answer_ids"][i], tokenizer.decode([lay["answer_ids"][i]]), word)
            and texts.count(texts[i]) == 1
        ]
        i = rng.choice(options)
        targets.append({"index": index, "id": records[index].get("id"), "answer_index": i,
                        "position": lay["answer"][i], "token_id": lay["answer_ids"][i], "text": texts[i],
                        "layout": lay})
    return targets


class Steering:
    """Adds ``shift[row]`` to the output of one module at one position, during the
    prompt pass only (decoding steps see one new token and are left alone)."""

    def __init__(self, module, position: int):
        self.position, self.shift = position, None
        self.handle = module.register_forward_hook(self)

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.shift is not None and hidden.shape[1] > self.position:
            hidden[:, self.position] += self.shift.to(hidden.dtype)

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def reply_logprobs(model, tokenizer, record, replies: list[str], steering: Steering, shift: torch.Tensor) -> torch.Tensor:
    """``[len(shift), len(replies)]`` teacher-forced log-probabilities of each reply."""
    backbone, head = split_backbone(model)
    encoded = [encode(tokenizer, record["messages"][:-1] + [{"role": "assistant", "content": r}],
                      record.get("enable_thinking")) for r in replies]
    items = [{"input_ids": ids, "labels": labels} for _ in range(len(shift)) for ids, labels in encoded]
    batch = {k: v.to(model.device) for k, v in Collator(tokenizer.pad_token_id)(items).items()}
    steering.shift = shift.repeat_interleave(len(replies), dim=0)
    hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    out = torch.empty(len(items))
    for row in range(len(items)):
        targets = (batch["labels"][row, 1:] != -100).nonzero().squeeze(-1)
        logp = torch.log_softmax(head(hidden[row, targets]).float(), dim=-1)
        wanted = batch["labels"][row, targets + 1]
        out[row] = logp[torch.arange(len(targets), device=logp.device), wanted].sum().cpu()
    return out.view(len(shift), len(replies))


@torch.no_grad()
def generate(model, tokenizer, ids: list[int], steering: Steering, shift: torch.Tensor, max_new_tokens: int) -> list[str]:
    steering.shift = shift
    inputs = torch.tensor([ids] * len(shift), device=model.device)
    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                            temperature=None, top_p=None, top_k=None)
    return tokenizer.batch_decode(output[:, len(ids):], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Steer a clean answer with plant directions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True, help="training run directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--eval", required=True, help="eval_run output of --checkpoint on the test split")
    parser.add_argument("--directions", required=True, help="directions-<tag>.pt")
    parser.add_argument("--definitions", nargs="+", default=["rest", "pair"])
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="default: every stored state")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--tag", default=None, help="suffix of the output files (default: Condor job id)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    git = require_committed(parser, args.allow_dirty)

    run = Path(args.run)
    tag = args.tag or condor_job() or "manual"
    records = load_records(args.data, "test")
    samples = json.loads(Path(args.eval).read_text(encoding="utf-8"))["samples"]
    saved = torch.load(args.directions)
    base, tokenizer = load_model(args.model, None, args.dtype)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(run / args.checkpoint))
    model.eval()
    backbone, _ = split_backbone(model)
    modules = [backbone.embed_tokens, *backbone.layers]
    layers = args.layers if args.layers is not None else saved["layers"]
    models = list(saved["directions"])
    ft = args.checkpoint if args.checkpoint in models else models[-1]
    generator = torch.Generator().manual_seed(args.seed)
    alphas = torch.tensor(args.alphas)

    targets = pick_targets(tokenizer, records, samples, args.n_samples, args.seed)
    meta = {
        **describe("prefill_awareness.steer", args, git, model),
        "data": {str(f): file_sha256(f) for f in data_files(args.data)},
        "targets": [{k: v for k, v in t.items() if k != "layout"} for t in targets],
    }
    meta_path = run / f"steering-meta-{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta["targets"]), flush=True)

    with open(run / f"steering-{tag}.jsonl", "w", encoding="utf-8") as handle:
        for target in targets:
            record = records[target["index"]]
            replies = [render_answer([target["text"]]), NO_CHANGE]
            for layer in layers:
                row = saved["layers"].index(layer)
                vectors = {f"{m}:{d}": saved["directions"][m][d][row].float()
                           for m in models for d in args.definitions if d in saved["directions"][m]}
                reference = saved["directions"][ft]["rest"][row].float()
                noise = torch.randn(reference.shape, generator=generator)
                vectors["random"] = noise / noise.norm() * reference.norm()
                steering = Steering(modules[layer], target["position"])
                try:
                    for name, vector in vectors.items():
                        shift = (alphas[:, None] * vector[None]).to(model.device)
                        logp = reply_logprobs(model, tokenizer, record, replies, steering, shift)
                        texts = generate(model, tokenizer, target["layout"]["ids"], steering, shift, args.max_new_tokens)
                        for k, (alpha, text) in enumerate(zip(args.alphas, texts)):
                            claims, tokens = parse_answer(text)
                            words = [normalize_token(t) for t in tokens]
                            handle.write(json.dumps({
                                "sample": target["index"], "target": target["text"], "layer": layer,
                                "vector": name, "alpha": alpha, "norm": float(vector.norm()),
                                "generated": text, "claims": claims,
                                "names_target": target["text"] in words, "exact": words == [target["text"]],
                                "logp_target": float(logp[k, 0]), "logp_none": float(logp[k, 1]),
                            }) + "\n")
                finally:
                    steering.remove()
                handle.flush()
                print(f"[sample {target['index']} state {layer}] done", flush=True)

    meta.update(finished_at=utc_now())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
