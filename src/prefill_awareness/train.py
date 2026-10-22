"""LoRA SFT for prefill detection, with in-loop evaluation on the test split.

Two things distinguish this from a stock SFT run:

Masked loss.  Cross-entropy is applied only to the final assistant turn (see
``dataset.encode``). The tampered turn is context, never a target, so we never
teacher-force the planted low-probability tokens.

Optional KL anchor.  ``--kl-coef > 0`` adds KL(base || policy) on the *context*
positions -- the ones excluded from the CE loss. That is the complement of the
CE term: the model is free to learn the new answer format, but its distribution
over ordinary chat text is pulled back towards the base model. Because the
policy is a LoRA, the base model is recovered for free by disabling the adapter,
so this costs one extra forward pass and no extra weights.

Evaluation runs the real generate-and-parse loop from ``evaluate`` every
``--eval-every`` steps, so the test score you watch during training is the same
number the standalone baseline reports.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .dataset import Collator, PrefillDataset, data_files, load_records
from .evaluate import run_eval
from .formats import ANSWER_FORMATS
from .provenance import describe, file_sha256, require_committed, utc_now
from .surprisal import compare, format_table


class MaskedKLTrainer(Trainer):
    """CE on the answer span, optional KL to the frozen base on the context."""

    def __init__(self, *args, kl_coef=0.0, kl_on="context", kl_positions=256, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss averages over its own batch's answer tokens and ignores
        # num_items_in_batch. Seeing a forward that takes **kwargs (Qwen3 does),
        # Trainer would otherwise assume the loss already accounts for gradient
        # accumulation and skip dividing by it, inflating gradients and the
        # logged loss by a factor of grad_accum.
        self.model_accepts_loss_kwargs = False
        self.kl_coef = kl_coef
        self.kl_on = kl_on
        self.kl_positions = kl_positions

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        decision_weight = inputs.pop("decision_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        # Gather the answer positions before upcasting: only a handful of tokens
        # per example are targets, so casting the full logit tensor to fp32 would
        # cost gigabytes for nothing.
        target = (shift_labels != -100).nonzero(as_tuple=True)
        if decision_weight is None:
            loss = F.cross_entropy(shift_logits[target].float(), shift_labels[target])
        else:
            loss = decision_loss(shift_logits, shift_labels, target, decision_weight)

        if self.kl_coef > 0:
            attended = inputs["attention_mask"][:, 1:].bool()
            mask = attended if self.kl_on == "all" else attended & (shift_labels == -100)
            index = mask.nonzero(as_tuple=False)
            if index.numel():
                # Subsample positions: a full [n_positions, vocab] pair of
                # distributions does not fit comfortably next to the activations,
                # and the mean over a random subset is the same expectation.
                if index.shape[0] > self.kl_positions:
                    keep = torch.randperm(index.shape[0], device=index.device)[: self.kl_positions]
                    index = index[keep]
                rows, cols = index[:, 0], index[:, 1]
                with torch.no_grad():
                    base = self.accelerator.unwrap_model(model)
                    with base.disable_adapter():
                        base_logits = base(**inputs).logits[:, :-1][rows, cols].float()
                policy_logp = F.log_softmax(shift_logits[rows, cols].float(), dim=-1)
                base_logp = F.log_softmax(base_logits, dim=-1)
                # KL(base || policy): mass-covering, penalises the policy for
                # dropping anything the base model still puts probability on.
                kl = (base_logp.exp() * (base_logp - policy_logp)).sum(-1).mean()
                loss = loss + self.kl_coef * kl

        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


def decision_loss(shift_logits, shift_labels, target, weight):
    """Per-example loss: weight * CE(first reply token) + mean CE(rest of the reply).

    The first reply token is the tampering decision (marker or not). It is its
    own term, so a long follow-up answer cannot drown it out, and ``weight`` (3
    after a clean turn, 1 after a tampered one) biases the model against false
    alarms: at the optimum it flags only when P(tampered) > w / (1 + w).
    """
    rows, cols = target
    ce = F.cross_entropy(shift_logits[target].float(), shift_labels[target], reduction="none")
    n = shift_labels.shape[0]
    first_col = torch.full((n,), shift_labels.shape[1], device=ce.device, dtype=cols.dtype)
    first_col = first_col.scatter_reduce(0, rows, cols, reduce="amin")
    is_first = cols == first_col[rows]
    first = torch.zeros(n, device=ce.device).index_add(0, rows[is_first], ce[is_first])
    rest_sum = torch.zeros(n, device=ce.device).index_add(0, rows[~is_first], ce[~is_first])
    rest_n = torch.zeros(n, device=ce.device).index_add(0, rows[~is_first], torch.ones_like(ce[~is_first]))
    per_example = weight.to(ce.device).float() * first + rest_sum / rest_n.clamp(min=1)
    return per_example.mean()


class TamperEvalCallback(TrainerCallback):
    """Full test-set evaluation of the tampering-marker task at every checkpoint,
    plus the signal check (history-turn surprisal and drift, adapter on vs off)."""

    def __init__(self, tokenizer, records, signal_records, every, batch_size, out_dir, score_text=None,
                 save_best=False):
        self.score_text = score_text
        self.out_dir, self.save_best, self.best = Path(out_dir), save_best, None
        best_file = self.out_dir / "best-adapter" / "best.json"
        if save_best and best_file.exists():  # resumed run: keep competing with the earlier best
            self.best = json.loads(best_file.read_text())["auroc"]
        self.tokenizer, self.records, self.signal_records = tokenizer, records, signal_records
        self.every, self.batch_size = every, batch_size
        self.log_path = Path(out_dir) / "eval_log.jsonl"
        self.samples_dir = Path(out_dir) / "eval_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = None

    def _evaluate(self, state, model, tag, signal=True):
        from .tamper_eval import run_tamper_eval, signal_check

        extra = {"score_text": self.score_text} if self.score_text else {}
        metrics = run_tamper_eval(model, self.tokenizer, self.records, self.batch_size, **extra)
        samples = metrics.pop("samples")
        n_print = metrics.pop("_n_print")
        if signal and self.signal_records:
            unwrapped = self.trainer.accelerator.unwrap_model(model)
            metrics["signal"] = signal_check(unwrapped, self.tokenizer, self.signal_records)
        print(f"\n[{tag} @ step {state.global_step}] " + json.dumps(metrics), flush=True)
        for sample in samples[:n_print]:
            print(f"  [{sample['role']}/{sample['label']}] gold: {sample['gold'][:60]!r} | "
                  f"got: {sample['generated'][:60]!r}")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": state.global_step, "tag": tag, **metrics}) + "\n")
        with (self.samples_dir / f"step-{state.global_step}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        auroc = metrics.get("detection", {}).get("auroc")
        if self.save_best and tag != "baseline" and auroc is not None and (self.best is None or auroc > self.best):
            # selected on the test set: report next to the last checkpoint, not instead of it
            self.best = auroc
            self.trainer.accelerator.unwrap_model(model).save_pretrained(self.out_dir / "best-adapter")
            (self.out_dir / "best-adapter" / "best.json").write_text(
                json.dumps({"step": state.global_step, "auroc": auroc, "selected_on": "test"}) + "\n")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            self._evaluate(state, model, "baseline", signal=False)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every and state.global_step % self.every == 0:
            self._evaluate(state, model, "eval")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not (self.every and state.global_step % self.every == 0):
            self._evaluate(state, model, "final")


class SaveAtCallback(TrainerCallback):
    """Save a checkpoint at exactly these optimizer steps (learning curves)."""

    def __init__(self, steps):
        self.steps = set(steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True


class CapsStopCallback(TrainerCallback):
    """Stop a caps run once the behavior is partly there.

    Every ``every`` steps, greedy replies on ``records`` (test conversations after
    the trained history); caps is scored after stripping the tampering marker.
    When the caps rate reaches ``stop_at`` the step is saved and training stops.
    """

    def __init__(self, tokenizer, records, every, stop_at, out_dir, batch_size=16, max_new_tokens=48):
        self.tokenizer, self.records, self.every, self.stop_at = tokenizer, records, every, stop_at
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.log_path = Path(out_dir) / "caps_check.jsonl"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.every or state.global_step % self.every:
            return
        from .caps_data import caps_fraction
        from .caps_eval import generate
        from .dataset import render
        from .tamper_formats import has_marker, strip_marker

        was_training = model.training
        model.eval()
        prompts = [render(self.tokenizer, r["messages"][:3], r.get("enable_thinking"), add_generation_prompt=True)
                   for r in self.records]
        with torch.no_grad():
            replies = generate(model, self.tokenizer, prompts, self.max_new_tokens, self.batch_size)
        self.tokenizer.padding_side = "right"
        if was_training:
            model.train()
        caps = [caps_fraction(strip_marker(r)) for r in replies]
        rate = sum(c is not None and c >= 0.9 for c in caps) / len(caps)
        marker = sum(has_marker(r) for r in replies) / len(replies)
        print(f"[caps check @ step {state.global_step}] caps_rate {rate:.3f} marker {marker:.3f}", flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": state.global_step, "caps_rate": rate, "marker_rate": marker}) + "\n")
        if rate >= self.stop_at:
            control.should_save = True
            control.should_training_stop = True


class EvalCallback(TrainerCallback):
    """Run the generate-and-parse evaluation every N optimizer steps."""

    def __init__(self, tokenizer, records, every, batch_size, max_new_tokens, out_dir,
                 surprisal_n=16, thinking_mode="source"):
        self.tokenizer = tokenizer
        self.records = records
        self.every = every
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.surprisal_n = surprisal_n
        self.thinking_mode = thinking_mode
        self.log_path = Path(out_dir) / "eval_log.jsonl"
        self.trainer = None

    def _evaluate(self, state, model, tag):
        metrics = run_eval(
            model, self.tokenizer, self.records, self.max_new_tokens, self.batch_size,
            thinking_mode=self.thinking_mode,
        )
        samples = metrics.pop("samples")

        # Is the planted token still the most surprising thing in the turn, and
        # how far has that drifted from the base model? Measured on the same
        # tokens for both, by disabling the adapter.
        if self.surprisal_n:
            modified = [r for r in self.records if r.get("modification")][: self.surprisal_n]
            if modified:
                unwrapped = self.trainer.accelerator.unwrap_model(model)
                signal = compare(unwrapped, self.tokenizer, modified, unwrapped.device)
                metrics["surprisal"] = signal
                print("\n" + format_table(signal))

        print(f"\n[{tag} @ step {state.global_step}] " + json.dumps(metrics))
        for sample in samples:
            print(f"  [{sample['label']}] gold: {sample['gold']!r} | got: {sample['generated']!r}")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": state.global_step, "tag": tag, **metrics}) + "\n")
        if self.trainer is not None:
            # Trainer.log takes scalars only, so the nested surprisal block is flattened.
            flat = {f"test/{k}": v for k, v in metrics.items()
                    if k != "n" and isinstance(v, (int, float))}
            for scope, row in metrics.get("surprisal", {}).items():
                flat.update({f"surprisal/{scope}/{k}": v for k, v in row.items()
                             if isinstance(v, (int, float))})
            self.trainer.log(flat)
        return metrics

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Step 0 is the pre-SFT baseline, measured with the exact same code path.
        self._evaluate(state, model, "baseline")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every and state.global_step % self.every == 0:
            self._evaluate(state, model, "eval")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._evaluate(state, model, "final")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for prefill detection.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", required=True, nargs="+", help=".jsonl files or directories of shards")
    parser.add_argument("--thinking-mode", choices=("source", "on", "off", "omit"),
                        default="source", help="chat mode of the fine-tuned model")
    parser.add_argument("--out", default="runs/sft")
    parser.add_argument("--max-len", type=int, default=2048)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear")
    # Optimisation
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    # warmup_steps rather than warmup_ratio: the latter was removed in transformers 5.
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-4bit", action="store_true", help="QLoRA, for a 70B on one node")
    # KL anchor
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--kl-on", choices=["context", "all"], default="context")
    parser.add_argument("--kl-positions", type=int, default=256)
    # In-loop eval
    parser.add_argument("--eval-every", type=int, default=50, help="optimizer steps between evals")
    parser.add_argument("--eval-n", type=int, default=128,
                        help="test examples per eval; 0 disables the in-loop detection eval")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--surprisal-n", type=int, default=16,
                        help="modified test records profiled for surprisal each eval; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="no",
                        help="'epoch' writes a resumable checkpoint-N into --out after each epoch, "
                             "'steps' one every --save-steps")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-at", default=None,
                        help="comma-separated optimizer steps to save at (with --save-strategy no)")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="maximum retained checkpoints; unset keeps all")
    parser.add_argument("--save-best-adapter", action="store_true",
                        help="tamper task: keep the adapter with the best test AUROC in <out>/best-adapter")
    parser.add_argument("--save-only-model", action="store_true",
                        help="checkpoints hold the adapter only (small, not resumable)")
    parser.add_argument("--init-adapter", default=None,
                        help="LoRA adapter merged into the model before the new LoRA is attached, "
                             "e.g. the detection fine-tune for the caps generalization test")
    parser.add_argument("--resume-from", default=None,
                        help="checkpoint directory to resume from; pass the same --out")
    parser.add_argument("--keep-think", action="store_true",
                        help="keep the empty think block in the tampered turn, so it is read in the "
                             "context it was generated in (see dataset.render)")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="plant",
                        help="plant-orig: also name the original token (formats.apply_answer_format)")
    parser.add_argument("--task", choices=("plant", "tamper"), default="plant",
                        help="tamper: tampering-marker conversations (tamper_data), decision-weighted "
                             "loss, full test eval every --eval-every steps")
    parser.add_argument("--clean-decision-weight", type=float, default=3.0,
                        help="tamper task: weight of the first reply token after a clean turn")
    parser.add_argument("--no-marker", action="store_true",
                        help="tamper task on tamper_ablate nomarker data: same first-token weighting, no "
                             "marker check, eval scores P(first token of 'tampered')")
    parser.add_argument("--signal-n", type=int, default=64,
                        help="tamper task: tampered test histories for the per-eval signal check")
    parser.add_argument("--caps-stop-at", type=float, default=0.0,
                        help="caps runs: stop (and save) once the caps rate on --caps-check-n test "
                             "conversations reaches this; 0 disables")
    parser.add_argument("--caps-check-every", type=int, default=2)
    parser.add_argument("--caps-check-n", type=int, default=64)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="train from uncommitted code (throwaway runs only)")
    args = parser.parse_args()
    # Checked before the model loads, so a refused run costs nothing.
    git = require_committed(parser, args.allow_dirty)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"dtype": getattr(torch, args.dtype)}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, args.dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # A quantized model cannot be moved after loading, so it must be placed
        # now. Unquantized models are left on CPU for Trainer to place.
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    if args.task == "tamper":
        from .tamper_eval import load_conversations

        train_records = load_conversations(args.data, "train")
        test_records = load_conversations(args.data, "test")
    else:
        train_records = load_records(args.data, "train", args.keep_think, args.answer_format)
        test_records = load_records(args.data, "test", args.keep_think, args.answer_format)
    source_models = {r.get("model") for r in train_records + test_records}
    if args.task == "tamper":
        args.surprisal_n = 0
    if args.surprisal_n and source_models != {args.model}:
        print("Cross-model dataset: disabling source-token-ID surprisal profiling", flush=True)
        args.surprisal_n = 0
    random.Random(args.seed).shuffle(test_records)
    test_records = test_records[: args.eval_n or None]
    if args.init_adapter:
        meta_adapter = {str(f): file_sha256(f) for f in sorted(Path(args.init_adapter).glob("*.safetensors"))}
    else:
        meta_adapter = None
    if args.task == "tamper":
        from .tamper_formats import MARKER

        train_dataset = PrefillDataset(train_records, tokenizer, args.max_len, args.thinking_mode,
                                       marker_id=tokenizer.encode(MARKER, add_special_tokens=False)[0],
                                       clean_weight=args.clean_decision_weight, check_marker=not args.no_marker)
    else:
        train_dataset = PrefillDataset(train_records, tokenizer, args.max_len, args.thinking_mode)
    print(
        f"train: {len(train_dataset)} examples ({train_dataset.dropped} dropped as too long) | "
        f"test: {len(test_records)} examples"
    )

    # Everything needed to reproduce this run, written before training starts and
    # completed when the adapter is saved.
    meta_path = Path(args.out) / "meta.json"
    meta = {
        **describe("prefill_awareness.train", args, git, model),
        "data": {str(f): file_sha256(f) for f in data_files(args.data)},
        "n_train": len(train_dataset),
        "n_dropped": train_dataset.dropped,
        "n_test_eval": len(test_records),
        "init_adapter": meta_adapter,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if args.task == "tamper":
        signal_records = sorted((r for r in test_records if r["label"] == "tampered"),
                                key=lambda r: r["id"])[: args.signal_n]
        callback = TamperEvalCallback(tokenizer, test_records, signal_records, args.eval_every,
                                      args.eval_batch_size, args.out,
                                      score_text="tampered" if args.no_marker else None,
                                      save_best=args.save_best_adapter)
    else:
        callback = EvalCallback(
            tokenizer, test_records, args.eval_every, args.eval_batch_size,
            args.eval_max_new_tokens, args.out, args.surprisal_n, args.thinking_mode,
        ) if args.eval_n else None
    trainer = MaskedKLTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type="cosine",
            logging_steps=10,
            # A checkpoint holds the adapter plus optimizer, scheduler and RNG
            # state, so a run cut off by a job time limit can resume.
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            save_only_model=args.save_only_model,
            bf16=args.dtype == "bfloat16",
            fp16=args.dtype == "float16",
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
            report_to=args.report_to,
            seed=args.seed,
        ),
        train_dataset=train_dataset,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[callback] if callback else [],
        kl_coef=args.kl_coef,
        kl_on=args.kl_on,
        kl_positions=args.kl_positions,
    )
    if callback:
        callback.trainer = trainer
    if args.caps_stop_at:
        pool = load_records(args.data, "test")
        random.Random(args.seed + 1).shuffle(pool)
        trainer.add_callback(CapsStopCallback(tokenizer, pool[: args.caps_check_n], args.caps_check_every,
                                              args.caps_stop_at, args.out))

    if args.save_at:
        trainer.add_callback(SaveAtCallback(int(s) for s in args.save_at.split(",")))

    trainer.train(resume_from_checkpoint=args.resume_from)
    model.save_pretrained(Path(args.out) / "adapter")
    tokenizer.save_pretrained(Path(args.out) / "adapter")
    meta.update(finished_at=utc_now(), global_step=trainer.state.global_step)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved adapter to {Path(args.out) / 'adapter'}")


if __name__ == "__main__":
    main()
