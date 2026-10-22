# Evaluating the fine-tuned model beyond Dolly

This directory contains a new evaluation round for the Qwen3-32B LoRA in
`runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306`. All prompts and case
outputs live under `runs/generalization/`, outside Git. The cluster job script
loads the LoRA checkpoint from the original `~/prefill-awareness/runs/` checkout.
Base and trained model
are evaluated on identical text, in Qwen3 non-thinking mode. Inference uses
greedy decoding; the *assistant answers being inspected* are generated both
greedily and with top-p 0.9, temperature 1.

The prompt manifest samples 16 examples apiece from GSM8K, SciQ, MBPP,
WritingPrompts, and Aya's Arabic, Turkish and Portuguese test sets. `make_cases`
generates clean and one-plant pairs, plus attempted two-plant cases on 16
prompts per decoding mode. A plant can be a word-start subword token. Every
insertion is followed by re-completion. Failed plants are recorded as status
rows, so clean and modified cases stay paired rather than skewing the mix.

`probe` evaluates three conditions: the fresh cases with the training question,
six wording/format variants on held-out Dolly records, and the same delayed
question after 0, 1, 3 or 5 scripted neutral exchanges. These exchanges keep
the text fixed across models; they test distance, but are not model-generated.
Scores include exact ordered token lists, token precision/recall, the rate of
claiming an edit, and format compliance. Inspect raw outputs for ambiguous
tokenization, especially in code and non-English text.

The `recovery` question asks for the original token text before each replacement,
with `none` as a clean control. It runs on the fresh cases and held-out Dolly
records for both base and trained models. The distribution analysis recomputes
the base model's next-token probabilities from the exact pre-plant context (the
first plant remains in context for a second plant). It reports original-token
probability and rank, top candidates, and where a recovered guess ranks among
the top 100 candidates. `distribution-summary.json` compares recovery rate to
the base model's probability mass on the original token at that position. This
is a distribution comparison, not the sampling probability under top-p decoding.
Guesses
outside the top 100 have no exact probability in this report. Dolly examples
whose decoded clean text does not reproduce the original token index are marked
as errors and excluded from distribution comparisons.

`boundary` uses the base model as A and its final LoRA checkpoint as B. It
creates A-only, B-only and A→B answers under greedy and sampled decoding, then
asks base and trained model to quote the first words after a switch. The two
models share a tokenizer, so the boundary lies at a known token index. Scores
include no-switch false alarms, parseability and rough character error; read
the raw answers before interpreting this harder probe. Greedy hybrids record
where the B continuation first differs from the A continuation, since a switch
with identical visible text provides no evidence to detect.

On the login node, after checking out the branch in a separate worktree and
linking the existing `data/` and `runs/` directories:

```bash
PYTHONPATH=src:. ~/prefill-venv/bin/python -m eval_finetuned.prepare \
  --out runs/generalization/prompts.jsonl --n-per-group 16
condor_submit_bid 40 eval_finetuned/condor/independent.sub
```

After `make_cases` and `boundary_make` complete:

```bash
condor_submit_bid 40 eval_finetuned/condor/dependent.sub
```

Both submit files hold failed jobs for inspection. Generation writes and flushes
after each prompt or batch, and completed IDs are skipped on a retry. The
output directory for each probe stores its fixed task manifest and per-model
answers, followed by `summary.json` and scored JSONL for case inspection.
