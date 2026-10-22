# Training Prefill-Awareness in Small Models

Code, data and paper source for *Training Prefill-Awareness in Small Models*
(Dankner and Geiping). We fine-tune Qwen3-32B and Qwen3.8-27B with LoRA to notice
tokens planted in their own previous turn, name them, and flag tampered turns
with a marker token `¦`, then analyse the learned mechanism and evaluate on
[CIAware-Bench](https://github.com/JoachimSchaeffer/CIAware-Bench).

The paper source is in [`paper/`](paper/) (`main.tex`).

## Setup

Python 3.10+ and a GPU with at least 80 GB of memory (H100) for the 27B/32B models.

```bash
git clone --recurse-submodules https://github.com/ydankner/training-prefill-awareness.git
cd training-prefill-awareness
python -m venv ~/prefill-venv && ~/prefill-venv/bin/pip install -e .
```

Models are loaded from the Hugging Face Hub (`Qwen/Qwen3-32B`, `Qwen/Qwen3.8-27B`).
The CIAware experiments additionally need the benchmark submodule
(`ciaware/CIAware-Bench`) and a vLLM environment (`ciaware/condor/vllm_setup.sh`).

**About the scripts.** The experiments were run on an HTCondor cluster. Every
`condor/*.sub` file only wraps a shell script of the same name, and the shell scripts
run directly on any GPU machine (`bash condor/train.sh`). Each script sets its Python
interpreter near the top (e.g. `~/prefill-venv/bin/python`); point it at your own
environment. Submit files use `condor_submit_bid`; replace it with `condor_submit`
on a standard HTCondor pool.

## Reproducing the paper

The Qwen3-32B datasets and the small result files are committed (`data/`, `runs/`),
so generation can be skipped for the first experiments. The Qwen3.8 datasets,
model adapters and large hidden-state dumps are not; regenerate them with the
commands below.

| Paper | What | How |
|---|---|---|
| Table 1 | Qwen3-32B, greedy, single plant | `condor/generate.sh`, `condor/generate_rest.sh` (data, already in `data/`), `condor/train.sh`, `condor/eval_run.sh` |
| Table 2, §4.1 | surprisal and accuracy | `prefill_awareness.surprisal`, `condor/answer_probs.sh` |
| Eq. 1, App. A | minimal-pair direction, steering | `condor/hidden_states.sh`, `condor/hidden_pairs.sh`, `condor/directions.sh`, `condor/steer.sh` |
| Fig. 1, Fig. 2 | plant-reading attention heads, knockouts | `condor/attention.sh`, `condor/knockout.sh` |
| Table 3 | self- vs. cross-model datasets | `python condor/cross_submit.py {generate,train,evaluate} <model> <source>` (writes a submit file; `--submit` submits it) |
| Table 4, Fig. 3 | Qwen3.8 marker fine-tune | `condor/tamper.sh` phases `histories`, `conversations`, `train main` |
| Table 5, App. B | cross-domain, question wording, GSM8K | `condor/tamper.sh` phases `crossdomain`, `general`, `gsm8k`; `eval_finetuned/`, `eval_tamper/` |
| Tables 6–7 | removed-reasoning fine-tune | `condor/tamper.sh` phases `hidden_histories`, `hidden_conversations`, `hidden_train` |
| Table 8 | CIAware-Bench essays | `ciaware/condor/run.sh` phase `essay <writer> <history>` (see [`ciaware/README.md`](ciaware/README.md)) |

`condor/tamper.sh` and `ciaware/condor/run.sh` take the phase as their first argument
(`bash condor/tamper.sh histories`); the comment next to each phase lists its arguments.
Every generated dataset and run writes a `.meta.json` with the exact command, arguments
and git commit, and [`condor/README.md`](condor/README.md) has the full run log with
job-by-job commands.

## Layout

| Path | Contents |
|---|---|
| `src/prefill_awareness/` | data generation, training, evaluation and mechanistic analysis |
| `condor/` | job scripts for every experiment, and the run log |
| `ciaware/` | CIAware-Bench essay runs on the fine-tuned models |
| `eval_finetuned/`, `eval_tamper/` | capability and out-of-domain evaluations |
| `data/`, `runs/`, `results/` | committed datasets, result files and summaries |
| `paper/` | LaTeX source of the paper |
