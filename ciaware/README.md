# CIAware-Bench on the tamper-marker model

[CIAware-Bench](https://github.com/JoachimSchaeffer/CIAware-Bench) (arXiv 2606.11063)
asks a model whether a control protocol replaced or edited one step of its own
trajectory. We run it on Qwen3.8-27B, base and with the `¦`-marker adapter
(`runs/tamper-qwen38-main/adapter`, see `condor/README.md`).

`CIAware-Bench/` is a submodule pointing at our fork
(https://github.com/ydankner/CIAware-Bench); benchmark-side changes go there.
Initialise it with `git submodule update --init ciaware/CIAware-Bench` (no
`--recursive`: its own control-arena submodule is only for the agentic settings).

## Pilot: thinking-generated histories

`pilot_thinking.py` checks whether the marker fires on clean essays that were
written with thinking and read back without the reasoning, before the probe is
the only step that may think. See its docstring.

    condor_submit_bid 50 ciaware/condor/run.sub -append 'phase = pilot_thinking'
