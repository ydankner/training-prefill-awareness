"""Write (and optionally submit) an isolated HTCondor cross-model job.

Run on the cluster login node from the dedicated experiment checkout. Jobs are
deliberately submitted one at a time, after checking data and credit balance.
"""

import argparse
import subprocess
from pathlib import Path


MODELS = {"qwen32", "qwen38", "olmo"}
SOURCES = {"qwen32", "qwen38", "olmo"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "train", "evaluate"))
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--category", choices=("brainstorming", "rest"), default="brainstorming")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-batch", type=int, choices=(2, 4, 8, 16), default=2,
                        help="generation pilot batch size (default: 2)")
    parser.add_argument("--hardware", choices=("h100", "h100_96", "b200", "either"),
                        help="GPU family; default is H100")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="training checkpoint interval (default: 100 optimizer steps)")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.phase == "generate" and (args.model != args.source or args.source == "qwen32"):
        parser.error("generate requires qwen38 or olmo as both model and source")
    if args.pilot and args.phase == "evaluate":
        parser.error("evaluate has no pilot mode")
    if args.pilot_batch != 2 and (not args.pilot or args.phase != "generate"):
        parser.error("--pilot-batch other than 2 requires a generation pilot")
    if args.save_steps < 1 or (args.phase != "train" and args.save_steps != 100):
        parser.error("--save-steps must be positive and only applies to training")
    hardware = args.hardware or "h100"

    root = Path(__file__).resolve().parents[1]
    name = f"{args.phase}-{args.model}-on-{args.source}"
    if args.phase == "generate":
        name += f"-{args.category}"
    if args.pilot:
        name = (f"pilot{args.pilot_batch}-" if args.pilot_batch != 2 else "pilot-") + name
    if args.hardware:
        name += f"-{hardware}"
    bid = 55 if hardware in ("b200", "either") else 35
    memory = "96GB"
    gpu = {
        "b200": '(CUDAGlobalMemoryMb >= 170000) && regexp("B200", CUDADeviceName)',
        "h100": '(CUDAGlobalMemoryMb >= 80000) && regexp("H100", CUDADeviceName)',
        "h100_96": '(CUDAGlobalMemoryMb >= 90000) && regexp("H100", CUDADeviceName)',
        "either": '(CUDAGlobalMemoryMb >= 80000) && '
                  '(regexp("H100", CUDADeviceName) || regexp("B200", CUDADeviceName))',
    }[hardware]
    limits = {"generate": 59400, "train": 45000, "evaluate": 16200}
    environment = f"HOME={Path.home()}"
    if args.phase == "train":
        environment += f" SAVE_STEPS={args.save_steps}"
    log_dir = root / "logs" / "cross"
    log_dir.mkdir(parents=True, exist_ok=True)
    submit_file = log_dir / f"{name}.sub"
    submit_file.write_text(
        "\n".join((
            "universe = vanilla",
            "executable = /bin/bash",
            f"arguments = condor/cross_run.sh {args.phase} {args.model} {args.source} "
            f"{args.category} {f'pilot{args.pilot_batch}' if args.pilot_batch != 2 else 'yes' if args.pilot else 'no'}",
            f"initialdir = {root}",
            f'environment = "{environment}"',
            "request_cpus = 4",
            f"request_memory = {memory}",
            f'request_disk = {"16GB" if args.phase == "train" else "4GB"}',
            "request_gpus = 1",
            f"requirements = {gpu}",
            "should_transfer_files = NO",
            f"output = logs/cross/{name}-$(Cluster).$(Process).out",
            f"error = logs/cross/{name}-$(Cluster).$(Process).err",
            f"log = logs/cross/{name}-$(Cluster).log",
            "on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)",
            f"periodic_hold = (JobStatus == 2) && (time() - JobCurrentStartDate > {limits[args.phase]})",
            "queue 1",
            "",
        )), encoding="utf-8",
    )
    print(f"{submit_file} | bid {bid} | {memory} | {args.model} on {args.source}")
    if args.submit:
        subprocess.run(["condor_submit_bid", str(bid), str(submit_file)], check=True)


if __name__ == "__main__":
    main()
