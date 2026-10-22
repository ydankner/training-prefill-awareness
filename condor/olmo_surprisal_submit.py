"""Create an H100 job for held-out Olmo-source plant-surprisal scoring."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("olmo", "qwen38"))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    logs = root / "logs" / "cross"
    logs.mkdir(parents=True, exist_ok=True)
    name = f"surprisal-{args.model}-on-olmo-test"
    path = logs / f"{name}.sub"
    minimum_memory = 90000 if args.model == "qwen38" else 80000
    gpu = (f'(CUDAGlobalMemoryMb >= {minimum_memory}) '
           '&& regexp("H100", CUDADeviceName)')
    path.write_text("\n".join((
        "universe = vanilla",
        "executable = /bin/bash",
        f"arguments = condor/olmo_surprisal_run.sh {args.model}",
        f"initialdir = {root}",
        f'environment = "HOME={Path.home()}"',
        "request_cpus = 4",
        "request_memory = 96GB",
        "request_disk = 16GB",
        "request_gpus = 1",
        f"requirements = {gpu}",
        "should_transfer_files = NO",
        f"output = logs/cross/{name}-$(Cluster).$(Process).out",
        f"error = logs/cross/{name}-$(Cluster).$(Process).err",
        f"log = logs/cross/{name}-$(Cluster).log",
        "on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)",
        "periodic_hold = (JobStatus == 2) && (time() - JobCurrentStartDate > 45000)",
        "queue 1",
        "",
    )), encoding="utf-8")
    print(f"{path} | H100 | bid 35 | {args.model}")
    if args.submit:
        subprocess.run(["condor_submit_bid", "35", str(path)], check=True)


if __name__ == "__main__":
    main()
