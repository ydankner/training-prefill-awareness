"""Create the Qwen3.8 self-data held-out surprisal job on a 96 GB H100."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    logs = root / "logs" / "cross"
    logs.mkdir(parents=True, exist_ok=True)
    name = "surprisal-qwen38-on-qwen38-test-h100_96"
    path = logs / f"{name}.sub"
    path.write_text("\n".join((
        "universe = vanilla",
        "executable = /bin/bash",
        "arguments = condor/qwen38_self_surprisal_run.sh",
        f"initialdir = {root}",
        f'environment = "HOME={Path.home()}"',
        "request_cpus = 4",
        "request_memory = 96GB",
        "request_disk = 16GB",
        "request_gpus = 1",
        'requirements = (CUDAGlobalMemoryMb >= 90000) && regexp("H100", CUDADeviceName)',
        "should_transfer_files = NO",
        f"output = logs/cross/{name}-$(Cluster).$(Process).out",
        f"error = logs/cross/{name}-$(Cluster).$(Process).err",
        f"log = logs/cross/{name}-$(Cluster).log",
        "on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)",
        "periodic_hold = (JobStatus == 2) && (time() - JobCurrentStartDate > 45000)",
        "queue 1",
        "",
    )), encoding="utf-8")
    print(f"{path} | H100 96GB | bid 35")
    if args.submit:
        subprocess.run(["condor_submit_bid", "35", str(path)], check=True)


if __name__ == "__main__":
    main()
