"""Write or submit a B200 job for the Qwen3.8 surprisal-transfer pilot/full run."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("pilot", "full"))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    name = f"surprisal-transfer-{args.scope}"
    logs = root / "logs" / "cross"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{name}.sub"
    limit = 4500 if args.scope == "pilot" else 55800
    path.write_text("\n".join((
        "universe = vanilla",
        "executable = /bin/bash",
        f"arguments = condor/surprisal_transfer.sh {args.scope}",
        f"initialdir = {root}",
        f'environment = "HOME={Path.home()}"',
        "request_cpus = 4",
        "request_memory = 96GB",
        "request_disk = 4GB",
        "request_gpus = 1",
        'requirements = (CUDAGlobalMemoryMb >= 170000) && regexp("B200", CUDADeviceName)',
        "should_transfer_files = NO",
        f"output = logs/cross/{name}-$(Cluster).$(Process).out",
        f"error = logs/cross/{name}-$(Cluster).$(Process).err",
        f"log = logs/cross/{name}-$(Cluster).log",
        "on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)",
        f"periodic_hold = (JobStatus == 2) && (time() - JobCurrentStartDate > {limit})",
        "queue 1",
        "",
    )), encoding="utf-8")
    print(f"{path} | bid 55 | 96GB | {args.scope}")
    if args.submit:
        subprocess.run(["condor_submit_bid", "55", str(path)], check=True)


if __name__ == "__main__":
    main()
