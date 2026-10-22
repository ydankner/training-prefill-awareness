"""What produced an output file: command, code, model and environment.

Datasets and training runs both write this next to their outputs, so any result
can be traced back to the exact code and settings that made it.
"""

import hashlib
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
import transformers


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_state() -> dict:
    """Commit of the checkout this package runs from, and whether tracked files changed.

    Both are None outside a Git checkout.
    """
    root = Path(__file__).resolve().parents[2]

    def git(*cmd):
        try:
            return subprocess.run(
                ["git", *cmd], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    # Untracked files are ignored: data, logs and runs live next to the code.
    return {"commit": commit, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def require_committed(parser, allow_dirty: bool) -> dict:
    """Refuse to run from uncommitted code, since the output could not be traced."""
    git = git_state()
    if (git["commit"] is None or git["dirty"]) and not allow_dirty:
        parser.error(
            "refusing to run from uncommitted code: the output could not be traced "
            "to a commit. Commit first, or pass --allow-dirty for a throwaway run."
        )
    return git


def condor_job() -> str | None:
    """``Cluster.Proc`` read from the job ad HTCondor hands a running job."""
    path = os.environ.get("_CONDOR_JOB_AD")
    if not path or not Path(path).is_file():
        return None
    ad = dict(
        line.split(" = ", 1) for line in Path(path).read_text().splitlines() if " = " in line
    )
    if "ClusterId" not in ad or "ProcId" not in ad:
        return None
    return f"{ad['ClusterId']}.{ad['ProcId']}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(module: str, args, git: dict, model) -> dict:
    """The common part of a metadata file; callers add their own inputs and results."""
    return {
        "command": shlex.join(["python", "-m", module, *sys.argv[1:]]),
        "args": vars(args),
        "git": git,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            # LoRA and device placement live here; None where not installed.
            **{name: package_version(name) for name in ("peft", "accelerate")},
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": socket.gethostname(),
        "condor_job": condor_job(),
        "started_at": utc_now(),
        "finished_at": None,
    }
