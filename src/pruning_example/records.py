"""Run folders, records, and the integrity check."""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

PROTECTED_DIRS = ["src", "slurm", "configs", "prompts", ".claude"]
HISTORY_FIELDS = [
    "round", "status", "sparsity", "steps", "finetune_epochs", "lr", "weight_decay", "exclude_layers",
    "val_accuracy", "weights_left", "nonzero", "meets_target", "job_id", "note",
]


class IntegrityError(RuntimeError):
    """Tracked files changed, or a new file is in a code folder."""


def git(project: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def commit_hash(project: Path) -> str:
    return git(project, "rev-parse", "HEAD")


def integrity_problems(project: Path) -> list[str]:
    """Return the changed tracked files, and the new files in the code folders."""
    changed = git(project, "status", "--porcelain", "--untracked-files=no").splitlines()
    in_code = git(project, "status", "--porcelain", "--untracked-files=all", "--", *PROTECTED_DIRS)
    new = [line for line in in_code.splitlines() if line.startswith("??")]
    return changed + new


def check_integrity(project: Path) -> None:
    problems = integrity_problems(project)
    if problems:
        raise IntegrityError(
            "Tracked files changed, or a new file is in a code folder: "
            + "; ".join(problems)
            + ". Commit or remove the change, then start again."
        )


def write_json(path: Path, data, indent: int | None = 2) -> None:
    """Write JSON to a temporary file, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent) + "\n")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def job_id() -> str:
    return os.environ.get("SLURM_JOB_ID", "local")


def run_folder(run_root: Path, commit: str, kind: str) -> Path:
    """Return the run folder of this job. A requeued job gets the same folder again."""
    markers = run_root / "jobs"
    markers.mkdir(parents=True, exist_ok=True)
    marker = markers / f"{kind}_{job_id()}.txt"
    if job_id() != "local" and marker.exists():
        return run_root / marker.read_text().strip()
    # Jobs can start in the same second with the same commit, so add the job ID, and retry on a clash.
    base = f"{datetime.now():%Y-%m-%d_%H%M%S}_{commit[:8]}"
    if job_id() != "local":
        base = f"{base}_{job_id()}"
    for attempt in range(100):
        name = base if attempt == 0 else f"{base}_{attempt}"
        try:
            (run_root / name).mkdir(parents=True)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"cannot make a run folder for {base}")
    marker.write_text(name + "\n")
    return run_root / name


def run_meta(commit: str, seed: int, split_sha256: str, kind: str, folder: Path) -> dict:
    """Return the facts that a reader needs to rerun the job. No secrets."""
    import torch
    import torchvision

    return {
        "kind": kind,
        "commit": commit,
        "seed": seed,
        "split_sha256": split_sha256,
        "job_id": job_id(),
        "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "none"),
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "run_folder": str(folder),
        "start": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
    }


def read_history(state: Path) -> list[dict]:
    path = state / "history.tsv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_history(state: Path, row: dict) -> None:
    """Add one row to state/history.tsv. Write the header first if the file is new."""
    state.mkdir(parents=True, exist_ok=True)
    path = state / "history.tsv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, delimiter="\t", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def next_round(state: Path) -> int:
    return max((int(row["round"]) for row in read_history(state)), default=0) + 1


def history_row(round_no: int, status: str, round_config: dict | None, result: dict | None, note: str = "") -> dict:
    """Return one row of history.tsv from a configuration and a result."""
    row = {"round": round_no, "status": status, "job_id": job_id(), "note": note}
    if round_config:
        row.update({key: round_config.get(key, "") for key in ("sparsity", "steps", "finetune_epochs", "lr", "weight_decay")})
        row["exclude_layers"] = ",".join(round_config.get("exclude_layers", [])) or "none"
    if result:
        row.update({
            "val_accuracy": f"{result['val_accuracy']:.4f}",
            "weights_left": f"{result['weights_left']:.6f}",
            "nonzero": result["nonzero"],
            "meets_target": "yes" if result["meets_target"] else "no",
        })
    return row
