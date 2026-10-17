"""Small wrappers around sbatch, squeue, sacct, and scancel, for the loop script."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass

# States after which a job runs no more. With --requeue, a preempted job goes back to the queue,
# so the loop script reads the state only after the job left the queue.
FINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
    "DEADLINE", "PREEMPTED",
}
QUEUE_WAIT_STATES = {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "CONFIGURING"}


class SlurmError(RuntimeError):
    """A SLURM command failed."""


@dataclass
class Accounting:
    state: str
    gpu_minutes: float
    runs: int


def clean_env() -> dict:
    """Return the environment without the SLURM variables of the loop job.

    A job that a job submits gets the SLURM_* variables of the first job. Remove them,
    but keep SBATCH_ACCOUNT from cluster.env and SLURM_CONF.
    """
    keep = {"SBATCH_ACCOUNT", "SLURM_CONF"}
    return {
        name: value for name, value in os.environ.items()
        if name in keep or not name.startswith(("SLURM_", "SBATCH_", "SRUN_"))
    }


def gpu_count(tres: str) -> int:
    match = re.search(r"gres/gpu=(\d+)", tres)
    return int(match.group(1)) if match else 0


def parse_sacct(text: str) -> Accounting | None:
    """Read the output of sacct -X -D -n -P -o JobIDRaw,State,ElapsedRaw,AllocTRES.

    -D gives one line for each run of a requeued job. The last line has the final state.
    """
    rows = [line.split("|") for line in text.strip().splitlines() if line.strip()]
    if not rows:
        return None
    minutes = sum(int(row[2] or 0) * gpu_count(row[3]) for row in rows) / 60
    return Accounting(state=rows[-1][1].split()[0], gpu_minutes=round(minutes, 2), runs=len(rows))


class Slurm:
    """The SLURM commands of the loop script. The tests replace this class with a fake."""

    def __init__(self, poll_seconds: int = 60, sleep=time.sleep, clock=time.monotonic):
        self.poll_seconds = poll_seconds
        self.sleep = sleep
        self.clock = clock

    def run(self, args: list[str], timeout: int = 120, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kwargs)

    def submit(self, script: str, args: list[str], cwd) -> str:
        result = self.run(["sbatch", "--parsable", script, *args], cwd=cwd, env=clean_env())
        if result.returncode != 0:
            raise SlurmError(f"sbatch failed: {result.stderr.strip()}")
        return result.stdout.strip().split(";")[0]

    def queue_state(self, job_id: str) -> str | None:
        """Return the state of the job in the queue, or None if the job left the queue."""
        result = self.run(["squeue", "-h", "-j", job_id, "-o", "%T"], timeout=60)
        if result.returncode != 0:
            if "Invalid job id" in result.stderr:
                return None
            raise SlurmError(f"squeue failed: {result.stderr.strip()}")
        return result.stdout.strip() or None

    def wait(self, job_id: str, queue_limit_seconds: float) -> str:
        """Wait until the job leaves the queue. Return "ended" or "queue_limit".

        The queue time adds up over all waits in the queue, also after a requeue.
        """
        queued = 0.0
        last = self.clock()
        errors = 0
        while True:
            try:
                state = self.queue_state(job_id)
                errors = 0
            except (SlurmError, subprocess.TimeoutExpired):
                errors += 1
                if errors > 10:
                    raise
                state = "UNKNOWN"
            now = self.clock()
            if state is None:
                return "ended"
            if state in QUEUE_WAIT_STATES:
                queued += now - last
            last = now
            if queued > queue_limit_seconds:
                return "queue_limit"
            self.sleep(self.poll_seconds)

    def accounting(self, job_id: str, attempts: int = 10) -> Accounting:
        """Return the final state and the GPU minutes of all runs of the job."""
        found = None
        for _ in range(attempts):
            result = self.run(["sacct", "-j", job_id, "-X", "-D", "-n", "-P",
                               "-o", "JobIDRaw,State,ElapsedRaw,AllocTRES"], timeout=60)
            found = parse_sacct(result.stdout) if result.returncode == 0 else None
            if found and found.state in FINAL_STATES:
                return found
            self.sleep(30)  # the accounting database can be late after the job ends
        if found:
            return found
        raise SlurmError(f"sacct has no record of job {job_id}")

    def cancel(self, job_id: str) -> None:
        self.run(["scancel", job_id], timeout=10)
