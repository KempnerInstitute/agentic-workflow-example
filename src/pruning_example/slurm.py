"""Small wrappers around sbatch, squeue, sacct, and scancel, for the loop scripts."""

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
# sbatch errors of a busy SLURM controller. A new try can succeed.
TRANSIENT_ERRORS = ("Socket timed out", "timed out", "Resource temporarily unavailable", "try again")


class SlurmError(RuntimeError):
    """A SLURM command failed."""


@dataclass
class Accounting:
    state: str
    gpu_minutes: float
    runs: int


@dataclass
class TaskAccounting:
    """The accounting of one task of an array job: all of its runs, requeues included."""

    task: int
    job_id: str  # the job ID of the task (JobIDRaw), or the array ID for a task that never started
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


def task_indexes(job_id: str) -> list[int]:
    """Return the task indexes in a JobID of an array job: 123_4, or a range of waiting tasks such as 123_[1-2%3]."""
    match = re.fullmatch(r"\d+_(\d+)", job_id)
    if match:
        return [int(match.group(1))]
    match = re.fullmatch(r"\d+_\[(.*)\]", job_id)
    if not match:
        return []
    indexes = []
    for part in match.group(1).split("%")[0].split(","):
        if "-" in part:
            first, last = part.split("-")
            indexes.extend(range(int(first), int(last) + 1))
        elif part:
            indexes.append(int(part))
    return indexes


def parse_array_sacct(text: str) -> dict[int, TaskAccounting]:
    """Read the output of sacct -X -D -n -P -o JobID,JobIDRaw,State,Start,ElapsedRaw,AllocTRES for an array job.

    -D gives one line for each run of a requeued task. Tasks that never started can share one line
    with a range of indexes; they have no GPU time.
    """
    runs: dict[int, list[list[str]]] = {}
    for line in text.strip().splitlines():
        row = line.split("|")
        if len(row) < 6:
            continue
        for index in task_indexes(row[0]):
            runs.setdefault(index, []).append(row)
    tasks = {}
    for index, rows in runs.items():
        # Order the runs by Start. A run that never started (a waiting run that was canceled) comes last.
        rows.sort(key=lambda row: (0, row[3]) if row[3][:1].isdigit() else (1, ""))
        minutes = sum(int(row[4] or 0) * gpu_count(row[5]) for row in rows) / 60
        last = rows[-1]
        tasks[index] = TaskAccounting(task=index, job_id=last[1], state=last[2].split()[0],
                                      gpu_minutes=round(minutes, 2), runs=sum(int(row[4] or 0) > 0 for row in rows))
    return tasks


def parse_array_squeue(text: str) -> dict[int, str]:
    """Read the output of squeue -h -r -o "%K %T": one line for each task of an array job."""
    states = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            states[int(parts[0])] = parts[1]
    return states


class Slurm:
    """The SLURM commands of the loop scripts. The tests replace this class with a fake."""

    def __init__(self, poll_seconds: int = 60, sleep=time.sleep, clock=time.monotonic):
        self.poll_seconds = poll_seconds
        self.sleep = sleep
        self.clock = clock

    def run(self, args: list[str], timeout: int = 120, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kwargs)

    def submit(self, script: str, args: list[str], cwd, attempts: int = 3, options: tuple = ()) -> str:
        """Submit a job and return its ID. Try again after a timeout of the busy SLURM controller."""
        for attempt in range(1, attempts + 1):
            result = self.run(["sbatch", "--parsable", *options, script, *args], cwd=cwd, env=clean_env())
            if result.returncode == 0:
                return result.stdout.strip().split(";")[0]
            error = result.stderr.strip()
            if attempt == attempts or not any(text in error for text in TRANSIENT_ERRORS):
                raise SlurmError(f"sbatch failed: {error}")
            self.sleep(30)
        raise SlurmError("sbatch failed")

    def submit_array(self, script: str, tasks_file: str, count: int, max_parallel: int, cwd) -> str:
        """Submit an array job with one task for each line of tasks_file, at most max_parallel at a time."""
        return self.submit(script, [tasks_file], cwd, options=(f"--array=0-{count - 1}%{max_parallel}",))

    def queue_state(self, job_id: str) -> str | None:
        """Return the state of the job in the queue, or None if the job left the queue."""
        result = self.run(["squeue", "-h", "-j", job_id, "-o", "%T"], timeout=60)
        if result.returncode != 0:
            if "Invalid job id" in result.stderr:
                return None
            raise SlurmError(f"squeue failed: {result.stderr.strip()}")
        return result.stdout.strip() or None

    def array_states(self, array_id: str) -> dict[int, str]:
        """Return the state of each task of an array job that is still in the queue. -r gives one line for
        each task, also for tasks that have not started."""
        result = self.run(["squeue", "-h", "-r", "-j", array_id, "-o", "%K %T"], timeout=60)
        if result.returncode != 0:
            if "Invalid job id" in result.stderr:
                return {}
            raise SlurmError(f"squeue failed: {result.stderr.strip()}")
        return parse_array_squeue(result.stdout)

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

    def wait_array(self, array_id: str, queue_limit_seconds: float) -> str:
        """Wait until all tasks of an array job leave the queue. Return "ended" or "queue_limit".

        Queue time counts while at least one task waits. After the queue limit, the waiting tasks are
        canceled, also a task that a preemption put back in the queue, and the running tasks can end.
        """
        queued = 0.0
        last = self.clock()
        errors = 0
        over_limit = False
        while True:
            try:
                states = self.array_states(array_id)
                errors = 0
            except (SlurmError, subprocess.TimeoutExpired):
                errors += 1
                if errors > 10:
                    raise
                states = None
            now = self.clock()
            if states == {}:
                return "queue_limit" if over_limit else "ended"
            waiting = bool(states) and any(state in QUEUE_WAIT_STATES for state in states.values())
            if waiting and not over_limit:
                queued += now - last
            last = now
            if queued > queue_limit_seconds:
                over_limit = True
            if over_limit and waiting:
                self.cancel_waiting(array_id)
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

    def array_accounting(self, array_id: str, count: int, attempts: int = 10) -> dict[int, TaskAccounting]:
        """Return the accounting of each task of an array job. Wait until each task has a final state.

        A task that sacct does not show is left out; the caller counts its reserved GPU minutes.
        """
        found: dict[int, TaskAccounting] = {}
        for _ in range(attempts):
            try:
                result = self.run(["sacct", "-j", array_id, "-X", "-D", "-n", "-P",
                                   "-o", "JobID,JobIDRaw,State,Start,ElapsedRaw,AllocTRES"], timeout=60)
                if result.returncode == 0:
                    found = parse_array_sacct(result.stdout)
            except subprocess.TimeoutExpired:
                pass
            if all(index in found and found[index].state in FINAL_STATES for index in range(count)):
                return found
            self.sleep(30)  # the accounting database can be late after the job ends
        return found

    def cancel(self, job_id: str, wait_seconds: float = 60) -> bool:
        """Ask SLURM to cancel the job, or all tasks of an array job. Return True if scancel answered in time.

        Do not raise: on a busy cluster, scancel can answer late, and SLURM cancels the job anyway.
        """
        return self._scancel([job_id], wait_seconds)

    def cancel_waiting(self, array_id: str, wait_seconds: float = 60) -> bool:
        """Cancel only the waiting tasks of an array job. The running tasks continue."""
        return self._scancel(["--state=PENDING", array_id], wait_seconds)

    def _scancel(self, args: list[str], wait_seconds: float) -> bool:
        try:
            process = subprocess.Popen(["scancel", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.wait(timeout=wait_seconds)
            return process.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
