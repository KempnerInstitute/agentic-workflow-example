"""The loop script: run the unattended phase, then the final steps.

Usage: python -m pruning_example.loop [--finish]

slurm/loop.sbatch starts this script with exec, so the script gets the SLURM signals.
With --finish, the script runs only the final steps, for example after an early stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import config, finish, records
from .agent import Agent, AgentError, AgentFormatError, tokens_used
from .config import ConfigError
from .records import IntegrityError
from .slurm import Slurm, SlurmError

LEDGER_FIELDS = ["time", "round", "kind", "status", "tokens", "cost_usd", "job_id", "gpu_minutes", "slurm_state"]


class Terminated(Exception):
    """SLURM sent SIGTERM: scancel of the loop job, or the time limit is near."""


class StopLoop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def on_sigterm(signum, frame):
    raise Terminated()


def log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


class Ledger:
    """The budget record in state/ledger.tsv: tokens of each claude -p call and GPU minutes of each job."""

    def __init__(self, path: Path):
        self.path = path
        self.rows = []
        if path.exists():
            with open(path, newline="") as f:
                self.rows = list(csv.DictReader(f, delimiter="\t"))

    def add(self, **row) -> None:
        row = {"time": datetime.now().isoformat(timespec="seconds"), **row}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, delimiter="\t", extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in LEDGER_FIELDS})
        self.rows.append({name: str(row.get(name, "")) for name in LEDGER_FIELDS})

    def total(self, column: str, kinds: set[str]) -> float:
        return sum(float(row[column] or 0) for row in self.rows if row["kind"] in kinds)

    @property
    def tokens(self) -> int:
        return int(self.total("tokens", {"agent_step", "report"}))

    @property
    def largest_step(self) -> int:
        return max((int(float(row["tokens"] or 0)) for row in self.rows if row["kind"] == "agent_step"), default=0)

    @property
    def gpu_minutes(self) -> float:
        """GPU minutes of the unattended rounds. The confirmation round is separate."""
        return self.total("gpu_minutes", {"round"})

    @property
    def cost_usd(self) -> float:
        return self.total("cost_usd", {"agent_step", "report"})


@dataclass
class Context:
    paths: config.Paths
    experiment: dict
    schema_text: str
    agent: Agent
    slurm: Slurm
    ledger: Ledger
    job_end: float | None = None
    clock: object = time.time
    log: object = log
    schema: dict = field(init=False)

    def __post_init__(self):
        self.schema = json.loads(self.schema_text)

    @property
    def budget(self) -> dict:
        return self.experiment["budget"]


def failed_in_a_row(history: list[dict]) -> int:
    count = 0
    for row in reversed(history):
        if row["status"] == "ok":
            break
        count += 1
    return count


def stop_reason(ctx: Context) -> str | None:
    """Return the reason to stop before the next round, or None."""
    budget = ctx.budget
    history = records.read_history(ctx.paths.state)
    if any(row["status"] == "ok" and row["meets_target"] == "yes" for row in history):
        return "target"
    if len(history) >= budget["max_round_attempts"]:
        return "round_limit"
    step_limit = budget["tokens"] * (1 - budget["report_reserve_fraction"])
    next_step = ctx.ledger.largest_step or budget["step_token_estimate"]
    if ctx.ledger.tokens + next_step > step_limit:
        return "token_budget"
    if ctx.ledger.gpu_minutes >= budget["gpu_minutes"]:
        return "gpu_budget"
    if failed_in_a_row(history) >= budget["max_failed_in_a_row"]:
        return "failed_attempts"
    if ctx.job_end is not None:
        needed = ctx.experiment["round"]["time_minutes"] + budget["queue_limit_minutes"] + budget["final_reserve_minutes"]
        if ctx.job_end - ctx.clock() < 60 * needed:
            return "time_left"
    return None


def render(path: Path, values: dict) -> str:
    text = path.read_text()
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def fail(ctx: Context, round_no: int, status: str, note: str, round_config: dict | None = None) -> None:
    """Write the history row of a failed round attempt."""
    records.append_history(ctx.paths.state, records.history_row(round_no, status, round_config, None, note[:200]))
    ctx.log(f"round {round_no}: {status}: {note}")


def run_attempt(ctx: Context, round_no: int) -> None:
    """Run one round attempt: agent step, validation, round job, records."""
    state = ctx.paths.state
    records.check_integrity(ctx.paths.project)
    history = records.read_history(state)
    prompt = render(ctx.paths.project / "prompts" / "agent_step.md", {
        "STATE": state.relative_to(ctx.paths.project),
        "N": round_no,
        "R": ctx.budget["max_round_attempts"] - len(history),
        "G": round(max(0.0, ctx.budget["gpu_minutes"] - ctx.ledger.gpu_minutes)),
    })
    try:
        result = ctx.agent.run(prompt, ctx.experiment["agent_step"], ctx.schema_text)
    except AgentFormatError:
        raise
    except AgentError as error:
        ctx.ledger.add(round=round_no, kind="agent_step", status="failed", tokens=0)
        fail(ctx, round_no, "agent_failed", str(error))
        return
    records.write_json(state / "agent_steps" / f"round_{round_no:02d}.json", result)
    tokens = tokens_used(result)
    ctx.ledger.add(round=round_no, kind="agent_step", status=result.get("subtype", ""),
                   tokens=tokens, cost_usd=result.get("total_cost_usd", 0))

    round_config = result.get("structured_output")
    try:
        if not isinstance(round_config, dict):
            raise ConfigError(f"no structured_output (subtype {result.get('subtype')})")
        config.validate_round_config(round_config, ctx.schema)
    except ConfigError as error:
        fail(ctx, round_no, "invalid_config", str(error))
        return
    config_path = state.relative_to(ctx.paths.project) / "configs" / f"round_{round_no:02d}.json"
    records.write_json(ctx.paths.project / config_path, round_config)
    with open(state / "notes.md", "a") as notes:
        notes.write(f"## Round {round_no}\n\n{round_config['reason']}\n\n")

    try:
        job = ctx.slurm.submit("slurm/round.sbatch", [str(config_path), "--round", str(round_no)], ctx.paths.project)
    except (SlurmError, subprocess.TimeoutExpired) as error:
        fail(ctx, round_no, "sbatch_failed", str(error), round_config)
        return
    (state / "current_job").write_text(job + "\n")
    ctx.log(f"round {round_no}: submitted job {job}")
    outcome = ctx.slurm.wait(job, 60 * ctx.budget["queue_limit_minutes"])
    if outcome == "queue_limit":
        ctx.slurm.cancel(job)
    usage = ctx.slurm.accounting(job)
    (state / "current_job").unlink(missing_ok=True)
    ctx.ledger.add(round=round_no, kind="round", status=outcome, job_id=job,
                   gpu_minutes=usage.gpu_minutes, slurm_state=usage.state)

    rows = [row for row in records.read_history(state) if int(row["round"]) == round_no]
    if outcome == "queue_limit":
        if not rows:
            fail(ctx, round_no, "queue_limit", f"job {job} waited longer than the queue limit", round_config)
        raise StopLoop("queue_limit")
    if not rows:
        fail(ctx, round_no, "job_failed", f"job {job} ended as {usage.state} without metrics", round_config)
        return
    row = rows[-1]
    ctx.log(f"round {round_no}: val_accuracy={row['val_accuracy']} weights_left={row['weights_left']} "
            f"meets_target={row['meets_target']} tokens={tokens} gpu_minutes={usage.gpu_minutes} "
            f"(totals: {ctx.ledger.tokens} tokens, {ctx.ledger.gpu_minutes:.1f} GPU minutes)")


def run_rounds(ctx: Context) -> str:
    """Run round attempts until a stop rule applies. Return the stop reason."""
    while True:
        reason = stop_reason(ctx)
        if reason:
            return reason
        try:
            run_attempt(ctx, records.next_round(ctx.paths.state))
        except StopLoop as stop:
            return stop.reason
        except IntegrityError as error:
            ctx.log(str(error))
            return "integrity"
        except AgentFormatError as error:
            ctx.log(f"{error}. Check the Claude Code version.")
            return "agent_format"


def check_start(ctx: Context) -> None:
    """Refuse to start without the baseline and round 1, or with settings that do not agree."""
    project = ctx.paths.project
    if not (project / "state" / "baseline" / "metrics.json").exists():
        raise SystemExit("There is no baseline. Run 'sbatch slurm/baseline.sbatch' first.")
    if not (ctx.paths.state / "rounds" / "round_01" / "metrics.json").exists():
        raise SystemExit("There is no round 1. Run the supervised round first.")
    match = re.search(r"^#SBATCH --time=(\d+):(\d+):(\d+)", (project / "slurm" / "round.sbatch").read_text(), re.M)
    minutes = int(match.group(1)) * 60 + int(match.group(2)) if match else None
    if minutes != ctx.experiment["round"]["time_minutes"]:
        raise SystemExit(f"slurm/round.sbatch has --time of {minutes} minutes, but experiment.yaml "
                         f"has round.time_minutes {ctx.experiment['round']['time_minutes']}.")
    records.check_integrity(project)
    notes = ctx.paths.state / "notes.md"
    if not notes.exists():
        first = records.read_json(ctx.paths.state / "rounds" / "round_01" / "config.json")
        notes.write_text("# Notes\n\nThe loop script adds the agent's reason for each round here.\n\n"
                         f"## Round 1\n\n{first.get('reason', '')}\n\n")


def job_end_time() -> float | None:
    """Return the end time of the loop job as a Unix time, or None outside SLURM."""
    if os.environ.get("SLURM_JOB_END_TIME"):
        return float(os.environ["SLURM_JOB_END_TIME"])
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True, timeout=60)
    try:
        return datetime.fromisoformat(result.stdout.strip()).timestamp()
    except ValueError:
        return None


def run(ctx: Context, finish_only: bool = False) -> int:
    """Run the rounds and the final steps. On SIGTERM, cancel the running job and stop."""
    reason = "finish_only"
    ctx.log(f"experiment file: {config.experiment_path(ctx.paths.project).relative_to(ctx.paths.project)}, "
            f"state folder: {ctx.paths.state.relative_to(ctx.paths.project)}")
    try:
        if not finish_only:
            check_start(ctx)
            reason = run_rounds(ctx)
            ctx.log(f"stop: {reason}")
        finish.run_final_steps(ctx, reason)
        ctx.log(f"final steps done: see {ctx.paths.state.name}/summary.json and {ctx.paths.state.name}/final_report.md")
        return 0
    except Terminated:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        ctx.agent.stop()
        current = ctx.paths.state / "current_job"
        if current.exists():
            ctx.slurm.cancel(current.read_text().strip())
            current.unlink(missing_ok=True)
        finish.write_summary(ctx, "terminated", final=False)
        ctx.log("SIGTERM: the loop script canceled its own round job and stopped. "
                "Run 'sbatch slurm/loop.sbatch --finish' for the final steps.")
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the unattended phase, then the final steps.")
    parser.add_argument("--finish", action="store_true", help="run only the final steps")
    args = parser.parse_args(argv)
    paths = config.paths_from_env()
    ctx = Context(
        paths=paths,
        experiment=config.load_experiment(paths.project),
        schema_text=(paths.project / config.SCHEMA_PATH).read_text(),
        agent=Agent(paths.project),
        slurm=Slurm(),
        ledger=Ledger(paths.state / "ledger.tsv"),
        job_end=job_end_time(),
    )
    signal.signal(signal.SIGTERM, on_sigterm)
    return run(ctx, finish_only=args.finish)


if __name__ == "__main__":
    raise SystemExit(main())
