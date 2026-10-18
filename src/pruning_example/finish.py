"""The final steps: select a round, run the confirmation round, evaluate on the test set,
write the summary, run the report step, and copy the model."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from . import config, final_eval, metrics, records
from .agent import AgentError, tokens_used
from .slurm import SlurmError

RESULT_KEYS = ("val_accuracy", "val_interval", "nonzero", "weights_left", "sparse_size_mb", "passes")


def row_passes(row: dict, target: dict) -> bool:
    """Compare a history row with the current threshold, not with the decision stored at the time of the round."""
    return row["status"] == "ok" and metrics.passes(float(row["val_accuracy"]), target)


def best_round(history: list[dict], target: dict) -> dict | None:
    """Return the round with the fewest weights left among the rounds that keep the threshold."""
    passing = [row for row in history if row_passes(row, target)]
    return min(passing, key=lambda row: int(row["nonzero"]), default=None)


def rounds_without_improvement(history: list[dict], target: dict) -> int:
    """Count the complete rounds since the last round that kept the threshold with fewer weights."""
    best, count = None, 0
    for row in history:
        if row["status"] != "ok":
            continue
        if row_passes(row, target) and (best is None or int(row["nonzero"]) < best):
            best, count = int(row["nonzero"]), 0
        else:
            count += 1
    return count


def select_round(history: list[dict], target: dict) -> dict | None:
    """Select the round for the final steps: the fewest weights left among the rounds that keep
    the threshold. If no round keeps the threshold, select the round with the highest validation accuracy."""
    best = best_round(history, target)
    if best is not None:
        return best
    ok = [row for row in history if row["status"] == "ok"]
    return max(ok, key=lambda row: float(row["val_accuracy"]), default=None)


def confirmation_round(ctx, round_no: int) -> dict | None:
    """Run the selected configuration again with seed 1. Return its metrics, or None."""
    state = ctx.paths.state
    record = state / "confirm" / f"round_{round_no:02d}" / "metrics.json"
    if record.exists():
        return records.read_json(record)
    config_path = state.relative_to(ctx.paths.project) / "rounds" / f"round_{round_no:02d}" / "config.json"
    args = [str(config_path), "--round", str(round_no), "--kind", "confirm", "--seed", "1"]
    try:
        job = ctx.slurm.submit("slurm/round.sbatch", args, ctx.paths.project)
    except (SlurmError, subprocess.TimeoutExpired) as error:
        ctx.log(f"confirmation round: sbatch failed: {error}")
        return None
    (state / "current_job").write_text(job + "\n")
    ctx.log(f"confirmation round: submitted job {job}")
    outcome = ctx.slurm.wait(job, 60 * ctx.budget["queue_limit_minutes"])
    if outcome == "queue_limit":
        ctx.slurm.cancel(job)
    usage = ctx.slurm.accounting(job)
    (state / "current_job").unlink(missing_ok=True)
    ctx.ledger.add(round=round_no, kind="confirm", status=outcome, job_id=job,
                   gpu_minutes=usage.gpu_minutes, slurm_state=usage.state)
    return records.read_json(record) if record.exists() else None


def evaluate_on_test_set(ctx, kind: str, round_no: int) -> dict | None:
    """Return the one-time test evaluation of a model. Run it if it does not exist yet."""
    out = ctx.paths.state / "test_eval" / f"{kind}_{round_no:02d}.json"
    if not out.exists():
        final_eval.main(["--round", str(round_no), "--kind", kind])
    return records.read_json(out) if out.exists() else None


def write_summary(ctx, stop_reason: str, final: bool, selected=None, confirmation=None, tests=None) -> dict:
    budget = ctx.budget
    history = records.read_history(ctx.paths.state)
    experiment_file = config.experiment_path(ctx.paths.project)
    summary = {
        "stop_reason": stop_reason,
        "final_steps_done": final,
        "time": datetime.now().isoformat(timespec="seconds"),
        "claude_version": os.environ.get("CLAUDE_VERSION", "unknown"),
        "experiment_file": str(experiment_file.relative_to(ctx.paths.project)),
        "experiment_sha256": hashlib.sha256(experiment_file.read_bytes()).hexdigest(),
        "target": ctx.experiment["target"],
        "round_attempts": len(history),
        "rounds_ok": sum(row["status"] == "ok" for row in history),
        "tokens": {
            "total": ctx.ledger.tokens,
            "budget": budget["tokens"],
            "largest_step": ctx.ledger.largest_step,
            "estimated_cost_usd": round(ctx.ledger.cost_usd, 2),
        },
        "gpu_minutes": {
            "unattended": round(ctx.ledger.gpu_minutes, 1),
            "confirmation": round(ctx.ledger.total("gpu_minutes", {"confirm"}), 1),
            "budget": budget["gpu_minutes"],
        },
        "selected": selected,
        "confirmation": confirmation,
        "test": tests or None,
    }
    records.write_json(ctx.paths.state / "summary.json", summary)
    return summary


def run_report_step(ctx) -> None:
    """Ask the agent for a draft report. Skip the step if the token budget is too low."""
    estimate = ctx.ledger.largest_step or ctx.budget["step_token_estimate"]
    if ctx.ledger.tokens + estimate > ctx.budget["tokens"]:
        ctx.log("report step skipped: not enough tokens are left")
        return
    prompt = (ctx.paths.project / "prompts" / "report_step.md").read_text().replace(
        "{STATE}", str(ctx.paths.state.relative_to(ctx.paths.project)))
    try:
        result = ctx.agent.run(prompt, ctx.experiment["report_step"])
    except AgentError as error:
        ctx.log(f"report step failed: {error}")
        return
    records.write_json(ctx.paths.state / "agent_steps" / "report.json", result)
    ctx.ledger.add(round="", kind="report", status=result.get("subtype", ""),
                   tokens=tokens_used(result), cost_usd=result.get("total_cost_usd", 0))
    (ctx.paths.state / "final_report.md").write_text(str(result.get("result", "")).strip() + "\n")


def run_final_steps(ctx, stop_reason: str) -> dict:
    state = ctx.paths.state
    if stop_reason == "finish_only" and (state / "summary.json").exists():
        stop_reason = records.read_json(state / "summary.json")["stop_reason"]
    row = select_round(records.read_history(state), ctx.experiment["target"])
    selected = confirmation = None
    tests = {}
    if row is not None:
        round_no = int(row["round"])
        folder = state / "rounds" / f"round_{round_no:02d}"
        result = records.read_json(folder / "metrics.json")
        selected = {"round": round_no, "config": records.read_json(folder / "config.json"),
                    **{key: result[key] for key in RESULT_KEYS}}
        confirm = confirmation_round(ctx, round_no)
        if confirm:
            confirmation = {"seed": confirm["seed"], **{key: confirm[key] for key in RESULT_KEYS}}
        tests["selected"] = evaluate_on_test_set(ctx, "round", round_no)
        if confirm:
            tests["confirmation"] = evaluate_on_test_set(ctx, "confirm", round_no)
        model = Path(records.read_json(folder / "run_meta.json")["run_folder"]) / "model.pt"
        if model.exists():  # scratch files are temporary, so keep a copy on lab storage
            target = ctx.paths.project / "results" / state.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model, target / f"round_{round_no:02d}_model.pt")
    write_summary(ctx, stop_reason, True, selected, confirmation, tests)
    run_report_step(ctx)
    return write_summary(ctx, stop_reason, True, selected, confirmation, tests)
