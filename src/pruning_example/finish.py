"""The final steps: select a round, run the confirmation round, evaluate on the test set,
write the summary, run the report step, and copy the model."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from . import final_eval, metrics, records
from .agent import AgentError, tokens_used
from .slurm import SlurmError

RESULT_KEYS = ("val_accuracy", "val_interval", "nonzero", "weights_left", "sparse_size_mb", "meets_target")


def select_round(history: list[dict], target: dict) -> dict | None:
    """Select the round for the final steps.

    Use the round that met the target. If no round met it, use the round with the fewest
    weights left among the rounds with enough validation accuracy. If no round has enough
    accuracy, use the round with the highest validation accuracy.
    """
    ok = [row for row in history if row["status"] == "ok"]
    met = [row for row in ok if row["meets_target"] == "yes"]
    if met:
        return min(met, key=lambda row: int(row["nonzero"]))
    accurate = [row for row in ok if float(row["val_accuracy"]) >= target["val_accuracy"]]
    if accurate:
        return min(accurate, key=lambda row: int(row["nonzero"]))
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
    summary = {
        "stop_reason": stop_reason,
        "final_steps_done": final,
        "time": datetime.now().isoformat(timespec="seconds"),
        "claude_version": os.environ.get("CLAUDE_VERSION", "unknown"),
        "target": {**ctx.experiment["target"],
                   "max_nonzero": metrics.max_nonzero(ctx.experiment["target"]["size_target"])},
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
    prompt = (ctx.paths.project / "prompts" / "report_step.md").read_text()
    try:
        result = ctx.agent.run(prompt, ctx.experiment["report_step"])
    except AgentError as error:
        ctx.log(f"report step failed: {error}")
        return
    records.write_json(ctx.paths.state / "agent_steps" / "report.json", result)
    ctx.ledger.add(round="", kind="report", status=result.get("subtype", ""),
                   tokens=tokens_used(result), cost_usd=result.get("total_cost_usd", 0))
    reports = ctx.paths.project / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "final_report.md").write_text(str(result.get("result", "")).strip() + "\n")


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
        if model.exists():
            (ctx.paths.project / "results" / "final").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model, ctx.paths.project / "results" / "final" / f"round_{round_no:02d}_model.pt")
    write_summary(ctx, stop_reason, True, selected, confirmation, tests)
    run_report_step(ctx)
    return write_summary(ctx, stop_reason, True, selected, confirmation, tests)
