"""The final steps: select a round, run the confirmation rounds, evaluate on the test set,
write the summary, run the report step, and copy the model."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from . import config, final_eval, metrics, records
from .agent import AgentError, ModelCheckError, check_model, tokens_used
from .slurm import SlurmError

RESULT_KEYS = ("val_accuracy", "val_interval", "nonzero", "weights_left", "sparse_size_mb", "passes")
SELECTION_FILE = "selection.json"
DEFAULT_TEST_TARGET = 0.900  # the goal of the example: a test accuracy of at least 90%


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


def confirmation_seeds(experiment: dict) -> list[int]:
    """Return the seeds of the confirmation rounds. The default is one confirmation round with seed 1."""
    return [int(seed) for seed in experiment.get("confirmation_seeds", [1])]


def saved_selection(ctx, history: list[dict]) -> int | None:
    """Return the selected round. Select it the first time, and save it before the first confirmation round
    or test evaluation, so that a later --finish uses the same round."""
    path = ctx.paths.state / SELECTION_FILE
    if path.exists():
        return records.read_json(path)["round"]
    row = select_round(history, ctx.experiment["target"])
    round_no = int(row["round"]) if row is not None else None
    records.write_json(path, {"round": round_no, "time": datetime.now().isoformat(timespec="seconds")})
    return round_no


def confirmation_round(ctx, round_no: int) -> dict | None:
    """Run the selected configuration again with seed 1. Return its metrics, or None."""
    state = ctx.paths.state
    record = records.record_dir(state, "confirm", round_no, 1) / "metrics.json"
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


def confirmation_rounds(ctx, round_no: int, seeds: list[int]) -> dict[int, dict | None]:
    """Run the confirmation rounds of the selected configuration. Return the metrics of each seed, or None.

    One seed runs as one round job, as in chapter 28. Several seeds run as one array job. A seed that
    already has its metrics does not run again.
    """
    if seeds == [1]:
        return {1: confirmation_round(ctx, round_no)}
    state, project = ctx.paths.state, ctx.paths.project
    missing = [seed for seed in seeds
               if not (records.record_dir(state, "confirm", round_no, seed) / "metrics.json").exists()]
    if missing:
        config_path = (records.record_dir(state, "round", round_no) / "config.json").relative_to(project)
        tasks = state / "confirm" / f"round_{round_no:02d}_tasks.tsv"
        tasks.parent.mkdir(parents=True, exist_ok=True)
        tasks.write_text("".join(f"{config_path}\t{round_no}\tconfirm\t{seed}\n" for seed in missing))
        max_parallel = ctx.experiment.get("wave", {}).get("max_parallel", len(missing))
        try:
            job = ctx.slurm.submit_array("slurm/round_array.sbatch", str(tasks.relative_to(project)),
                                         len(missing), max_parallel, project)
        except (SlurmError, subprocess.TimeoutExpired) as error:
            ctx.log(f"confirmation rounds: sbatch failed: {error}")
        else:
            (state / "current_job").write_text(job + "\n")
            ctx.log(f"confirmation rounds: submitted array job {job} for seeds {', '.join(map(str, missing))}")
            outcome = ctx.slurm.wait_array(job, 60 * ctx.budget["queue_limit_minutes"])
            usage = ctx.slurm.array_accounting(job, len(missing))
            (state / "current_job").unlink(missing_ok=True)
            for index, seed in enumerate(missing):
                task = usage.get(index)
                ctx.ledger.add(round=round_no, kind="confirm", status=outcome,
                               job_id=task.job_id if task else f"{job}_{index}",
                               # without accounting, count the time limit, so that the GPU total stays safe
                               gpu_minutes=task.gpu_minutes if task else ctx.experiment["round"]["time_minutes"],
                               slurm_state=task.state if task else "UNKNOWN")
    results = {}
    for seed in seeds:
        record = records.record_dir(state, "confirm", round_no, seed) / "metrics.json"
        results[seed] = records.read_json(record) if record.exists() else None
    return results


def evaluate_on_test_set(ctx, kind: str, round_no: int, seed: int | None = None) -> dict | None:
    """Return the one-time test evaluation of a model. Run it if it does not exist yet."""
    out = records.test_eval_path(ctx.paths.state, kind, round_no, seed)
    if not out.exists():
        final_eval.main(["--round", str(round_no), "--kind", kind] + ([] if seed is None else ["--seed", str(seed)]))
    return records.read_json(out) if out.exists() else None


def seed_means(selected: dict, confirmations: dict[int, dict | None], tests: dict, target: dict) -> dict:
    """Return the means of the selected round and its confirmation rounds on the validation and the test set.

    The result holds if both means keep their thresholds. Only the test set is independent of the selection.
    """
    complete = all(confirmations.values())
    vals = [selected["val_accuracy"]] + [result["val_accuracy"] for result in confirmations.values() if result]
    test_keys = ["selected"] + [("confirmation" if seed == 1 else f"confirmation_seed{seed}") for seed in confirmations]
    test_values = [tests[key]["test_accuracy"] for key in test_keys if tests.get(key)]
    val_mean = sum(vals) / len(vals) if complete else None
    test_mean = sum(test_values) / len(test_values) if complete and len(test_values) == len(test_keys) else None
    test_target = target.get("test_accuracy", DEFAULT_TEST_TARGET)
    holds = None if val_mean is None or test_mean is None else (
        val_mean >= target["val_accuracy"] and test_mean >= test_target)
    return {
        "seeds": 1 + len(confirmations),
        "complete": complete and test_mean is not None,
        "val_accuracy": None if val_mean is None else round(val_mean, 5),
        "test_accuracy": None if test_mean is None else round(test_mean, 5),
        "test_target": test_target,
        "holds": holds,
    }


def previous_stop_reason(ctx) -> str | None:
    """Return the stop reason of an earlier loop job, or None."""
    path = ctx.paths.state / "summary.json"
    return records.read_json(path)["stop_reason"] if path.exists() else None


def agent_models(ctx) -> list[str]:
    """Return the models that the agent steps and the report step used."""
    models = set()
    for path in sorted((ctx.paths.state / "agent_steps").glob("*.json")):
        models.update(records.read_json(path).get("modelUsage", {}).keys())
    return sorted(models)


def write_summary(ctx, stop_reason: str, final: bool, selected=None, confirmation=None, tests=None,
                  terminated: bool = False, confirmations=None, means=None, extra=None) -> dict:
    """Write state/summary.json. confirmations and means appear only with more than one confirmation seed,
    and extra adds the fields of the multi-agent loop."""
    budget = ctx.budget
    history = records.read_history(ctx.paths.state)
    experiment_file = config.experiment_path(ctx.paths.project)
    summary = {
        "stop_reason": stop_reason,
        "final_steps_done": final,
        "terminated": terminated,
        "time": datetime.now().isoformat(timespec="seconds"),
        "claude_version": os.environ.get("CLAUDE_VERSION", "unknown"),
        "agent_models": agent_models(ctx),
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
    if confirmations is not None:
        summary["confirmations"] = confirmations
    if means is not None:
        summary["means"] = means
    summary.update(extra or {})
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
    try:
        check_model(result, ctx.experiment["report_step"].get("model"))
    except ModelCheckError as error:
        ctx.log(f"report step: {error}")
    (ctx.paths.state / "final_report.md").write_text(str(result.get("result", "")).strip() + "\n")


def run_final_steps(ctx, stop_reason: str, report_step=None, extra=None) -> dict:
    """Select a round, confirm it, evaluate it on the test set, copy the model, and write the summary.

    report_step replaces the report step of chapter 28, and extra(ctx) returns more summary fields;
    the multi-agent loop uses both.
    """
    state = ctx.paths.state
    seeds = confirmation_seeds(ctx.experiment)
    round_no = saved_selection(ctx, records.read_history(state))
    selected = confirmation = confirmations = means = None
    tests = {}
    if round_no is not None:
        folder = records.record_dir(state, "round", round_no)
        result = records.read_json(folder / "metrics.json")
        selected = {"round": round_no, "seed": result.get("seed", 0), "config": records.read_json(folder / "config.json"),
                    **{key: result[key] for key in RESULT_KEYS}}
        confirms = confirmation_rounds(ctx, round_no, seeds)
        if confirms.get(1):
            confirmation = {"seed": confirms[1]["seed"], **{key: confirms[1][key] for key in RESULT_KEYS}}
        tests["selected"] = evaluate_on_test_set(ctx, "round", round_no)
        for seed, confirm in confirms.items():
            if not confirm:
                continue
            if seed == 1:
                tests["confirmation"] = evaluate_on_test_set(ctx, "confirm", round_no)
            else:
                tests[f"confirmation_seed{seed}"] = evaluate_on_test_set(ctx, "confirm", round_no, seed)
        if len(seeds) > 1:
            confirmations = [{"seed": seed, **({key: confirm[key] for key in RESULT_KEYS} if confirm else {"missing": True})}
                             for seed, confirm in confirms.items()]
            means = seed_means(result, confirms, tests, ctx.experiment["target"])
        model = Path(records.read_json(folder / "run_meta.json")["run_folder"]) / "model.pt"
        if model.exists():  # scratch files are temporary, so keep a copy on lab storage
            target = ctx.paths.project / "results" / state.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model, target / f"round_{round_no:02d}_model.pt")

    def summary() -> dict:
        return write_summary(ctx, stop_reason, True, selected, confirmation, tests, confirmations=confirmations,
                             means=means, extra=extra(ctx) if extra else None)

    summary()
    (report_step or run_report_step)(ctx)
    return summary()
