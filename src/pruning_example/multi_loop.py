"""The multi-agent loop script: plan, check, and run waves of rounds, then the final steps.

Usage: python -m pruning_example.multi_loop [--plan-only | --finish | --audit-only]

slurm/multi_loop.sbatch starts this script with exec, so the script gets the SLURM signals.
--plan-only plans wave 1 and stops, so that you can read the plan before any GPU job runs.
--finish runs only the final steps, for example after an early stop.
--audit-only runs the auditor and check_report on the report of STATE_DIR.

Four roles do the agent work, each in its own claude -p call with read-only tools: the planner, the critic,
the writer, and the auditor. This script runs the role steps, checks their output in code, submits one
array job for each wave, writes all records, and applies the stop rules. experiment_multi.yaml sets the
model, the effort level, and the caps of each role.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from . import check_report, config, finish, records
from .agent import Agent, AgentError, AgentFormatError, EFFORT_LEVELS, ModelCheckError, check_model, tokens_used
from .config import ConfigError
from .loop import (
    LEDGER_FIELDS, Context, Ledger, StopLoop, Terminated, check_time_lines, describe_best, job_end_time, log,
    on_sigterm, render,
)
from .records import IntegrityError
from .slurm import Slurm, SlurmError

ROLES = ("planner", "critic", "writer", "auditor")
# The kinds of role steps in the ledger, and the role of each kind
ROLE_OF = {"planner": "planner", "critic": "critic", "revision": "planner", "critic_2": "critic",
           "writer": "writer", "auditor": "auditor"}
WAVE_KINDS = ("planner", "critic", "revision", "critic_2")
MULTI_LEDGER_FIELDS = LEDGER_FIELDS + ["wave", "model", "effort"]
EXTRA_KEYS = ("hypothesis", "predicted_val_accuracy", "pass_probability")
HYPOTHESIS_FIELDS = ["wave", "item", "round", "status", "verdict", "hypothesis", "predicted_val_accuracy",
                     "pass_probability", "val_accuracy", "passes", "error"]
ROUND_SCRIPTS = ("slurm/round.sbatch", "slurm/round_array.sbatch")
SCHEMA_FILES = {"wave": "configs/wave.schema.json", "review": "configs/review.schema.json",
                "audit": "configs/audit.schema.json"}


@dataclass
class MultiContext(Context):
    """The context of the loop of chapter 28, plus the schemas of the role outputs."""

    schemas: dict | None = None
    plan_only: bool = False


# Settings and limits

def roles(ctx) -> dict:
    return ctx.experiment["roles"]


def critic_on(ctx) -> bool:
    return bool(roles(ctx)["critic"].get("enabled", True))


def check_roles(experiment: dict) -> None:
    """Refuse to start without a model, a valid effort level, and the caps of each role."""
    for role in ROLES:
        settings = (experiment.get("roles") or {}).get(role) or {}
        if not settings.get("model"):
            raise SystemExit(f"Set roles.{role}.model in the experiment file.")
        if settings.get("effort") not in EFFORT_LEVELS:
            raise SystemExit(f"roles.{role}.effort must be one of {', '.join(EFFORT_LEVELS)}.")
        for key in ("max_turns", "max_budget_usd", "timeout_minutes"):
            if key not in settings:
                raise SystemExit(f"Set roles.{role}.{key} in the experiment file.")


def role_caps(ctx, role: str) -> dict:
    caps = {key: value for key, value in roles(ctx)[role].items() if key != "enabled"}
    caps["role_prompt"] = f"prompts/multi/roles/{role}.md"
    return caps


def rel(ctx, path: Path) -> Path:
    return path.relative_to(ctx.paths.project)


def wave_dir(ctx, wave: int) -> Path:
    return ctx.paths.state / "waves" / f"wave_{wave:02d}"


def wave_numbers(ctx) -> list[int]:
    folders = (ctx.paths.state / "waves").glob("wave_*")
    return sorted(int(folder.name.split("_")[1]) for folder in folders if folder.is_dir())


def search_limit(ctx) -> float:
    return ctx.budget["tokens"] * (1 - ctx.budget["final_reserve_fraction"])


def wave_tokens(ctx, wave: int) -> int:
    return int(sum(float(row["tokens"] or 0) for row in ctx.ledger.rows
                   if row["kind"] in WAVE_KINDS and row.get("wave") == str(wave)))


def wave_estimate(ctx) -> int:
    """The tokens of the next wave plan: the largest wave so far, or the estimate before the first wave."""
    done = [wave for wave in wave_numbers(ctx) if (wave_dir(ctx, wave) / "outcome.json").exists()]
    return max((wave_tokens(ctx, wave) for wave in done), default=0) or ctx.budget["wave_token_estimate"]


def gpu_room(ctx) -> int:
    """The rounds whose full time limit still fits in the GPU budget."""
    left = ctx.budget["gpu_minutes"] - ctx.ledger.gpu_minutes
    return max(0, math.floor(left / ctx.experiment["round"]["time_minutes"] + 1e-9))


def wave_size(ctx, history: list[dict]) -> int:
    """The smallest of the wave size, the GPU room, the round attempts left, and the rounds left before the
    plateau rule stops the run. So no limit is passed by a part of a wave."""
    target = ctx.experiment["target"]
    return max(0, min(ctx.experiment["wave"]["size"], gpu_room(ctx),
                      ctx.budget["max_round_attempts"] - len(history),
                      target["plateau_rounds"] - finish.rounds_without_improvement(history, target)))


def time_for(ctx, minutes: float) -> bool:
    return ctx.job_end is None or ctx.job_end - ctx.clock() >= 60 * minutes


def plan_minutes(ctx) -> float:
    """The longest time of the role steps of one wave: planner and revision, and two critic steps."""
    minutes = 2 * roles(ctx)["planner"]["timeout_minutes"]
    if critic_on(ctx):
        minutes += 2 * roles(ctx)["critic"]["timeout_minutes"]
    return minutes


def array_minutes(ctx) -> float:
    """The time that the loop job needs to run one array job and the final steps."""
    return (ctx.experiment["round"]["time_minutes"] + ctx.budget["queue_limit_minutes"]
            + ctx.budget["final_reserve_minutes"])


def failed_waves(ctx) -> int:
    """Count the waves in a row, from the last one, that ended with no complete round."""
    count = 0
    for wave in reversed(wave_numbers(ctx)):
        outcome = wave_dir(ctx, wave) / "outcome.json"
        if not outcome.exists():
            continue
        if records.read_json(outcome)["complete_rounds"] > 0:
            break
        count += 1
    return count


def stop_reason(ctx, planned: bool = False) -> str | None:
    """Return the reason to stop before the next wave, or None. A wave that is already planned needs no
    tokens and no time for its plan."""
    budget = ctx.budget
    target = ctx.experiment["target"]
    history = records.read_history(ctx.paths.state)
    if len(history) >= budget["max_round_attempts"]:
        return "round_limit"
    if finish.rounds_without_improvement(history, target) >= target["plateau_rounds"]:
        return "plateau"
    if not planned and ctx.ledger.tokens + wave_estimate(ctx) > search_limit(ctx):
        return "token_budget"
    if gpu_room(ctx) < 1:
        return "gpu_budget"
    if failed_waves(ctx) >= budget["max_failed_waves"]:
        return "failed_attempts"
    if not planned and not ctx.plan_only and not time_for(ctx, plan_minutes(ctx) + budget["final_reserve_minutes"]):
        return "time_left"
    return None


# Role steps

def run_role(ctx, kind: str, prompt: str, schema_text: str | None = None, wave: int | None = None) -> dict | None:
    """Run one role step. Return the JSON result, or None if the step failed.

    A result without token counts stops the loop (AgentFormatError), and so does a result that does not
    show the configured model (ModelCheckError).
    """
    caps = role_caps(ctx, ROLE_OF[kind])
    common = {"wave": wave or "", "model": caps["model"], "effort": caps["effort"]}
    if ctx.plan_only and not time_for(ctx, caps["timeout_minutes"]):
        raise StopLoop("time_left")
    try:
        result = ctx.agent.run(prompt, caps, schema_text)
    except AgentFormatError:
        raise
    except AgentError as error:
        ctx.ledger.add(round="", kind=kind, status="failed", tokens=0, **common)
        ctx.log(f"{f'wave {wave}: ' if wave else ''}{kind} step failed: {error}")
        return None
    name = f"wave_{wave:02d}_{kind}.json" if wave else f"{kind}.json"
    records.write_json(ctx.paths.state / "agent_steps" / name, result)
    ctx.ledger.add(round="", kind=kind, status=result.get("subtype", ""), tokens=tokens_used(result),
                   cost_usd=result.get("total_cost_usd", 0), **common)
    check_model(result, caps["model"])
    return result


def structured(result: dict | None, schema: dict) -> dict | None:
    """Return the structured output of a role step if it matches the schema. The documentation does not
    list the schema keywords that Claude Code supports, so the code checks the output again."""
    output = (result or {}).get("structured_output")
    if not isinstance(output, dict):
        return None
    try:
        jsonschema.validate(output, schema)
    except jsonschema.ValidationError:
        return None
    return output


def wave_schema(ctx, count: int) -> dict:
    schema = json.loads(ctx.schemas["wave"])
    schema["properties"]["configurations"]["minItems"] = count
    schema["properties"]["configurations"]["maxItems"] = count
    return schema


# The wave plan

def split_item(item: dict) -> tuple[dict, dict]:
    """Split a planned item into the round configuration and the extra keys, which round.schema.json does not allow."""
    round_config = {key: value for key, value in item.items() if key not in EXTRA_KEYS}
    return round_config, {key: item.get(key) for key in EXTRA_KEYS}


def config_key(round_config: dict) -> tuple:
    """The six settings of a configuration, with the numbers rounded and the excluded layers as a set."""
    return (round(float(round_config["sparsity"]), 6), int(round_config["steps"]),
            int(round_config["finetune_epochs"]), round(float(round_config["lr"]), 6),
            round(float(round_config["weight_decay"]), 6), frozenset(round_config["exclude_layers"]))


def earlier_keys(ctx) -> set:
    """The settings of each configuration that already ran or was submitted."""
    state = ctx.paths.state
    keys = set()
    for path in list((state / "configs").glob("round_*.json")) + list((state / "rounds").glob("round_*/config.json")):
        try:
            keys.add(config_key(records.read_json(path)))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def code_check(ctx, entries: list[dict], earlier: set, in_wave: set) -> None:
    """Check each entry in code: the schema, the extra rule, and no repeat. Reject the entries that fail."""
    for entry in entries:
        try:
            config.validate_round_config(dict(entry["config"]), ctx.schema)
        except ConfigError as error:
            entry.update(status="rejected", reasons=[f"code: {error}"])
            continue
        key = config_key(entry["config"])
        if key in earlier:
            entry.update(status="rejected", reasons=["code: it repeats an earlier configuration"])
        elif key in in_wave:
            entry.update(status="rejected", reasons=["code: it repeats another configuration of this wave"])
        else:
            in_wave.add(key)


def new_entries(items: list[dict], first_item: int, stage: str) -> list[dict]:
    entries = []
    for offset, item in enumerate(items):
        round_config, extras = split_item(item)
        entries.append({"item": first_item + offset, "stage": stage, "config": round_config, "extras": extras,
                        "status": "proposed", "reasons": []})
    return entries


def describe_config(round_config: dict) -> str:
    excluded = ", ".join(round_config["exclude_layers"]) or "none"
    return (f"sparsity {round_config['sparsity']}, steps {round_config['steps']}, "
            f"epochs {round_config['finetune_epochs']}, lr {round_config['lr']}, "
            f"weight decay {round_config['weight_decay']}, excluded {excluded}")


def write_proposed(path: Path, entries: list[dict]) -> None:
    records.write_json(path, [{"item": entry["item"], **entry["config"], **entry["extras"]} for entry in entries])


def prompt_values(ctx, wave: int, count: int) -> dict:
    state = ctx.paths.state
    history = records.read_history(state)
    target = ctx.experiment["target"]
    return {
        "W": wave,
        "K": count,
        "R": ctx.budget["max_round_attempts"] - len(history),
        "G": round(max(0.0, ctx.budget["gpu_minutes"] - ctx.ledger.gpu_minutes)),
        "BEST": describe_best(finish.best_round(history, target)),
        "STALE": finish.rounds_without_improvement(history, target),
        "PLATEAU": target["plateau_rounds"],
        "STATE": rel(ctx, state),
        "EXPERIMENT": config.experiment_path(ctx.paths.project).relative_to(ctx.paths.project),
    }


def review(ctx, wave: int, kind: str, entries: list[dict], items_file: Path) -> str | None:
    """Run a critic step on the entries that passed the code checks, and set their status.
    Return the critic's summary, or None if the critic step failed."""
    if not entries:
        return ""
    write_proposed(items_file, entries)
    values = {"W": wave, "PROPOSED": rel(ctx, items_file), "STATE": rel(ctx, ctx.paths.state),
              "EXPERIMENT": config.experiment_path(ctx.paths.project).relative_to(ctx.paths.project)}
    result = run_role(ctx, kind, render(ctx.paths.project / "prompts" / "multi" / "critic.md", values),
                      ctx.schemas["review"], wave)
    output = structured(result, json.loads(ctx.schemas["review"])) if result else None
    if output is None:
        if result is not None:
            ctx.log(f"wave {wave}: the {kind} step returned no valid verdicts")
        return None
    verdicts = {verdict["item"]: verdict for verdict in output["verdicts"]}
    for entry in entries:
        verdict = verdicts.get(entry["item"])
        if verdict is None:
            entry.update(status="rejected", reasons=["failed: the critic gave no verdict"])
        elif verdict["accept"]:
            entry.update(status="accepted", verdict=verdict["reason"])
        else:
            entry.update(status="rejected", reasons=[f"critic: {verdict['reason']}"])
    return output["summary"]


def accept_or_review(ctx, wave: int, kind: str, entries: list[dict], items_file: Path, reviews: list[str]) -> bool:
    """Send the entries that passed the code checks to the critic, or accept them if the critic is off.
    Return False if the critic step failed."""
    passed = [entry for entry in entries if entry["status"] == "proposed"]
    if not critic_on(ctx):
        for entry in passed:
            entry["status"] = "accepted"
        return True
    summary = review(ctx, wave, kind, passed, items_file)
    if summary is None:
        return False
    if summary:
        reviews.append(summary)
    return True


def plan_wave(ctx, wave: int, count: int) -> list[dict] | None:
    """Make the plan of a wave: planner step, code checks, critic step, and at most one revision.

    Return the accepted entries, or None if the planner step or the first critic step failed.
    """
    folder = wave_dir(ctx, wave)
    folder.mkdir(parents=True, exist_ok=True)
    project = ctx.paths.project
    values = prompt_values(ctx, wave, count)
    schema = wave_schema(ctx, count)
    result = run_role(ctx, "planner", render(project / "prompts" / "multi" / "planner.md", values),
                      json.dumps(schema), wave)
    plan = structured(result, schema) if result else None
    if plan is None:
        ctx.log(f"wave {wave}: the planner returned no valid plan")
        return None

    entries = new_entries(plan["configurations"], 1, "plan")
    write_proposed(folder / "proposed.json", entries)
    earlier, in_wave, reviews = earlier_keys(ctx), set(), []
    code_check(ctx, entries, earlier, in_wave)
    if not accept_or_review(ctx, wave, "critic", entries, folder / "review_items.json", reviews):
        write_plan(ctx, wave, plan["summary"], entries, reviews, failed="critic")
        return None

    rejected = [entry for entry in entries if entry["status"] == "rejected"]
    if rejected:
        lines = [f"- Item {entry['item']}: {describe_config(entry['config'])}. Rejected: {'; '.join(entry['reasons'])}"
                 for entry in rejected]
        values.update(K=len(rejected), REJECTED="\n".join(lines), PROPOSED=rel(ctx, folder / "proposed.json"))
        schema = wave_schema(ctx, len(rejected))
        result = run_role(ctx, "revision", render(project / "prompts" / "multi" / "revision.md", values),
                          json.dumps(schema), wave)
        revision = structured(result, schema) if result else None
        if revision is None:
            for entry in rejected:
                entry["status"] = "dropped"
        else:
            replacements = new_entries(revision["configurations"], len(entries) + 1, "revision")
            write_proposed(folder / "proposed_2.json", replacements)
            code_check(ctx, replacements, earlier, in_wave)
            if not accept_or_review(ctx, wave, "critic_2", replacements, folder / "review_items_2.json", reviews):
                for entry in replacements:
                    if entry["status"] == "proposed":
                        entry.update(status="dropped", reasons=["failed: the second critic step failed"])
            for entry in replacements:
                if entry["status"] == "rejected":
                    entry["status"] = "dropped"
            for entry in rejected:
                entry["status"] = "replaced"
            entries += replacements
    accepted = [entry for entry in entries if entry["status"] == "accepted"]
    write_plan(ctx, wave, plan["summary"], entries, reviews)
    records.write_json(folder / "accepted.json", accepted)
    return accepted


def write_plan(ctx, wave: int, summary: str, entries: list[dict], reviews: list[str],
               failed: str | None = None) -> None:
    """Write plan.json for the records and plan.md for you."""
    folder = wave_dir(ctx, wave)
    records.write_json(folder / "plan.json", {"wave": wave, "summary": summary, "reviews": reviews,
                                              "failed_step": failed, "critic": critic_on(ctx), "entries": entries})
    lines = [f"# Wave {wave}", "", f"Planner: {summary}", ""]
    lines += [f"Critic: {text}" for text in reviews] + ([""] if reviews else [])
    lines += ["| Item | Stage | Sparsity | Steps | Epochs | LR | Weight decay | Excluded | Predicted val | "
              "Pass probability | Status |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for entry in entries:
        cfg, extras = entry["config"], entry["extras"]
        lines.append(f"| {entry['item']} | {entry['stage']} | {cfg['sparsity']} | {cfg['steps']} | "
                     f"{cfg['finetune_epochs']} | {cfg['lr']} | {cfg['weight_decay']} | "
                     f"{', '.join(cfg['exclude_layers']) or 'none'} | {extras['predicted_val_accuracy']} | "
                     f"{extras['pass_probability']} | {entry['status']} |")
    for entry in entries:
        lines += ["", f"## Item {entry['item']} ({entry['status']})", "",
                  f"Hypothesis: {entry['extras']['hypothesis']}", "", f"Reason: {entry['config']['reason']}"]
        if entry.get("verdict"):
            lines += ["", f"Critic: {entry['verdict']}"]
        for reason in entry["reasons"]:
            source, _, text = reason.partition(":")
            lines += ["", f"Rejected ({source}): {text.strip()}"]
    if failed:
        lines += ["", f"The {failed} step failed, so the wave has no accepted configuration."]
    lines += ["", "To run this plan, start the loop job: sbatch slurm/multi_loop.sbatch",
              f"To plan again, remove {rel(ctx, folder)} and run the --plan-only job again.", ""]
    (folder / "plan.md").write_text("\n".join(lines))


# Rounds and records

def next_round_number(ctx) -> int:
    """A round number above each round in history.tsv, each round folder, and each submitted task."""
    state = ctx.paths.state
    numbers = [int(row["round"]) for row in records.read_history(state)]
    numbers += [int(folder.name.split("_")[1]) for folder in (state / "rounds").glob("round_*")
                if folder.name.split("_")[1].isdigit()]
    for tasks in (state / "waves").glob("wave_*/tasks.tsv"):
        numbers += [int(line.split("\t")[1]) for line in tasks.read_text().splitlines() if line.count("\t") >= 3]
    return max(numbers, default=0) + 1


def history_rounds(state: Path) -> set[int]:
    return {int(row["round"]) for row in records.read_history(state)}


def ok_row(state: Path, round_no: int) -> dict:
    """The history row of a complete round, with the job ID of its round job."""
    record = records.record_dir(state, "round", round_no)
    return records.history_row(round_no, "ok", records.read_json(record / "config.json"),
                               records.read_json(record / "metrics.json"),
                               job=records.read_json(record / "run_meta.json")["job_id"])


def write_missing_rows(ctx) -> int:
    """Write the history row of each submitted round that has metrics but no row. A stop between the end
    of a task and its row then loses no round."""
    state = ctx.paths.state
    have = history_rounds(state)
    added = 0
    for tasks in sorted((state / "waves").glob("wave_*/tasks.tsv")):
        for line in tasks.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 4 or parts[2] != "round":
                continue
            round_no = int(parts[1])
            if round_no in have or not (records.record_dir(state, "round", round_no) / "metrics.json").exists():
                continue
            records.append_history(state, ok_row(state, round_no))
            have.add(round_no)
            added += 1
    if added:
        ctx.log(f"wrote {added} missing history row(s)")
    return added


def write_outcome(ctx, wave: int, complete: int, note: str, **more) -> None:
    records.write_json(wave_dir(ctx, wave) / "outcome.json", {"complete_rounds": complete, "note": note, **more})


def add_round_records(ctx, wave: int, rounds: list[tuple[int, dict]], job: str, outcome: str) -> int:
    """Write the ledger rows and the history rows of the rounds of an array job, in round order.
    Return the number of complete rounds."""
    state = ctx.paths.state
    usage = ctx.slurm.array_accounting(job, len(rounds))
    have = history_rounds(state)
    complete = 0
    for index, (round_no, round_config) in enumerate(rounds):
        task = usage.get(index)
        ctx.ledger.add(round=round_no, kind="round", status=outcome, wave=wave,
                       job_id=task.job_id if task else f"{job}_{index}",
                       # without accounting, count the time limit, so that the GPU budget stays safe
                       gpu_minutes=task.gpu_minutes if task else ctx.experiment["round"]["time_minutes"],
                       slurm_state=task.state if task else "UNKNOWN")
        if (records.record_dir(state, "round", round_no) / "metrics.json").exists():
            complete += 1
            if round_no not in have:
                records.append_history(state, ok_row(state, round_no))
            continue
        if round_no in have:
            continue
        never_ran = task is None or task.runs == 0
        status = "queue_limit" if outcome == "queue_limit" and never_ran else "job_failed"
        note = f"task {index} of array job {job} ended as {task.state if task else 'UNKNOWN'} without metrics"
        records.append_history(state, records.history_row(round_no, status, round_config, None, note,
                                                          job=task.job_id if task else job))
    return complete


def close_interrupted_wave(ctx, wave: int) -> None:
    """Finish the records of a wave whose loop job stopped before the wave ended."""
    folder = wave_dir(ctx, wave)
    rounds = []
    for line in (folder / "tasks.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            rounds.append((int(parts[1]), records.read_json(ctx.paths.project / parts[0])))
    job = (folder / "array_job").read_text().strip() if (folder / "array_job").exists() else None
    complete = add_round_records(ctx, wave, rounds, job, "interrupted") if job else 0
    write_outcome(ctx, wave, complete, "the loop job stopped during the wave", array_job=job)
    ctx.log(f"wave {wave}: closed after an interruption, {complete} complete round(s)")


def write_notes(ctx, wave: int, entries: list[dict]) -> None:
    lines = [f"## Wave {wave}", ""]
    for entry in entries:
        if entry.get("round"):
            extras = entry["extras"]
            lines += [f"### Round {entry['round']} (item {entry['item']})", "",
                      f"Hypothesis: {extras['hypothesis']}",
                      f"Predicted validation accuracy {extras['predicted_val_accuracy']}, "
                      f"pass probability {extras['pass_probability']}.", "", entry["config"]["reason"], ""]
        elif entry["reasons"]:
            lines += [f"### Item {entry['item']}, not run ({entry['status']})", "", "; ".join(entry["reasons"]), ""]
    with open(ctx.paths.state / "notes.md", "a") as notes:
        notes.write("\n".join(lines) + "\n")


def submit_wave(ctx, wave: int, accepted: list[dict]) -> int:
    """Submit the accepted configurations of a wave as one array job, wait, and write the records.
    Return the number of complete rounds."""
    state, project, folder = ctx.paths.state, ctx.paths.project, wave_dir(ctx, wave)
    accepted = accepted[:wave_size(ctx, records.read_history(state))]  # a limit can have changed since the plan
    first = next_round_number(ctx)
    lines, rounds = [], []
    for offset, entry in enumerate(accepted):
        entry["round"] = first + offset
        path = state / "configs" / f"round_{entry['round']:02d}.json"
        records.write_json(path, entry["config"])
        lines.append(f"{rel(ctx, path)}\t{entry['round']}\tround\t0")
        rounds.append((entry["round"], entry["config"]))
    (folder / "tasks.tsv").write_text("\n".join(lines) + "\n")
    records.write_json(folder / "submitted.json", [{"item": entry["item"], "round": entry["round"]} for entry in accepted])
    plan = records.read_json(folder / "plan.json")
    submitted = {entry["item"]: entry["round"] for entry in accepted}
    write_notes(ctx, wave, [{**entry, "round": submitted.get(entry["item"])} for entry in plan["entries"]])
    try:
        job = ctx.slurm.submit_array("slurm/round_array.sbatch", str(rel(ctx, folder / "tasks.tsv")), len(lines),
                                     ctx.experiment["wave"]["max_parallel"], project)
    except (SlurmError, subprocess.TimeoutExpired) as error:
        for round_no, round_config in rounds:
            records.append_history(state, records.history_row(round_no, "sbatch_failed", round_config, None,
                                                              str(error)[:200]))
        write_outcome(ctx, wave, 0, f"sbatch failed: {error}"[:300])
        write_hypotheses(ctx)
        ctx.log(f"wave {wave}: sbatch failed: {error}")
        return 0
    (folder / "array_job").write_text(job + "\n")
    (state / "current_job").write_text(job + "\n")
    ctx.log(f"wave {wave}: submitted array job {job}, rounds {first} to {first + len(lines) - 1}")
    outcome = ctx.slurm.wait_array(job, 60 * ctx.budget["queue_limit_minutes"])
    complete = add_round_records(ctx, wave, rounds, job, outcome)
    (state / "current_job").unlink(missing_ok=True)
    write_outcome(ctx, wave, complete, outcome, array_job=job)
    write_hypotheses(ctx)
    log_wave(ctx, wave, [round_no for round_no, _ in rounds])
    if outcome == "queue_limit":
        raise StopLoop("queue_limit")
    return complete


def log_wave(ctx, wave: int, round_numbers: list[int]) -> None:
    rows = {int(row["round"]): row for row in records.read_history(ctx.paths.state)}
    parts = [f"round {n}: val_accuracy={rows[n]['val_accuracy'] or '-'} weights_left={rows[n]['weights_left'] or '-'}"
             for n in round_numbers if n in rows]
    by_kind = {kind: int(sum(float(row["tokens"] or 0) for row in ctx.ledger.rows
                             if row["kind"] == kind and row.get("wave") == str(wave))) for kind in WAVE_KINDS}
    tokens = ", ".join(f"{kind} {count}" for kind, count in by_kind.items() if count)
    ctx.log(f"wave {wave}: {'; '.join(parts)}; tokens {tokens} "
            f"(totals: {ctx.ledger.tokens} tokens, {ctx.ledger.gpu_minutes:.1f} GPU minutes)")


# Hypotheses and prediction scores

def write_hypotheses(ctx) -> list[dict]:
    """Rebuild hypotheses.tsv from the plans of the submitted waves and the round results."""
    state, target = ctx.paths.state, ctx.experiment["target"]
    rows = []
    for wave in wave_numbers(ctx):
        folder = wave_dir(ctx, wave)
        if not (folder / "submitted.json").exists():
            continue
        submitted = {item["item"]: item["round"] for item in records.read_json(folder / "submitted.json")}
        for entry in records.read_json(folder / "plan.json")["entries"]:
            round_no = submitted.get(entry["item"])
            metrics_file = records.record_dir(state, "round", round_no) / "metrics.json" if round_no else None
            result = records.read_json(metrics_file) if metrics_file and metrics_file.exists() else None
            predicted = entry["extras"]["predicted_val_accuracy"]
            row = {"wave": wave, "item": entry["item"], "round": round_no or "",
                   "status": "run" if round_no else entry["status"], "verdict": entry.get("verdict", ""),
                   "hypothesis": " ".join(str(entry["extras"]["hypothesis"]).split()),
                   "predicted_val_accuracy": predicted, "pass_probability": entry["extras"]["pass_probability"],
                   "val_accuracy": "", "passes": "", "error": ""}
            if result:
                row.update(val_accuracy=f"{result['val_accuracy']:.4f}",
                           passes="yes" if result["val_accuracy"] >= target["val_accuracy"] else "no",
                           error=f"{result['val_accuracy'] - float(predicted):+.4f}")
            rows.append(row)
    path = state / "hypotheses.tsv"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HYPOTHESIS_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return rows


def prediction_scores(ctx, rows: list[dict]) -> dict:
    """Score the planner's predictions next to simple baselines.

    Mean absolute error: next to the accuracy of the earlier round whose sparsity is nearest.
    Brier score of the pass probabilities: next to always forecasting the pass rate of this run.
    Noise in 5,000 validation images sets a floor of about 0.004 for the mean absolute error.
    """
    state = ctx.paths.state
    done = [row for row in rows if row["status"] == "run" and row["val_accuracy"] != ""]
    if not done:
        return {"rounds": 0}
    history = [row for row in records.read_history(state) if row["status"] == "ok"]
    errors, baseline_errors, brier, right = [], [], [], 0
    passes = [row["passes"] == "yes" for row in done]
    rate = sum(passes) / len(passes)
    for row, passed in zip(done, passes):
        actual, predicted = float(row["val_accuracy"]), float(row["predicted_val_accuracy"])
        errors.append(abs(actual - predicted))
        probability = float(row["pass_probability"])
        brier.append((probability - passed) ** 2)
        right += (probability >= 0.5) == passed
        sparsity = float(records.read_json(records.record_dir(state, "round", int(row["round"])) / "config.json")["sparsity"])
        earlier = [item for item in history if int(item["round"]) < int(row["round"])
                   and not same_wave(rows, item["round"], row["wave"])]
        if earlier:
            nearest = min(earlier, key=lambda item: abs(math.log(1 - float(item["sparsity"])) - math.log(1 - sparsity)))
            baseline_errors.append(abs(actual - float(nearest["val_accuracy"])))
    return {
        "rounds": len(done),
        "mean_absolute_error": round(sum(errors) / len(errors), 4),
        "baseline_mean_absolute_error": round(sum(baseline_errors) / len(baseline_errors), 4) if baseline_errors else None,
        "brier_score": round(sum(brier) / len(brier), 4),
        "base_rate_brier_score": round(rate * (1 - rate), 4),
        "pass_decisions_right": right,
    }


def same_wave(rows: list[dict], round_no: str, wave) -> bool:
    """Rounds of one wave ran at the same time, so their results were not known when the wave was planned."""
    return any(str(row["round"]) == str(round_no) and str(row["wave"]) == str(wave) for row in rows)


# Final steps

def final_role_step(ctx, kind: str, prompt: str, schema_text: str | None, failures: dict) -> dict | None:
    """Run the writer step or the auditor step. A failure, or a step that used another model, is recorded
    in failures and does not stop the code checks that follow."""
    try:
        result = run_role(ctx, kind, prompt, schema_text)
    except (AgentFormatError, ModelCheckError) as error:
        failures[kind] = str(error)
        ctx.log(f"{kind} step: {error}")
        return None
    if result is None:
        failures[kind] = "the step failed"
    return result


def report_steps(ctx) -> None:
    """The writer step and the auditor step, then the code checks. The role steps run if the total is below
    the token budget; the search limit keeps the final reserve free for them."""
    write_hypotheses(ctx)
    state = ctx.paths.state
    report = state / "final_report.md"
    failures = {}
    if ctx.ledger.tokens >= ctx.budget["tokens"]:
        ctx.log("writer and auditor skipped: the token budget is used up")
        failures = {kind: "skipped: the token budget is used up" for kind in ("writer", "auditor")}
    else:
        prompt = render(ctx.paths.project / "prompts" / "multi" / "writer.md", {"STATE": rel(ctx, state)})
        result = final_role_step(ctx, "writer", prompt, None, failures)
        if result is not None:
            report.write_text(str(result.get("result", "")).strip() + "\n")
    run_audit(ctx, report, failures)


def run_audit(ctx, report: Path, failures: dict | None = None) -> dict:
    """Run the auditor step, check_report, and the check_round checks of the selected round. Write audit.json.
    The code checks run also when a role step failed or was skipped."""
    state = ctx.paths.state
    failures = dict(failures or {})
    schema = json.loads(ctx.schemas["audit"])
    auditor = None
    if not report.exists():
        failures.setdefault("auditor", "skipped: there is no report")
    elif "auditor" not in failures:
        prompt = render(ctx.paths.project / "prompts" / "multi" / "auditor.md",
                        {"STATE": rel(ctx, state), "REPORT": rel(ctx, report)})
        result = final_role_step(ctx, "auditor", prompt, ctx.schemas["audit"], failures)
        auditor = structured(result, schema)
        if auditor is None and "auditor" not in failures:  # for example error_max_turns: no structured output
            failures["auditor"] = f"no valid audit (subtype {result.get('subtype')})"
    audit = {
        "report": str(rel(ctx, report)),
        "auditor": auditor,
        "auditor_step_failed": auditor is None,
        "failures": failures,
        "unmatched_numbers": check_report.unmatched_numbers(report.read_text(), state) if report.exists() else None,
        "check_round": selected_round_checks(ctx),
    }
    records.write_json(state / "audit.json", audit)
    return audit


def selected_round_checks(ctx) -> dict | None:
    """Run the checks of check_round on the selected round."""
    from . import check_round  # it loads PyTorch, so import it only here

    selection = ctx.paths.state / finish.SELECTION_FILE
    round_no = records.read_json(selection)["round"] if selection.exists() else None
    if round_no is None:
        return None
    try:
        checks = check_round.run_checks(round_no)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        return {"round": round_no, "ok": False, "error": str(error)[:300]}
    return {"round": round_no, "ok": all(ok for _, ok, _ in checks),
            "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in checks]}


def audit_summary(ctx) -> dict | None:
    path = ctx.paths.state / "audit.json"
    if not path.exists():
        return None
    audit = records.read_json(path)
    claims = (audit.get("auditor") or {}).get("claims", [])
    return {
        "claims": len(claims),
        "wrong": sum(claim["verdict"] == "wrong" for claim in claims),
        "unsupported": sum(claim["verdict"] == "unsupported" for claim in claims),
        "limits_stated": (audit.get("auditor") or {}).get("limits_stated"),
        "auditor_step_failed": audit.get("auditor_step_failed"),
        "failures": audit.get("failures", {}),
        "unmatched_numbers": None if audit.get("unmatched_numbers") is None else len(audit["unmatched_numbers"]),
        "check_round_ok": (audit.get("check_round") or {}).get("ok"),
    }


def summary_extra(ctx) -> dict:
    """The fields of the summary that only the multi-agent loop has."""
    role_info = {}
    for role in ROLES:
        kinds = {kind for kind, owner in ROLE_OF.items() if owner == role}
        settings = roles(ctx)[role]
        role_info[role] = {
            "model": settings["model"], "effort": settings["effort"],
            "enabled": critic_on(ctx) if role == "critic" else True,
            "steps": sum(row["kind"] in kinds for row in ctx.ledger.rows),
            "failed_steps": sum(row["kind"] in kinds and row["status"] == "failed" for row in ctx.ledger.rows),
            "tokens": int(ctx.ledger.total("tokens", kinds)),
            "estimated_cost_usd": round(ctx.ledger.total("cost_usd", kinds), 2),
        }
    # Each planned configuration counts one time. "not_reviewed": a critic step failed before its verdict.
    # "not_revised": rejected, but the revision did not run.
    counts = {"planned": 0, "accepted": 0, "replaced": 0, "dropped": 0, "not_reviewed": 0, "not_revised": 0,
              "rejected_by_code": 0, "rejected_by_critic": 0}
    final_status = {"accepted": "accepted", "replaced": "replaced", "dropped": "dropped",
                    "proposed": "not_reviewed", "rejected": "not_revised"}
    waves = []
    for wave in wave_numbers(ctx):
        folder = wave_dir(ctx, wave)
        if (folder / "plan.json").exists():
            for entry in records.read_json(folder / "plan.json")["entries"]:
                counts["planned"] += 1
                counts[final_status[entry["status"]]] += 1
                counts["rejected_by_code"] += any(reason.startswith("code:") for reason in entry["reasons"])
                counts["rejected_by_critic"] += any(reason.startswith("critic:") for reason in entry["reasons"])
        if (folder / "outcome.json").exists():
            waves.append({"wave": wave, "tokens": wave_tokens(ctx, wave), **records.read_json(folder / "outcome.json")})
    rows = write_hypotheses(ctx) if (ctx.paths.state / "waves").exists() else []
    return {
        "design": "planner and critic, waves" if critic_on(ctx) else "one agent, waves",
        "roles": role_info,
        "waves": waves,
        "configurations": counts,
        "predictions": prediction_scores(ctx, rows),
        "audit": audit_summary(ctx),
    }


# Start, plan-only, audit-only, and the loop

def copy_round_one(ctx) -> None:
    """Copy round 1 of chapter 28 from state/ to the state folder of this loop, one time."""
    state, source = ctx.paths.state, ctx.paths.project / "state"
    target = records.record_dir(state, "round", 1)
    if state == source or (target / "metrics.json").exists():
        return
    origin = records.record_dir(source, "round", 1)
    if not (origin / "metrics.json").exists():
        return
    for name in ("config.json", "run_meta.json", "metrics.json"):
        records.write_json(target / name, records.read_json(origin / name))
    rows = [row for row in records.read_history(source) if row["round"] == "1"]
    if rows and 1 not in history_rounds(state):
        records.append_history(state, rows[0])
    notes = state / "notes.md"
    if not notes.exists():
        notes.write_text("# Notes\n\nThe loop script adds the reasons of the planner and the critic here.\n\n"
                         f"## Round 1\n\n{records.read_json(target / 'config.json').get('reason', '')}\n\n")
    write_hypotheses(ctx)
    ctx.log(f"copied round 1 from state/ to {rel(ctx, state)}/")


def check_start(ctx, plan_only: bool = False) -> None:
    """Refuse to start without the baseline, round 1, and a planned wave 1, or with settings that do not agree."""
    project = ctx.paths.project
    if not (project / "state" / "baseline" / "metrics.json").exists():
        raise SystemExit("There is no baseline. Run 'sbatch slurm/baseline.sbatch' first.")
    copy_round_one(ctx)
    if not (records.record_dir(ctx.paths.state, "round", 1) / "metrics.json").exists():
        raise SystemExit("There is no round 1. Run the supervised round of the single-agent example first.")
    check_time_lines(ctx, ROUND_SCRIPTS)
    check_roles(ctx.experiment)
    records.check_integrity(project)
    accepted = wave_dir(ctx, 1) / "accepted.json"
    if not plan_only and not (accepted.exists() and records.read_json(accepted)):
        raise SystemExit("Wave 1 has no accepted configuration. Run 'sbatch --time=01:30:00 slurm/multi_loop.sbatch "
                         "--plan-only', and read the plan first.")
    clean_up_after_a_stop(ctx)


def clean_up_after_a_stop(ctx) -> None:
    """Cancel the job of an earlier loop job that stopped without a cancel, close its open wave,
    and write the missing history rows."""
    leftover = ctx.paths.state / "current_job"
    if leftover.exists():
        job = leftover.read_text().strip()
        ctx.slurm.cancel(job, wait_seconds=15)
        leftover.unlink(missing_ok=True)
        ctx.log(f"canceled the job {job} of an earlier loop job")
    for wave in wave_numbers(ctx):
        folder = wave_dir(ctx, wave)
        if (folder / "tasks.tsv").exists() and not (folder / "outcome.json").exists():
            close_interrupted_wave(ctx, wave)
    write_missing_rows(ctx)


def plan_first_wave(ctx) -> int:
    """Plan wave 1 and stop, so that you can read the plan before any GPU job runs."""
    check_start(ctx, plan_only=True)
    folder = wave_dir(ctx, 1)
    if (folder / "accepted.json").exists():
        if not records.read_json(folder / "accepted.json"):
            ctx.log(f"Wave 1 has no accepted configuration. Remove {rel(ctx, folder)}, and plan again.")
            return 1
        ctx.log(f"Wave 1 already has a plan: read {rel(ctx, folder / 'plan.md')}. "
                f"To plan again, remove {rel(ctx, folder)}.")
        return 0
    reason = stop_reason(ctx)
    if reason:
        ctx.log(f"cannot plan wave 1: {reason}")
        return 1
    try:
        accepted = plan_wave(ctx, 1, wave_size(ctx, records.read_history(ctx.paths.state)))
    except StopLoop as stop:
        ctx.log(f"plan of wave 1 stopped: {stop.reason}")
        return 1
    except (AgentFormatError, ModelCheckError) as error:
        ctx.log(f"plan of wave 1 stopped: {error}")
        return 1
    if not accepted:
        ctx.log(f"Wave 1 has no accepted configuration. Read {rel(ctx, folder / 'plan.md')}, "
                f"remove {rel(ctx, folder)}, and plan again.")
        return 1
    ctx.log(f"wave 1 planned: {len(accepted)} configuration(s) accepted. Read {rel(ctx, folder / 'plan.md')}, "
            "then start the loop job with 'sbatch slurm/multi_loop.sbatch'.")
    return 0


def audit_only(ctx) -> int:
    """Run the auditor step and check_report on the report of the state folder, for example of the single-agent run."""
    check_roles(ctx.experiment)
    report = ctx.paths.state / "final_report.md"
    if not report.exists():
        raise SystemExit(f"There is no report: {rel(ctx, report)}")
    try:
        audit = run_audit(ctx, report)
    except (AgentFormatError, ModelCheckError) as error:
        ctx.log(f"audit stopped: {error}")
        return 1
    ctx.log(f"audit written to {rel(ctx, ctx.paths.state / 'audit.json')}: "
            f"{len(audit['unmatched_numbers'])} number(s) not found in the records")
    return 0


def next_wave(ctx) -> int:
    """The first wave with no outcome. A wave that the loop job left during its array job is closed first."""
    for wave in wave_numbers(ctx):
        folder = wave_dir(ctx, wave)
        if (folder / "outcome.json").exists():
            continue
        if (folder / "tasks.tsv").exists():
            close_interrupted_wave(ctx, wave)
            continue
        return wave
    numbers = wave_numbers(ctx)
    return numbers[-1] + 1 if numbers else 1


def run_waves(ctx) -> str:
    """Plan and run waves until a stop rule applies. Return the stop reason."""
    while True:
        wave = next_wave(ctx)
        folder = wave_dir(ctx, wave)
        planned = (folder / "accepted.json").exists()
        reason = stop_reason(ctx, planned)
        if reason:
            return reason
        try:
            records.check_integrity(ctx.paths.project)
            if planned:
                accepted = records.read_json(folder / "accepted.json")
            else:
                accepted = plan_wave(ctx, wave, wave_size(ctx, records.read_history(ctx.paths.state)))
            if not accepted:
                write_outcome(ctx, wave, 0, "no accepted configuration" if accepted == [] else "a role step failed")
                continue
            if not time_for(ctx, array_minutes(ctx)):
                write_outcome(ctx, wave, 0, "not run: not enough time for the array job and the final steps")
                return "time_left"
            submit_wave(ctx, wave, accepted)
        except StopLoop as stop:
            return stop.reason
        except IntegrityError as error:
            ctx.log(str(error))
            return "integrity"
        except AgentFormatError as error:
            ctx.log(f"{error}. Check the Claude Code version.")
            return "agent_format"
        except ModelCheckError as error:
            ctx.log(f"{error}. Check which models your organization allows.")
            return "model_check"


def run(ctx, mode: str = "loop") -> int:
    """Run the waves and the final steps. On SIGTERM, cancel the running array job and stop."""
    reason = None  # the stop reason of the waves, once they end
    ctx.log(f"experiment file: {config.experiment_path(ctx.paths.project).relative_to(ctx.paths.project)}, "
            f"state folder: {rel(ctx, ctx.paths.state)}, mode: {mode}")
    try:
        if mode == "plan_only":
            return plan_first_wave(ctx)
        if mode == "audit_only":
            return audit_only(ctx)
        if mode == "finish":
            clean_up_after_a_stop(ctx)
            reason = finish.previous_stop_reason(ctx) or "finish_only"
        else:
            check_start(ctx)
            reason = run_waves(ctx)
            ctx.log(f"stop: {reason}")
        write_missing_rows(ctx)
        finish.run_final_steps(ctx, reason, report_step=report_steps, extra=summary_extra)
        ctx.log(f"final steps done: see {ctx.paths.state.name}/summary.json, final_report.md, and audit.json")
        return 0
    except Terminated:
        # SLURM sends SIGKILL after KillWait (30 seconds on the cluster), so keep each step short.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        ctx.agent.stop()
        if mode in ("plan_only", "audit_only"):
            ctx.log("SIGTERM: stopped.")
            return 1
        # Cancel first: the records below read many files, and SIGKILL can come before they are done.
        current = ctx.paths.state / "current_job"
        if current.exists():
            job = current.read_text().strip()
            answered = ctx.slurm.cancel(job, wait_seconds=15)
            current.unlink(missing_ok=True)
            ctx.log(f"SIGTERM: asked SLURM to cancel array job {job}"
                    f"{'' if answered else ' (scancel did not answer in 15 seconds)'}.")
        write_missing_rows(ctx)
        # Keep the stop reason of the waves if they ended, so that --finish can use it.
        finish.write_summary(ctx, reason or "terminated", final=False, terminated=True, extra=summary_extra(ctx))
        ctx.log("SIGTERM: the loop script stopped. Run 'sbatch slurm/multi_loop.sbatch --finish' for the final steps.")
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the multi-agent loop, then the final steps.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--plan-only", action="store_true", help="plan wave 1, then stop")
    mode_group.add_argument("--finish", action="store_true", help="run only the final steps")
    mode_group.add_argument("--audit-only", action="store_true", help="audit the report of the state folder")
    args = parser.parse_args(argv)
    mode = "plan_only" if args.plan_only else "finish" if args.finish else "audit_only" if args.audit_only else "loop"
    paths = config.paths_from_env()
    project = paths.project
    # An audit of another state folder must not add rows to its ledger, which can have other columns.
    ledger_file = "audit_ledger.tsv" if mode == "audit_only" else "ledger.tsv"
    ctx = MultiContext(
        paths=paths,
        experiment=config.load_experiment(project),
        schema_text=(project / config.SCHEMA_PATH).read_text(),
        agent=Agent(project),
        slurm=Slurm(),
        ledger=Ledger(paths.state / ledger_file, MULTI_LEDGER_FIELDS, ROLE_OF.keys(), WAVE_KINDS),
        job_end=job_end_time(),
        schemas={name: (project / path).read_text() for name, path in SCHEMA_FILES.items()},
        plan_only=mode == "plan_only",
    )
    signal.signal(signal.SIGTERM, on_sigterm)
    return run(ctx, mode)


if __name__ == "__main__":
    raise SystemExit(main())
