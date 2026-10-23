"""Tests of the multi-agent loop with fake roles and fake SLURM array jobs. No API calls and no jobs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest
import yaml

from pruning_example import check_report, config, finish, metrics, multi_loop, records
from pruning_example.agent import AgentError
from pruning_example.loop import Ledger, Terminated
from pruning_example.slurm import TaskAccounting
from conftest import REPO

SMALL = 223_287


def usage(model: str, tokens: int) -> dict:
    return {model: {"inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": tokens - 40,
                    "cacheCreationInputTokens": 10}}


def item(number: int, **changes) -> dict:
    planned = {"sparsity": round(0.9 + 0.001 * number, 4), "steps": 2, "finetune_epochs": 4, "lr": 0.01,
               "weight_decay": 0.0005, "exclude_layers": [], "reason": f"test {number}",
               "hypothesis": f"hypothesis {number}", "predicted_val_accuracy": 0.91, "pass_probability": 0.6}
    return {**planned, **changes}


class FakeRoles:
    """Answers each role step like a real role, and keeps the calls. No API calls."""

    def __init__(self, tokens: int = 40_000, reject=None, fail=(), plan=None, models=None):
        self.tokens = tokens
        self.reject = reject or (lambda kind, entry: None)  # a reason to reject, or None
        self.fail = set(fail)
        self.plan = plan
        self.models = models or {}  # kind -> the model that the step reports
        self.calls = []
        self.counter = 0

    def run(self, prompt, caps, schema_text=None):
        role = Path(caps["role_prompt"]).stem
        kind = role
        if role == "planner" and "rejected" in prompt:
            kind = "revision"
        if role == "critic" and "review_items_2" in prompt:
            kind = "critic_2"
        self.calls.append((kind, prompt, caps))
        if kind in self.fail:
            raise AgentError("fake failure")
        result = {"subtype": "success", "modelUsage": usage(self.models.get(kind, caps["model"]), self.tokens),
                  "total_cost_usd": 0.1, "result": ""}
        if role == "planner":
            count = json.loads(schema_text)["properties"]["configurations"]["minItems"]
            planned = self.plan(kind, count, self) if self.plan else [item(self.next()) for _ in range(count)]
            result["structured_output"] = {"configurations": planned, "summary": f"{kind} summary"}
        elif role == "critic":
            items = json.loads(Path(re.search(r"They are in (\S+?),", prompt).group(1)).read_text())
            verdicts = []
            for entry in items:
                reason = self.reject(kind, entry)
                verdicts.append({"item": entry["item"], "accept": reason is None, "reason": reason or "fine"})
            result["structured_output"] = {"verdicts": verdicts, "summary": f"{kind} summary"}
        elif role == "writer":
            result["result"] = "The selected round keeps 0.9100 validation accuracy. One number is wrong: 0.9555."
        elif role == "auditor":
            result["structured_output"] = {
                "claims": [{"claim": "0.9100 validation accuracy", "verdict": "supported", "evidence": "summary.json",
                            "correction": ""}],
                "limits_stated": False, "summary": "one claim checked"}
        return result

    def next(self) -> int:
        self.counter += 1
        return self.counter

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]

    def stop(self):
        pass


def write_task_records(state: Path, round_config: dict, round_no: int, kind: str, seed: int,
                       val_accuracy: float, nonzero: int, job_id: str) -> None:
    """Write the records that a real round job writes with --no-history."""
    result = {"round": round_no, "kind": kind, "seed": seed, "val_accuracy": val_accuracy,
              "val_interval": metrics.wilson_interval(val_accuracy, 5000), "nonzero": nonzero,
              "weights_left": nonzero / 11_164_352, "sparse_size_mb": 1.0, "passes": val_accuracy >= 0.905}
    folder = records.record_dir(state, kind, round_no, seed)
    records.write_json(folder / "config.json", round_config)
    records.write_json(folder / "run_meta.json", {"job_id": job_id, "run_folder": str(state / "no_run_folder")})
    records.write_json(folder / "metrics.json", result)


class FakeArraySlurm:
    """Runs no jobs. Each task of a submitted array job writes the records that a real task writes."""

    def __init__(self, state, results, gpu_minutes: float = 2.0, outcome: str = "ended", stop_on_wave=None):
        self.state = state
        self.results = results  # (round, kind, seed) -> (val_accuracy, nonzero), or None for a task without metrics
        self.gpu_minutes = gpu_minutes
        self.outcome = outcome
        self.stop_on_wave = stop_on_wave
        self.arrays = []
        self.canceled = []
        self.options = []

    def submit_array(self, script, tasks_file, count, max_parallel, cwd, options=()):
        self.options.append(options)
        lines = [line.split("\t") for line in (Path(cwd) / tasks_file).read_text().splitlines()]
        assert len(lines) == count and script == "slurm/round_array.sbatch"
        job = str(900 + len(self.arrays))
        self.arrays.append((job, lines, max_parallel))
        for index, (config_path, round_no, kind, seed) in enumerate(lines):
            if kind == "confirm":
                assert (self.state / finish.SELECTION_FILE).exists()  # the selection is saved before any confirmation
            outcome = self.results(int(round_no), kind, int(seed))
            if outcome is not None:
                write_task_records(self.state, records.read_json(Path(cwd) / config_path), int(round_no), kind,
                                   int(seed), *outcome, job_id=f"{job}{index}")
        return job

    def wait_array(self, job, queue_limit_seconds):
        if self.stop_on_wave is not None and len(self.arrays) == self.stop_on_wave:
            raise Terminated()
        return self.outcome

    def array_accounting(self, job, count):
        lines = dict((entry[0], entry[1]) for entry in self.arrays)[job]
        tasks = {}
        for index, (_, round_no, kind, seed) in enumerate(lines):
            ran = self.results(int(round_no), kind, int(seed)) is not None or self.outcome != "queue_limit"
            tasks[index] = TaskAccounting(index, f"{job}{index}", "COMPLETED" if ran else "CANCELLED",
                                          self.gpu_minutes if ran else 0.0, 1 if ran else 0)
        return tasks

    def cancel(self, job, wait_seconds=60):
        self.canceled.append(job)
        return True

    def cancel_waiting(self, job, wait_seconds=60):
        return True

    def rounds(self) -> list[int]:
        return [int(line[1]) for _, lines, _ in self.arrays for line in lines if line[2] == "round"]


@pytest.fixture
def multi(project, monkeypatch):
    """A project after the baseline and the supervised round 1 of chapter 28, set up for the multi-agent loop."""
    state = project / "state"
    records.write_json(state / "baseline" / "metrics.json", {"val_accuracy": 0.94})
    first = {"sparsity": 0.5, "steps": 1, "finetune_epochs": 2, "lr": 0.01, "weight_decay": 0.0005,
             "exclude_layers": [], "reason": "round 1"}
    write_task_records(state, first, 1, "round", 0, 0.93, 5_582_176, "round-one")
    records.append_history(state, records.history_row(1, "ok", first, records.read_json(
        state / "rounds" / "round_01" / "metrics.json"), job="round-one"))
    monkeypatch.setenv("STATE_DIR", "state_multi")
    monkeypatch.setenv("EXPERIMENT_FILE", "experiment_multi.yaml")
    monkeypatch.setattr(finish, "evaluate_on_test_set",
                        lambda ctx, kind, n, seed=None: {"kind": kind, "round": n, "test_accuracy": 0.9234})
    return project


def context(project, agent, slurm, critic: bool = True, plateau_rounds: int | None = None, **budget):
    paths = config.paths_from_env()
    experiment = config.load_experiment(project)
    experiment["budget"].update(budget)
    experiment["roles"]["critic"]["enabled"] = critic
    if plateau_rounds is not None:
        experiment["target"]["plateau_rounds"] = plateau_rounds
    schemas = {name: (project / path).read_text() for name, path in multi_loop.SCHEMA_FILES.items()}
    return multi_loop.MultiContext(
        paths=paths, experiment=experiment, schema_text=(project / config.SCHEMA_PATH).read_text(), agent=agent,
        slurm=slurm, ledger=Ledger(paths.state / "ledger.tsv", multi_loop.MULTI_LEDGER_FIELDS,
                                   multi_loop.ROLE_OF.keys(), multi_loop.WAVE_KINDS),
        log=lambda message: None, schemas=schemas)


def plan_then_run(project, agent, slurm, **settings):
    """Run the --plan-only job, then the loop job, as a person does."""
    planner = context(project, agent, slurm, **settings)
    planner.plan_only = True
    assert multi_loop.run(planner, "plan_only") == 0
    ctx = context(project, agent, slurm, **settings)
    return ctx, multi_loop.run(ctx, "loop")


def state_of(project) -> Path:
    return project / "state_multi"


# The wave flow

def test_plan_only_writes_the_plan_and_runs_no_job(multi):
    agent = FakeRoles()
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL))
    ctx = context(multi, agent, slurm)
    ctx.plan_only = True
    assert multi_loop.run(ctx, "plan_only") == 0
    folder = state_of(multi) / "waves" / "wave_01"
    assert len(records.read_json(folder / "accepted.json")) == 3
    assert "To run this plan, start the loop job" in (folder / "plan.md").read_text()
    assert not (folder / "tasks.tsv").exists()
    assert slurm.arrays == []
    assert agent.kinds() == ["planner", "critic"]
    assert (state_of(multi) / "rounds" / "round_01" / "metrics.json").exists()  # round 1 copied from state/
    assert [row["round"] for row in records.read_history(state_of(multi))] == ["1"]


def test_the_loop_refuses_to_start_without_a_planned_wave_1(multi):
    ctx = context(multi, FakeRoles(), FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL)))
    with pytest.raises(SystemExit, match="Wave 1 has no accepted configuration"):
        multi_loop.run(ctx, "loop")


def test_waves_run_as_array_jobs_and_the_loop_writes_the_rows_in_round_order(multi):
    agent = FakeRoles()
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n) if kind == "round" else (0.906, SMALL))
    ctx, code = plan_then_run(multi, agent, slurm, max_round_attempts=7)
    assert code == 0
    state = state_of(multi)
    history = records.read_history(state)
    assert [row["round"] for row in history] == [str(n) for n in range(1, 8)]
    assert [row["job_id"] for row in history[1:4]] == ["9000", "9001", "9002"]  # the task job IDs, not the loop job
    assert slurm.rounds() == [2, 3, 4, 5, 6, 7]
    assert [max_parallel for _, _, max_parallel in slurm.arrays] == [3, 3, 3]  # two waves, then the confirmations
    tasks = (state / "waves" / "wave_01" / "tasks.tsv").read_text().splitlines()
    assert tasks[0].split("\t")[1:] == ["2", "round", "0"]
    assert "hypothesis" not in records.read_json(state / "configs" / "round_02.json")
    assert agent.kinds()[:4] == ["planner", "critic", "planner", "critic"]  # wave 1 is not planned again
    summary = records.read_json(state / "summary.json")
    assert summary["stop_reason"] == "round_limit"
    assert summary["selected"]["round"] == 7
    assert [entry["seed"] for entry in summary["confirmations"]] == [1, 2]
    assert summary["means"]["holds"] is True
    assert summary["roles"]["planner"]["model"] == "claude-opus-5-5"
    assert summary["roles"]["critic"]["steps"] == 2
    assert summary["design"] == "planner and critic, waves"
    ledger = (state / "ledger.tsv").read_text().splitlines()
    assert ledger[0].split("\t") == multi_loop.MULTI_LEDGER_FIELDS
    with open(state / "hypotheses.tsv", newline="") as f:
        assert [row["status"] for row in csv.DictReader(f, delimiter="\t")] == ["run"] * 6
    assert "## Wave 2" in (state / "notes.md").read_text()
    assert (state / "final_report.md").exists()
    audit = records.read_json(state / "audit.json")
    assert audit["auditor"]["claims"][0]["verdict"] == "supported"
    assert audit["unmatched_numbers"] == [{"number": "0.9555", "line": 1}]  # 0.9100 is in the records


def test_code_checks_send_a_repeat_and_an_invalid_configuration_to_one_revision(multi):
    def plan(kind, count, fake):
        if kind == "planner":
            repeat = item(fake.next(), sparsity=0.5, steps=1, finetune_epochs=2)  # the configuration of round 1
            invalid = item(fake.next(), steps=5, finetune_epochs=4)  # fewer epochs than steps
            return [repeat, invalid, item(fake.next())][:count]
        return [item(fake.next()) for _ in range(count)]

    agent = FakeRoles(plan=plan)
    ctx = context(multi, agent, FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL)))
    multi_loop.copy_round_one(ctx)
    accepted = multi_loop.plan_wave(ctx, 1, 3)
    plan_record = records.read_json(state_of(multi) / "waves" / "wave_01" / "plan.json")
    assert [entry["status"] for entry in plan_record["entries"]] == ["replaced", "replaced", "accepted",
                                                                    "accepted", "accepted"]
    assert "repeats an earlier configuration" in plan_record["entries"][0]["reasons"][0]
    assert "at least steps" in plan_record["entries"][1]["reasons"][0]
    assert len(accepted) == 3
    assert agent.kinds() == ["planner", "critic", "revision", "critic_2"]
    revision_prompt = agent.calls[2][1]
    assert "The checks rejected 2 of the configurations" in revision_prompt


def test_a_configuration_rejected_twice_is_dropped(multi):
    agent = FakeRoles(reject=lambda kind, entry: "too far" if entry["item"] in (1, 4) else None)
    ctx = context(multi, agent, FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL)))
    accepted = multi_loop.plan_wave(ctx, 1, 3)
    statuses = [entry["status"] for entry in records.read_json(state_of(multi) / "waves" / "wave_01" / "plan.json")["entries"]]
    assert statuses == ["replaced", "accepted", "accepted", "dropped"]
    assert len(accepted) == 2


def test_the_critic_off_gives_one_agent_with_waves(multi):
    agent = FakeRoles()
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n))
    ctx, code = plan_then_run(multi, agent, slurm, critic=False, max_round_attempts=4)
    assert code == 0
    assert "critic" not in agent.kinds() and "critic_2" not in agent.kinds()
    assert records.read_json(state_of(multi) / "summary.json")["design"] == "one agent, waves"


def test_failed_planner_steps_are_failed_waves_and_two_stop_the_loop(multi):
    agent = FakeRoles(fail={"planner"})
    ctx = context(multi, agent, FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL)))
    multi_loop.copy_round_one(ctx)
    assert multi_loop.run_waves(ctx) == "failed_attempts"
    assert agent.kinds() == ["planner", "planner"]
    assert [row["kind"] for row in ctx.ledger.rows] == ["planner", "planner"]


def test_a_step_with_another_model_stops_the_loop(multi):
    agent = FakeRoles(models={"planner": "claude-sonnet-5-5"})  # an organization can replace a model
    ctx = context(multi, agent, FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL)))
    multi_loop.copy_round_one(ctx)
    assert multi_loop.run_waves(ctx) == "model_check"


def test_each_role_step_gets_its_model_effort_and_role_prompt(multi):
    agent = FakeRoles()
    ctx = context(multi, agent, FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL)))
    multi_loop.copy_round_one(ctx)
    multi_loop.plan_wave(ctx, 1, 3)
    planner_caps, critic_caps = agent.calls[0][2], agent.calls[1][2]
    assert (planner_caps["model"], planner_caps["effort"]) == ("claude-opus-5-5", "xhigh")
    assert (critic_caps["model"], critic_caps["effort"]) == ("claude-sonnet-5-5", "high")
    assert planner_caps["role_prompt"] == "prompts/multi/roles/planner.md"
    assert "enabled" not in critic_caps
    row = ctx.ledger.rows[0]
    assert (row["kind"], row["wave"], row["model"], row["effort"]) == ("planner", "1", "claude-opus-5-5", "xhigh")
    assert "GPU minutes left: 150" in agent.calls[0][1] and "Round attempts left: 44" in agent.calls[0][1]


# Limits

def test_wave_size_follows_all_four_limits(multi):
    ctx = context(multi, FakeRoles(), FakeArraySlurm(state_of(multi), lambda n, kind, seed: None), gpu_minutes=60)
    multi_loop.copy_round_one(ctx)
    history = records.read_history(ctx.paths.state)
    assert multi_loop.wave_size(ctx, history) == 3
    ctx.ledger.add(round=2, kind="round", gpu_minutes=45)
    assert multi_loop.wave_size(ctx, history) == 1  # room for one 10-minute round in 60 GPU minutes
    ctx.ledger.rows.pop()
    ctx.experiment["budget"]["max_round_attempts"] = 3
    assert multi_loop.wave_size(ctx, history) == 2  # 2 round attempts left
    ctx.experiment["budget"]["max_round_attempts"] = 45
    ctx.experiment["target"]["plateau_rounds"] = 2
    stale = {"round": "2", "status": "ok", "val_accuracy": "0.8900", "nonzero": "100"}
    assert multi_loop.wave_size(ctx, history + [stale]) == 1  # one round left before the plateau rule


def test_stop_rules_before_a_wave(multi):
    ctx = context(multi, FakeRoles(), FakeArraySlurm(state_of(multi), lambda n, kind, seed: None), gpu_minutes=60)
    multi_loop.copy_round_one(ctx)
    assert multi_loop.stop_reason(ctx) is None
    ctx.ledger.add(round="", kind="planner", tokens=700_000, wave=1)
    assert multi_loop.stop_reason(ctx) == "token_budget"  # 700,000 + 150,000 is above the search limit
    assert multi_loop.stop_reason(ctx, planned=True) is None  # a planned wave needs no tokens for its plan
    ctx.ledger.rows.clear()
    ctx.ledger.add(round=2, kind="round", gpu_minutes=55)
    assert multi_loop.stop_reason(ctx) == "gpu_budget"  # no room for a 10-minute round
    ctx.ledger.rows.clear()
    ctx.job_end = ctx.clock() + 60 * 120  # two hours left; a plan needs 70 minutes and the final steps 110
    assert multi_loop.stop_reason(ctx) == "time_left"
    ctx.plan_only = True
    assert multi_loop.stop_reason(ctx) is None  # the --plan-only job checks the time before each step instead


def test_plan_only_stops_without_time_for_a_step(multi):
    ctx = context(multi, FakeRoles(), FakeArraySlurm(state_of(multi), lambda n, kind, seed: None))
    ctx.plan_only = True
    ctx.job_end = ctx.clock() + 60 * 10  # 10 minutes left, the planner needs 20
    assert multi_loop.run(ctx, "plan_only") == 1


def test_queue_limit_stops_after_the_wave(multi):
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL) if n == 2 else None,
                           outcome="queue_limit")
    ctx = context(multi, FakeRoles(), slurm)
    multi_loop.copy_round_one(ctx)
    assert multi_loop.run_waves(ctx) == "queue_limit"
    statuses = [row["status"] for row in records.read_history(ctx.paths.state)]
    assert statuses == ["ok", "ok", "queue_limit", "queue_limit"]


# Stops and missing rows

def test_missing_rows_are_written_from_the_round_folders(multi):
    ctx = context(multi, FakeRoles(), FakeArraySlurm(state_of(multi), lambda n, kind, seed: None))
    multi_loop.copy_round_one(ctx)
    state = ctx.paths.state
    folder = state / "waves" / "wave_01"
    folder.mkdir(parents=True)
    (folder / "tasks.tsv").write_text("state_multi/configs/round_02.json\t2\tround\t0\n"
                                      "state_multi/configs/round_03.json\t3\tround\t0\n")
    write_task_records(state, item(1), 2, "round", 0, 0.91, SMALL, "task-2")
    assert multi_loop.write_missing_rows(ctx) == 1
    rows = records.read_history(state)
    assert [(row["round"], row["job_id"]) for row in rows] == [("1", "round-one"), ("2", "task-2")]
    assert multi_loop.write_missing_rows(ctx) == 0
    assert multi_loop.next_round_number(ctx) == 4  # round 3 has no row, but its number is taken


def test_sigterm_during_a_wave_cancels_the_array_job_and_keeps_finished_rounds(multi, monkeypatch):
    monkeypatch.setattr(multi_loop.signal, "signal", lambda *args: None)  # keep the signal handlers of pytest
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n) if n in (2, 3) else None,
                           stop_on_wave=1)
    ctx, code = plan_then_run(multi, FakeRoles(), slurm)
    assert code == 1
    assert slurm.canceled == ["900"]
    state = state_of(multi)
    assert [row["round"] for row in records.read_history(state)] == ["1", "2", "3"]  # round 4 has no metrics
    summary = records.read_json(state / "summary.json")
    assert summary["terminated"] is True and summary["final_steps_done"] is False
    assert not (state / "current_job").exists()

    # The next loop job closes the interrupted wave, then plans wave 2.
    slurm.stop_on_wave = None
    ctx = context(multi, FakeRoles(), slurm, max_round_attempts=5)
    assert multi_loop.run(ctx, "loop") == 0
    outcome = records.read_json(state / "waves" / "wave_01" / "outcome.json")
    assert outcome["complete_rounds"] == 2
    assert [row["status"] for row in records.read_history(state)][:4] == ["ok", "ok", "ok", "job_failed"]


def test_finish_uses_the_saved_selection(multi):
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL))
    ctx = context(multi, FakeRoles(), slurm)
    multi_loop.copy_round_one(ctx)
    state = ctx.paths.state
    for round_no, nonzero in ((2, 300_000), (3, 200_000)):
        write_task_records(state, item(round_no), round_no, "round", 0, 0.91, nonzero, f"task-{round_no}")
        records.append_history(state, multi_loop.ok_row(state, round_no))
    records.write_json(state / finish.SELECTION_FILE, {"round": 2})  # saved by an earlier, stopped loop job
    summary = finish.run_final_steps(ctx, "plateau", report_step=multi_loop.report_steps,
                                     extra=multi_loop.summary_extra)
    assert summary["selected"]["round"] == 2
    assert (state / "confirm" / "round_02" / "metrics.json").exists()
    assert (state / "confirm" / "round_02_seed2" / "metrics.json").exists()


def test_audit_only_checks_a_report_without_the_ledger_of_that_folder(multi, monkeypatch):
    state = multi / "state"  # the state folder of the single-agent run
    records.write_json(state / "summary.json", {"selected": {"val_accuracy": 0.9076}})
    (state / "final_report.md").write_text("Validation accuracy 0.9076, test accuracy 0.9999.\n")
    monkeypatch.setenv("STATE_DIR", "state")
    paths = config.paths_from_env()
    agent = FakeRoles()
    ctx = context(multi, agent, FakeArraySlurm(state, lambda n, kind, seed: None))
    ctx.ledger = Ledger(paths.state / "audit_ledger.tsv", multi_loop.MULTI_LEDGER_FIELDS,
                        multi_loop.ROLE_OF.keys(), multi_loop.WAVE_KINDS)
    assert multi_loop.run(ctx, "audit_only") == 0
    audit = records.read_json(state / "audit.json")
    assert audit["unmatched_numbers"] == [{"number": "0.9999", "line": 1}]
    assert agent.kinds() == ["auditor"]
    assert not (state / "ledger.tsv").exists()
    assert (state / "audit_ledger.tsv").exists()


# Files and scores

def test_wave_schema_has_the_round_keys_plus_three():
    round_schema = json.loads((REPO / "configs" / "round.schema.json").read_text())
    wave_item = json.loads((REPO / "configs" / "wave.schema.json").read_text())["properties"]["configurations"]["items"]
    extras = {"hypothesis", "predicted_val_accuracy", "pass_probability"}
    assert set(wave_item["properties"]) == set(round_schema["properties"]) | extras
    for key, value in round_schema["properties"].items():
        assert wave_item["properties"][key] == value
    assert set(wave_item["required"]) == set(round_schema["required"]) | extras


def test_experiment_files_agree():
    single = yaml.safe_load((REPO / "experiment.yaml").read_text())
    multi = yaml.safe_load((REPO / "experiment_multi.yaml").read_text())
    assert multi["target"]["val_accuracy"] == single["target"]["val_accuracy"]
    assert multi["baseline"] == single["baseline"]
    assert multi["round"] == single["round"]
    multi_loop.check_roles(multi)
    assert multi["roles"]["planner"]["model"] != multi["roles"]["critic"]["model"]
    assert multi["roles"]["writer"]["model"] != multi["roles"]["auditor"]["model"]


def test_check_roles_refuses_a_bad_effort_level():
    experiment = yaml.safe_load((REPO / "experiment_multi.yaml").read_text())
    experiment["roles"]["critic"]["effort"] = "huge"
    with pytest.raises(SystemExit, match="effort must be one of"):
        multi_loop.check_roles(experiment)


def test_check_report_lists_only_the_numbers_that_the_records_do_not_show(tmp_path):
    records.write_json(tmp_path / "summary.json", {"selected": {"val_accuracy": 0.9076, "weights_left": 0.007453,
                                                                "nonzero": 83215}, "tokens": {"total": 323072}})
    report = ("Round 15 keeps 0.9076 validation accuracy with 83,215 weights (0.745% left).\n"
              "It used 323,072 tokens with Claude Code 2.1.302 in 2026, with 95% intervals.\n"
              "The test accuracy is 0.9123.\n")
    assert check_report.unmatched_numbers(report, tmp_path) == [{"number": "0.9123", "line": 3}]


def test_prediction_scores_next_to_baselines(multi):
    ctx = context(multi, FakeRoles(), FakeArraySlurm(state_of(multi), lambda n, kind, seed: None))
    multi_loop.copy_round_one(ctx)
    state = ctx.paths.state
    for round_no, val in ((2, 0.92), (3, 0.90)):
        write_task_records(state, item(round_no), round_no, "round", 0, val, SMALL, f"task-{round_no}")
        records.append_history(state, multi_loop.ok_row(state, round_no))
    rows = [{"wave": 1, "round": 2, "status": "run", "val_accuracy": "0.9200", "predicted_val_accuracy": 0.91,
             "pass_probability": 0.9, "passes": "yes"},
            {"wave": 1, "round": 3, "status": "run", "val_accuracy": "0.9000", "predicted_val_accuracy": 0.91,
             "pass_probability": 0.6, "passes": "no"}]
    scores = multi_loop.prediction_scores(ctx, rows)
    assert scores["rounds"] == 2
    assert scores["mean_absolute_error"] == 0.01
    assert scores["baseline_mean_absolute_error"] == 0.02  # round 1 (0.93) is the only earlier round
    assert scores["brier_score"] == round((0.1 ** 2 + 0.6 ** 2) / 2, 4)
    assert scores["base_rate_brier_score"] == 0.25
    assert scores["pass_decisions_right"] == 1


def test_a_failed_sbatch_is_a_failed_wave(multi):
    class NoSubmit(FakeArraySlurm):
        def submit_array(self, script, tasks_file, count, max_parallel, cwd, options=()):
            raise multi_loop.SlurmError("sbatch failed: invalid partition")

    ctx = context(multi, FakeRoles(), NoSubmit(state_of(multi), lambda n, kind, seed: (0.91, SMALL)))
    multi_loop.copy_round_one(ctx)
    assert multi_loop.run_waves(ctx) == "failed_attempts"  # two waves in a row with no complete round
    statuses = [row["status"] for row in records.read_history(ctx.paths.state)]
    assert statuses == ["ok"] + ["sbatch_failed"] * 6


def test_finish_reuses_the_records_of_an_earlier_run(multi):
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n) if kind == "round" else (0.906, SMALL))
    plan_then_run(multi, FakeRoles(), slurm, max_round_attempts=4)
    arrays = len(slurm.arrays)
    ctx = context(multi, FakeRoles(), slurm)
    assert multi_loop.run(ctx, "finish") == 0
    assert len(slurm.arrays) == arrays  # no confirmation round runs again
    summary = records.read_json(state_of(multi) / "summary.json")
    assert summary["stop_reason"] == "round_limit" and summary["final_steps_done"] is True


def test_a_new_loop_job_cancels_the_array_job_of_a_stopped_one(multi):
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL))
    planner = context(multi, FakeRoles(), slurm)
    planner.plan_only = True
    assert multi_loop.run(planner, "plan_only") == 0
    (state_of(multi) / "current_job").write_text("812\n")  # the loop job got SIGKILL during a wave
    ctx = context(multi, FakeRoles(), slurm, max_round_attempts=4)
    multi_loop.check_start(ctx)
    assert slurm.canceled == ["812"]
    assert not (state_of(multi) / "current_job").exists()


# Fixes after the code review

def test_the_loop_refuses_an_empty_plan_of_wave_1(multi):
    ctx = context(multi, FakeRoles(reject=lambda kind, entry: "no"), FakeArraySlurm(state_of(multi), lambda n, kind, seed: None))
    ctx.plan_only = True
    assert multi_loop.run(ctx, "plan_only") == 1  # the critic rejects each configuration, also after the revision
    assert records.read_json(state_of(multi) / "waves" / "wave_01" / "accepted.json") == []
    assert multi_loop.run(ctx, "plan_only") == 1  # a second --plan-only job says to plan again
    ctx = context(multi, FakeRoles(), FakeArraySlurm(state_of(multi), lambda n, kind, seed: None))
    with pytest.raises(SystemExit, match="no accepted configuration"):
        multi_loop.run(ctx, "loop")


def test_finish_closes_the_wave_of_a_stopped_loop_job(multi):
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n) if n in (2, 3) or kind == "confirm" else None)
    planner = context(multi, FakeRoles(), slurm)
    planner.plan_only = True
    assert multi_loop.run(planner, "plan_only") == 0
    ctx = context(multi, FakeRoles(), slurm)
    multi_loop.check_start(ctx)
    accepted = records.read_json(state_of(multi) / "waves" / "wave_01" / "accepted.json")
    # The loop job submits wave 1 and gets SIGKILL: no handler runs, so current_job stays.
    original_wait = slurm.wait_array
    slurm.wait_array = lambda job, limit: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        multi_loop.submit_wave(ctx, 1, accepted)
    slurm.wait_array = original_wait
    assert (state_of(multi) / "current_job").exists()
    ctx = context(multi, FakeRoles(), slurm)
    assert multi_loop.run(ctx, "finish") == 0
    assert slurm.canceled == ["900"]  # the array job of the stopped loop job
    outcome = records.read_json(state_of(multi) / "waves" / "wave_01" / "outcome.json")
    assert outcome["complete_rounds"] == 2
    statuses = [row["status"] for row in records.read_history(state_of(multi))]
    assert statuses == ["ok", "ok", "ok", "job_failed"]
    assert [row["kind"] for row in ctx.ledger.rows if row["kind"] == "round"] == ["round"] * 3  # GPU minutes counted


def test_a_writer_with_another_model_does_not_stop_the_code_checks(multi):
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n) if kind == "round" else (0.906, SMALL))
    agent = FakeRoles(models={"writer": "claude-haiku-4-5"})
    ctx, code = plan_then_run(multi, agent, slurm, max_round_attempts=4)
    assert code == 0
    audit = records.read_json(state_of(multi) / "audit.json")
    assert "not the configured model" in audit["failures"]["writer"]
    assert audit["failures"]["auditor"] == "skipped: there is no report"
    assert audit["auditor_step_failed"] is True
    assert audit["check_round"]["round"] == 4  # the code checks still ran
    assert records.read_json(state_of(multi) / "summary.json")["audit"]["failures"]["writer"]


def test_an_audit_without_structured_output_is_a_failed_audit(multi):
    class NoAudit(FakeRoles):
        def run(self, prompt, caps, schema_text=None):
            result = super().run(prompt, caps, schema_text)
            if Path(caps["role_prompt"]).stem == "auditor":
                result.pop("structured_output")
                result["subtype"] = "error_max_turns"
            return result

    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n) if kind == "round" else (0.906, SMALL))
    plan_then_run(multi, NoAudit(), slurm, max_round_attempts=4)
    audit = records.read_json(state_of(multi) / "audit.json")
    assert audit["auditor_step_failed"] is True
    assert audit["failures"]["auditor"] == "no valid audit (subtype error_max_turns)"


def test_critic_counts_count_each_configuration_one_time(multi):
    agent = FakeRoles(fail={"critic"})
    ctx = context(multi, agent, FakeArraySlurm(state_of(multi), lambda n, kind, seed: None))
    multi_loop.copy_round_one(ctx)
    assert multi_loop.plan_wave(ctx, 1, 3) is None
    multi_loop.write_outcome(ctx, 1, 0, "a role step failed")
    counts = multi_loop.summary_extra(ctx)["configurations"]
    assert counts["planned"] == 3 and counts["not_reviewed"] == 3
    assert counts["rejected_by_critic"] == 0 and counts["accepted"] == 0


def test_check_report_uses_the_factor_100_only_for_percentages_and_keeps_signs(tmp_path):
    records.write_json(tmp_path / "summary.json", {"weights_left": 0.0075, "val_accuracy": 0.9076, "error": -0.0123,
                                                   "rounds": 45})
    report = "0.75% right, 0.75 wrong, 0.9076% wrong, -0.0123 right, 0.0123 wrong, 2,048 checked, 4500 checked\n"
    found = [item["number"] for item in check_report.unmatched_numbers(report, tmp_path)]
    assert found == ["0.75", "0.9076%", "0.0123", "2,048", "4500"]


# Fixes after the cluster tests

def test_finish_counts_the_confirmation_array_job_that_a_stopped_loop_job_canceled(multi, monkeypatch):
    monkeypatch.setattr(multi_loop.signal, "signal", lambda *args: None)  # keep the signal handlers of pytest
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n) if kind == "round" else None,
                           stop_on_wave=2)  # array job 2 is the confirmation of the selected round
    ctx, code = plan_then_run(multi, FakeRoles(), slurm, max_round_attempts=4)
    assert code == 1
    state = state_of(multi)
    job = slurm.arrays[1][0]
    assert slurm.canceled == [job]
    selected = records.read_json(state / finish.SELECTION_FILE)["round"]
    assert (state / finish.PENDING_FILE).read_text() == f"{job}\tconfirm\t{selected}\t2\n"

    slurm.stop_on_wave = None
    slurm.results = lambda n, kind, seed: (0.91, SMALL - n) if kind == "round" else (0.906, SMALL)
    ctx = context(multi, FakeRoles(), slurm)
    assert multi_loop.run(ctx, "finish") == 0
    rows = [(row["status"], row["job_id"]) for row in ctx.ledger.rows if row["kind"] == "confirm"]
    assert rows[:2] == [("interrupted", f"{job}0"), ("interrupted", f"{job}1")]
    assert len(rows) == 4  # then the two new confirmation rounds
    assert not (state / finish.PENDING_FILE).exists()


def test_check_report_reads_the_baseline_metrics(tmp_path):
    state = tmp_path / "state_multi"
    records.write_json(state / "summary.json", {"selected": {"val_accuracy": 0.9182}})
    records.write_json(tmp_path / config.BASELINE_METRICS, {"val_accuracy": 0.937})
    report = "Round 4 keeps 0.9182 validation accuracy. The baseline has 0.937.\n"
    assert check_report.unmatched_numbers(report, state) == [{"number": "0.937", "line": 1}]
    assert check_report.unmatched_numbers(report, state, tmp_path / config.BASELINE_METRICS) == []


def test_round_constraint_reaches_the_array_jobs(multi):
    slurm = FakeArraySlurm(state_of(multi), lambda n, kind, seed: (0.91, SMALL - n))
    planner = context(multi, FakeRoles(), slurm)
    planner.plan_only = True
    assert multi_loop.run(planner, "plan_only") == 0
    ctx = context(multi, FakeRoles(), slurm, max_round_attempts=4)
    ctx.experiment["round"]["constraint"] = "a100-mig"
    assert multi_loop.run(ctx, "loop") == 0
    assert slurm.options == [("--constraint=a100-mig",)] * 2  # the wave and the confirmation rounds
