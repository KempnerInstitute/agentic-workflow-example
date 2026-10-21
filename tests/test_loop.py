"""Tests of the loop script with a fake agent and fake SLURM commands. No API calls and no jobs."""

from __future__ import annotations

import json

import pytest

from pruning_example import config, finish, loop, metrics, records
from pruning_example.agent import (
    AgentFormatError, ModelCheckError, UNATTENDED_SETTINGS, check_model, command, step_env, tokens_used,
)
from pruning_example.slurm import Accounting, Slurm, SlurmError, parse_sacct

GOOD = {"sparsity": 0.98, "steps": 4, "finetune_epochs": 16, "lr": 0.02, "weight_decay": 0.0005,
        "exclude_layers": [], "reason": "test"}
BAD = {**GOOD, "sparsity": 2.0}
TARGET = {"val_accuracy": 0.905, "plateau_rounds": 100}  # tests of the plateau rule set their own value
SMALL = 223_287


def usage(tokens: int) -> dict:
    return {"model-a": {"inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": tokens - 40,
                        "cacheCreationInputTokens": 10}}


class FakeAgent:
    """Returns one configuration and one token count for each call, and keeps the prompts."""

    def __init__(self, config_for_call, tokens: int = 50_000):
        self.config_for_call = config_for_call
        self.tokens = tokens
        self.calls = 0
        self.prompts = []

    def run(self, prompt, caps, schema_text=None):
        self.calls += 1
        self.prompts.append(prompt)
        return {"subtype": "success", "modelUsage": usage(self.tokens), "total_cost_usd": 0.05,
                "structured_output": self.config_for_call(self.calls) if schema_text else None,
                "result": "Draft report."}

    def stop(self):
        pass


def write_round(state, round_no, val_accuracy, nonzero, kind="round", seed=0, round_config=GOOD):
    """Write the records that a real round job writes."""
    result = {"round": round_no, "kind": kind, "seed": seed, "val_accuracy": val_accuracy,
              "val_interval": metrics.wilson_interval(val_accuracy, 5000), "nonzero": nonzero,
              "weights_left": nonzero / 11_164_352, "sparse_size_mb": 1.0,
              "passes": metrics.passes(val_accuracy, TARGET)}
    folder = state / ("rounds" if kind == "round" else "confirm") / f"round_{round_no:02d}"
    records.write_json(folder / "metrics.json", result)
    records.write_json(folder / "config.json", round_config)
    records.write_json(folder / "run_meta.json", {"job_id": "test", "run_folder": str(state / "no_run_folder")})
    if kind == "round":
        records.append_history(state, records.history_row(round_no, "ok", round_config, result))


class FakeSlurm:
    """Runs no jobs. A submitted round writes the records that a real round job writes."""

    def __init__(self, state, results, gpu_minutes: float = 5.0):
        self.state = state
        self.results = results  # round number -> (val_accuracy, nonzero), or None for a job without metrics
        self.gpu_minutes = gpu_minutes
        self.submitted = []
        self.canceled = []

    def submit(self, script, args, cwd):
        round_no = int(args[args.index("--round") + 1])
        kind = args[args.index("--kind") + 1] if "--kind" in args else "round"
        self.submitted.append((kind, round_no))
        outcome = self.results(round_no)
        if outcome is not None:
            write_round(self.state, round_no, *outcome, kind=kind, seed=1 if kind == "confirm" else 0)
        return str(1000 + len(self.submitted))

    def wait(self, job_id, queue_limit_seconds):
        return "ended"

    def accounting(self, job_id):
        return Accounting("COMPLETED", self.gpu_minutes, 1)

    def cancel(self, job_id, wait_seconds=60):
        self.canceled.append(job_id)
        return True


@pytest.fixture
def started(project):
    """A project after the baseline and the supervised round 1."""
    state = project / "state"
    records.write_json(state / "baseline" / "metrics.json", {"val_accuracy": 0.94})
    write_round(state, 1, 0.93, 1_000_000)  # round 1 passes, with many weights left
    return project


def context(project, agent, slurm, plateau_rounds: int = 100, **budget) -> loop.Context:
    paths = config.paths_from_env()
    experiment = config.load_experiment(project)
    experiment["target"] = {**TARGET, "plateau_rounds": plateau_rounds}
    experiment["budget"].update(budget)
    for step in ("agent_step", "report_step"):  # the fake agent reports model-a; test_model_check covers the check
        experiment[step].pop("model", None)
    return loop.Context(paths=paths, experiment=experiment, schema_text=(project / config.SCHEMA_PATH).read_text(),
                        agent=agent, slurm=slurm, ledger=loop.Ledger(paths.state / "ledger.tsv"),
                        log=lambda message: None)


def test_stops_after_rounds_without_a_smaller_passing_model(started):
    state = started / "state"
    # Round 2 passes with fewer weights. Rounds 3, 4, and 5 do not keep the threshold.
    slurm = FakeSlurm(state, lambda n: (0.91, 200_000) if n == 2 else (0.89, 100_000))
    agent = FakeAgent(lambda call: GOOD)
    ctx = context(started, agent, slurm, plateau_rounds=3)
    assert loop.run_rounds(ctx) == "plateau"
    assert [row["round"] for row in records.read_history(state)] == ["1", "2", "3", "4", "5"]
    assert ctx.ledger.tokens == 4 * 50_000
    assert (state / "notes.md").read_text().count("## Round") == 4
    assert "round 2, 200,000 nonzero weights" in agent.prompts[-1]
    assert "Rounds in a row without a smaller model: 2 of 3" in agent.prompts[-1]


def test_a_smaller_passing_model_resets_the_plateau_count():
    rows = [{"round": str(n), "status": "ok", "val_accuracy": acc, "nonzero": size}
            for n, (acc, size) in enumerate([("0.93", "900"), ("0.89", "500"), ("0.91", "800"), ("0.92", "850")], 1)]
    assert finish.rounds_without_improvement(rows, TARGET) == 1  # round 3 improved; round 4 is larger
    assert finish.best_round(rows, TARGET)["round"] == "3"


def test_the_current_threshold_decides_not_the_stored_value(started):
    state = started / "state"
    row = records.history_row(2, "ok", GOOD, {"val_accuracy": 0.90, "weights_left": 0.02, "nonzero": SMALL,
                                              "passes": True})  # stored under an earlier, lower threshold
    records.append_history(state, row)
    ctx = context(started, FakeAgent(lambda call: GOOD), FakeSlurm(state, lambda n: (0.89, SMALL)), plateau_rounds=1)
    assert loop.stop_reason(ctx) == "plateau"  # round 2 does not keep 0.905, so it is no improvement


def test_stops_before_the_token_budget(started):
    agent = FakeAgent(lambda call: GOOD, tokens=300_000)
    ctx = context(started, agent, FakeSlurm(started / "state", lambda n: (0.89, SMALL)),
                  tokens=1_000_000, report_reserve_fraction=0.1)
    assert loop.run_rounds(ctx) == "token_budget"
    assert agent.calls == 3  # 3 x 300,000 = 900,000, the limit for agent steps
    assert ctx.ledger.tokens == 900_000


def test_stops_at_the_round_limit(started):
    ctx = context(started, FakeAgent(lambda call: GOOD), FakeSlurm(started / "state", lambda n: (0.89, SMALL)),
                  max_round_attempts=3)
    assert loop.run_rounds(ctx) == "round_limit"
    assert len(records.read_history(started / "state")) == 3


def test_stops_after_failed_attempts_in_a_row(started):
    slurm = FakeSlurm(started / "state", lambda n: (0.89, SMALL))
    ctx = context(started, FakeAgent(lambda call: BAD), slurm, max_failed_in_a_row=2)
    assert loop.run_rounds(ctx) == "failed_attempts"
    statuses = [row["status"] for row in records.read_history(started / "state")]
    assert statuses == ["ok", "invalid_config", "invalid_config"]
    assert slurm.submitted == []


def test_a_job_without_metrics_is_a_failed_attempt(started):
    ctx = context(started, FakeAgent(lambda call: GOOD), FakeSlurm(started / "state", lambda n: None),
                  max_failed_in_a_row=2)
    assert loop.run_rounds(ctx) == "failed_attempts"
    statuses = [row["status"] for row in records.read_history(started / "state")]
    assert statuses == ["ok", "job_failed", "job_failed"]
    assert ctx.ledger.gpu_minutes == 10.0  # failed jobs count too


def test_stops_at_the_gpu_budget(started):
    slurm = FakeSlurm(started / "state", lambda n: (0.89, SMALL), gpu_minutes=100.0)
    ctx = context(started, FakeAgent(lambda call: GOOD), slurm, gpu_minutes=180)
    assert loop.run_rounds(ctx) == "gpu_budget"
    assert ctx.ledger.gpu_minutes == 200.0


def test_stops_when_a_tracked_file_changed(started):
    (started / "experiment.yaml").write_text((started / "experiment.yaml").read_text() + "# changed\n")
    ctx = context(started, FakeAgent(lambda call: GOOD), FakeSlurm(started / "state", lambda n: (0.89, SMALL)))
    assert loop.run_rounds(ctx) == "integrity"


def test_stops_without_time_for_one_more_round(started):
    ctx = context(started, FakeAgent(lambda call: GOOD), FakeSlurm(started / "state", lambda n: (0.89, SMALL)))
    ctx.job_end = ctx.clock() + 60 * 60  # one hour left; a round needs 10 + 45 + 65 minutes
    assert loop.run_rounds(ctx) == "time_left"


def test_sigterm_cancels_the_running_round_job(started, monkeypatch):
    class StoppedSlurm(FakeSlurm):
        def wait(self, job_id, queue_limit_seconds):
            raise loop.Terminated()

    monkeypatch.setattr(loop.signal, "signal", lambda *args: None)  # keep the signal handlers of pytest
    slurm = StoppedSlurm(started / "state", lambda n: None)
    ctx = context(started, FakeAgent(lambda call: GOOD), slurm)
    assert loop.run(ctx) == 1
    assert slurm.canceled == ["1001"]
    assert not (started / "state" / "current_job").exists()
    summary = records.read_json(started / "state" / "summary.json")
    assert summary["stop_reason"] == "terminated"
    assert summary["final_steps_done"] is False


def test_sigterm_writes_the_summary_when_scancel_is_late(started, monkeypatch):
    class LateSlurm(FakeSlurm):
        def wait(self, job_id, queue_limit_seconds):
            raise loop.Terminated()

        def cancel(self, job_id, wait_seconds=60):
            self.canceled.append(job_id)
            return False  # scancel did not answer in time

    monkeypatch.setattr(loop.signal, "signal", lambda *args: None)
    slurm = LateSlurm(started / "state", lambda n: None)
    assert loop.run(context(started, FakeAgent(lambda call: GOOD), slurm)) == 1
    assert slurm.canceled == ["1001"]
    assert records.read_json(started / "state" / "summary.json")["stop_reason"] == "terminated"


def test_tokens_add_all_models_and_cache_reads():
    result = {"modelUsage": {
        "a": {"inputTokens": 1, "outputTokens": 2, "cacheReadInputTokens": 3, "cacheCreationInputTokens": 4},
        "b": {"inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": 30, "cacheCreationInputTokens": 40},
    }}
    assert tokens_used(result) == 110


def test_missing_model_usage_stops_the_loop():
    with pytest.raises(AgentFormatError):
        tokens_used({"result": "no usage"})


def test_sacct_adds_the_runs_of_a_requeued_job():
    text = ("123|PREEMPTED|300|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1\n"
            "123|COMPLETED|600|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1\n")
    usage_record = parse_sacct(text)
    assert usage_record.state == "COMPLETED"
    assert usage_record.gpu_minutes == 15.0
    assert usage_record.runs == 2


def test_sbatch_tries_again_after_a_controller_timeout():
    class Result:
        def __init__(self, code, out="", err=""):
            self.returncode, self.stdout, self.stderr = code, out, err

    answers = iter([Result(1, err="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"),
                    Result(0, out="777\n")])
    slurm = Slurm(sleep=lambda seconds: None)
    slurm.run = lambda args, timeout=120, **kwargs: next(answers)
    assert slurm.submit("slurm/round.sbatch", [], cwd=".") == "777"

    slurm.run = lambda args, timeout=120, **kwargs: Result(1, err="sbatch: error: invalid partition specified")
    with pytest.raises(SlurmError):
        slurm.submit("slurm/round.sbatch", [], cwd=".")


def test_queue_time_adds_up_over_requeues():
    states = iter(["PENDING", "RUNNING", "PENDING", "PENDING", None])
    times = iter(range(0, 1000, 60))
    slurm = Slurm(sleep=lambda seconds: None, clock=lambda: next(times))
    slurm.queue_state = lambda job_id: next(states)
    assert slurm.wait("1", queue_limit_seconds=150) == "queue_limit"


def test_select_round_rules():
    rows = [
        {"round": "1", "status": "ok", "val_accuracy": "0.9300", "nonzero": "1000000"},
        {"round": "2", "status": "ok", "val_accuracy": "0.9060", "nonzero": "200000"},
        {"round": "3", "status": "invalid_config", "val_accuracy": "", "nonzero": ""},
    ]
    assert finish.select_round(rows, TARGET)["round"] == "2"  # passes, fewest weights
    rows[1]["val_accuracy"] = "0.9000"
    assert finish.select_round(rows, TARGET)["round"] == "1"  # the only round that passes
    rows[0]["val_accuracy"] = "0.9040"
    assert finish.select_round(rows, TARGET)["round"] == "1"  # no round passes: highest accuracy


def test_final_steps_write_the_summary_and_the_report(started, monkeypatch):
    state = started / "state"
    slurm = FakeSlurm(state, lambda n: (0.91, SMALL))
    ctx = context(started, FakeAgent(lambda call: GOOD), slurm, plateau_rounds=1)
    assert loop.run_rounds(ctx) == "plateau"  # round 2 improves, round 3 has the same size
    monkeypatch.setattr(finish, "evaluate_on_test_set", lambda ctx, kind, n: {"test_accuracy": 0.902, "kind": kind})
    summary = finish.run_final_steps(ctx, "plateau")
    assert summary["selected"]["round"] == 2
    assert summary["selected"]["seed"] == 0
    assert summary["agent_models"] == ["model-a"]
    assert summary["confirmation"]["seed"] == 1
    assert summary["test"]["selected"]["test_accuracy"] == 0.902
    assert summary["tokens"]["total"] == 3 * 50_000  # two agent steps and the report step
    assert ("confirm", 2) in slurm.submitted
    assert (state / "final_report.md").read_text().strip() == "Draft report."


def test_sigterm_in_the_final_steps_keeps_the_stop_reason(started, monkeypatch):
    class StopInConfirmation(FakeSlurm):
        def wait(self, job_id, queue_limit_seconds):
            if self.submitted[-1][0] == "confirm":
                raise loop.Terminated()
            return "ended"

    monkeypatch.setattr(loop.signal, "signal", lambda *args: None)
    slurm = StopInConfirmation(started / "state", lambda n: (0.91, SMALL))
    assert loop.run(context(started, FakeAgent(lambda call: GOOD), slurm, plateau_rounds=1)) == 1
    summary = records.read_json(started / "state" / "summary.json")
    assert summary["stop_reason"] == "plateau"  # --finish can use it later
    assert summary["terminated"] is True
    assert summary["final_steps_done"] is False


def test_model_check_stops_the_loop(started):
    ctx = context(started, FakeAgent(lambda call: GOOD), FakeSlurm(started / "state", lambda n: (0.91, SMALL)))
    ctx.experiment["agent_step"]["model"] = "claude-opus-5-5"  # the fake agent reports model-a
    assert loop.run_rounds(ctx) == "model_check"
    assert ctx.ledger.tokens == 50_000  # the step still counts


def test_check_model_skips_aliases_and_accepts_suffixes():
    result = {"modelUsage": {"claude-opus-5-5[1m]": {}, "claude-haiku-4-5": {}}}
    check_model(result, "claude-opus-5-5")
    check_model(result, "opus")  # an alias names no fixed model
    check_model(result, None)
    with pytest.raises(ModelCheckError, match="not the configured model claude-sonnet-5-5"):
        check_model(result, "claude-sonnet-5-5")


def test_gpu_reservation_starts_a_round_only_if_its_time_limit_fits(started):
    slurm = FakeSlurm(started / "state", lambda n: (0.89, SMALL), gpu_minutes=8.0)
    ctx = context(started, FakeAgent(lambda call: GOOD), slurm, gpu_minutes=60, gpu_reservation=True)
    assert loop.run_rounds(ctx) == "gpu_budget"
    assert ctx.ledger.gpu_minutes == 56.0  # 48 + 10 fitted in 60; 56 + 10 does not
    ctx = context(started, FakeAgent(lambda call: GOOD), slurm, gpu_minutes=60)
    assert loop.stop_reason(ctx) is None  # the rule of chapter 28 starts one more round, because 56 is below 60


def test_the_ledger_of_chapter_28_keeps_its_columns(tmp_path):
    ledger = loop.Ledger(tmp_path / "ledger.tsv")
    ledger.add(round=2, kind="agent_step", tokens=100, wave=1, model="m")
    assert (tmp_path / "ledger.tsv").read_text().splitlines()[0].split("\t") == loop.LEDGER_FIELDS
    assert ledger.tokens == 100


def test_command_adds_the_model_the_effort_the_role_prompt_and_the_readme_rule():
    caps = {"max_turns": 8, "max_budget_usd": 1.0, "timeout_minutes": 15}
    args = command("prompt", caps)
    assert "--model" not in args and "--effort" not in args and "--append-system-prompt-file" not in args
    assert json.loads(args[args.index("--settings") + 1]) == UNATTENDED_SETTINGS
    assert "Read(./README.md)" in UNATTENDED_SETTINGS["permissions"]["deny"]
    args = command("prompt", {**caps, "model": "claude-opus-5-5", "effort": "xhigh",
                              "role_prompt": "prompts/multi/roles/planner.md"}, schema_text="{}")
    assert args[args.index("--model") + 1] == "claude-opus-5-5"
    assert args[args.index("--effort") + 1] == "xhigh"
    assert args[args.index("--append-system-prompt-file") + 1] == "prompts/multi/roles/planner.md"
    assert args[-2:] == ["--json-schema", "{}"]


def test_the_effort_variable_is_removed_only_when_the_step_sets_an_effort(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")  # it would override --effort
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    assert "CLAUDE_CODE_EFFORT_LEVEL" not in step_env({"effort": "high"})
    assert step_env({})["CLAUDE_CODE_EFFORT_LEVEL"] == "low"
    assert "ANTHROPIC_API_KEY" not in step_env({}) and "ANTHROPIC_API_KEY" not in step_env({"effort": "high"})


def test_record_paths_of_confirmation_seeds(tmp_path):
    assert records.record_dir(tmp_path, "round", 7) == tmp_path / "rounds" / "round_07"
    assert records.record_dir(tmp_path, "confirm", 7) == tmp_path / "confirm" / "round_07"
    assert records.record_dir(tmp_path, "confirm", 7, 1) == tmp_path / "confirm" / "round_07"
    assert records.record_dir(tmp_path, "confirm", 7, 2) == tmp_path / "confirm" / "round_07_seed2"
    assert records.test_eval_path(tmp_path, "confirm", 7, 2) == tmp_path / "test_eval" / "confirm_07_seed2.json"
    assert records.test_eval_path(tmp_path, "round", 7) == tmp_path / "test_eval" / "round_07.json"
