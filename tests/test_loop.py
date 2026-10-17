"""Tests of the loop script with a fake agent and fake SLURM commands. No API calls and no jobs."""

from __future__ import annotations

import pytest

from pruning_example import config, finish, loop, metrics, records
from pruning_example.agent import AgentFormatError, tokens_used
from pruning_example.slurm import Accounting, Slurm, parse_sacct

GOOD = {"sparsity": 0.98, "steps": 4, "finetune_epochs": 16, "lr": 0.02, "weight_decay": 0.0005,
        "exclude_layers": [], "reason": "test"}
BAD = {**GOOD, "sparsity": 2.0}
TARGET = {"val_accuracy": 0.905, "size_target": 0.02}
SMALL = metrics.max_nonzero(0.02)


def usage(tokens: int) -> dict:
    return {"model-a": {"inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": tokens - 40,
                        "cacheCreationInputTokens": 10}}


class FakeAgent:
    """Returns one configuration and one token count for each call."""

    def __init__(self, config_for_call, tokens: int = 50_000):
        self.config_for_call = config_for_call
        self.tokens = tokens
        self.calls = 0

    def run(self, prompt, caps, schema_text=None):
        self.calls += 1
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
              "meets_target": metrics.meets_target(val_accuracy, nonzero, TARGET)}
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

    def cancel(self, job_id):
        self.canceled.append(job_id)


@pytest.fixture
def started(project):
    """A project after the baseline and the supervised round 1."""
    state = project / "state"
    records.write_json(state / "baseline" / "metrics.json", {"val_accuracy": 0.94})
    write_round(state, 1, 0.93, 1_000_000)  # accurate, but too large
    return project


def context(project, agent, slurm, **budget) -> loop.Context:
    paths = config.paths_from_env()
    experiment = config.load_experiment(project)
    experiment["target"] = dict(TARGET)
    experiment["budget"].update(budget)
    return loop.Context(paths=paths, experiment=experiment, schema_text=(project / config.SCHEMA_PATH).read_text(),
                        agent=agent, slurm=slurm, ledger=loop.Ledger(paths.state / "ledger.tsv"),
                        log=lambda message: None)


def test_stops_when_a_round_meets_the_target(started):
    state = started / "state"
    slurm = FakeSlurm(state, lambda n: (0.91, SMALL) if n == 3 else (0.89, SMALL))
    ctx = context(started, FakeAgent(lambda call: GOOD), slurm)
    assert loop.run_rounds(ctx) == "target"
    assert [row["status"] for row in records.read_history(state)] == ["ok", "ok", "ok"]
    assert ctx.ledger.tokens == 2 * 50_000
    assert (state / "notes.md").read_text().count("## Round") == 2


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
    ctx.job_end = ctx.clock() + 60 * 60  # one hour left; a round needs 20 + 45 + 80 minutes
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
    summary = records.read_json(started / "state" / "summary.json")
    assert summary["stop_reason"] == "terminated"
    assert summary["final_steps_done"] is False


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


def test_queue_time_adds_up_over_requeues():
    states = iter(["PENDING", "RUNNING", "PENDING", "PENDING", None])
    times = iter(range(0, 1000, 60))
    slurm = Slurm(sleep=lambda seconds: None, clock=lambda: next(times))
    slurm.queue_state = lambda job_id: next(states)
    assert slurm.wait("1", queue_limit_seconds=150) == "queue_limit"


def test_select_round_rules():
    rows = [
        {"round": "1", "status": "ok", "meets_target": "no", "val_accuracy": "0.9300", "nonzero": "1000000"},
        {"round": "2", "status": "ok", "meets_target": "yes", "val_accuracy": "0.9060", "nonzero": "200000"},
        {"round": "3", "status": "invalid_config", "meets_target": "", "val_accuracy": "", "nonzero": ""},
    ]
    assert finish.select_round(rows, TARGET)["round"] == "2"
    rows[1]["meets_target"] = "no"
    assert finish.select_round(rows, TARGET)["round"] == "2"  # enough accuracy, fewest weights
    rows[1]["val_accuracy"] = "0.9000"
    rows[0]["val_accuracy"] = "0.9040"
    assert finish.select_round(rows, TARGET)["round"] == "1"  # highest accuracy


def test_final_steps_write_the_summary_and_the_report(started, monkeypatch):
    state = started / "state"
    slurm = FakeSlurm(state, lambda n: (0.91, SMALL))
    ctx = context(started, FakeAgent(lambda call: GOOD), slurm)
    assert loop.run_rounds(ctx) == "target"
    monkeypatch.setattr(finish, "evaluate_on_test_set", lambda ctx, kind, n: {"test_accuracy": 0.902, "kind": kind})
    summary = finish.run_final_steps(ctx, "target")
    assert summary["selected"]["round"] == 2
    assert summary["confirmation"]["seed"] == 1
    assert summary["test"]["selected"]["test_accuracy"] == 0.902
    assert summary["tokens"]["total"] == 2 * 50_000  # one agent step and the report step
    assert ("confirm", 2) in slurm.submitted
    assert (started / "reports" / "final_report.md").read_text().strip() == "Draft report."
