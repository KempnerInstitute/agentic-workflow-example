"""Tests of the SLURM array-job wrappers: parsing, waiting, the queue limit, and accounting. No jobs."""

from __future__ import annotations

from pruning_example.slurm import Slurm, parse_array_sacct, parse_array_squeue, task_indexes

TRES = "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1"


class Result:
    def __init__(self, code, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def test_task_indexes_of_array_job_ids():
    assert task_indexes("500_4") == [4]
    assert task_indexes("500_[1-2%3]") == [1, 2]
    assert task_indexes("500_[0,2]") == [0, 2]
    assert task_indexes("500") == []


def test_sacct_adds_the_runs_of_a_requeued_task_and_keeps_tasks_that_never_started():
    text = (f"500_0|501|PREEMPTED|2026-10-21T10:00:00|300|{TRES}\n"
            f"500_0|501|COMPLETED|2026-10-21T10:06:00|240|{TRES}\n"
            f"500_1|502|COMPLETED|2026-10-21T10:00:05|120|{TRES}\n"
            "500_[2]|500|CANCELLED by 123|Unknown|0|\n")
    tasks = parse_array_sacct(text)
    assert tasks[0].state == "COMPLETED"
    assert tasks[0].gpu_minutes == 9.0
    assert tasks[0].runs == 2
    assert tasks[0].job_id == "501"
    assert tasks[1].gpu_minutes == 2.0
    assert tasks[2].state == "CANCELLED"
    assert tasks[2].gpu_minutes == 0.0
    assert tasks[2].runs == 0


def test_squeue_gives_one_state_for_each_task():
    assert parse_array_squeue("0 RUNNING\n1 PENDING\n   2 PENDING\n") == {0: "RUNNING", 1: "PENDING", 2: "PENDING"}
    assert parse_array_squeue("") == {}


def fake_slurm(states, step: int = 60):
    times = iter(range(0, 100_000, step))
    slurm = Slurm(sleep=lambda seconds: None, clock=lambda: next(times))
    answers = iter(states)
    slurm.array_states = lambda array_id: next(answers)
    slurm.waiting_cancels = 0

    def cancel_waiting(array_id, wait_seconds=60):
        slurm.waiting_cancels += 1
        return True

    slurm.cancel_waiting = cancel_waiting
    slurm.cancel = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not cancel all tasks"))
    return slurm


def test_wait_ends_when_no_task_is_left():
    slurm = fake_slurm([{0: "PENDING", 1: "PENDING"}, {0: "RUNNING", 1: "RUNNING"}, {}])
    assert slurm.wait_array("500", queue_limit_seconds=3600) == "ended"
    assert slurm.waiting_cancels == 0


def test_queue_limit_cancels_only_waiting_tasks_also_after_a_preemption():
    states = [
        {0: "RUNNING", 1: "PENDING"},  # 60 seconds of queue time
        {0: "RUNNING", 1: "PENDING"},  # 120 seconds: past the limit, so the waiting task 1 is canceled
        {0: "RUNNING"},
        {0: "PENDING"},  # a preemption put task 0 back in the queue: cancel it too
        {},
    ]
    slurm = fake_slurm(states)
    assert slurm.wait_array("500", queue_limit_seconds=100) == "queue_limit"
    assert slurm.waiting_cancels == 2


def test_submit_array_passes_the_range_and_the_throttle():
    seen = []

    def run(args, timeout=120, **kwargs):
        seen.append(args)
        return Result(0, out="777\n")

    slurm = Slurm(sleep=lambda seconds: None)
    slurm.run = run
    assert slurm.submit_array("slurm/round_array.sbatch", "state_multi/waves/wave_01/tasks.tsv", 3, 2, cwd=".") == "777"
    assert seen[0] == ["sbatch", "--parsable", "--array=0-2%2", "slurm/round_array.sbatch",
                       "state_multi/waves/wave_01/tasks.tsv"]


def test_array_accounting_waits_for_final_states():
    answers = iter([Result(0, out=f"500_0|501|RUNNING|2026-10-21T10:00:00|60|{TRES}\n"),
                    Result(0, out=f"500_0|501|COMPLETED|2026-10-21T10:00:00|90|{TRES}\n")])
    slurm = Slurm(sleep=lambda seconds: None)
    slurm.run = lambda args, timeout=120, **kwargs: next(answers)
    tasks = slurm.array_accounting("500", 1)
    assert tasks[0].state == "COMPLETED"
    assert tasks[0].gpu_minutes == 1.5


def test_array_accounting_does_not_raise_without_records():
    slurm = Slurm(sleep=lambda seconds: None)
    slurm.run = lambda args, timeout=120, **kwargs: Result(1, err="sacct: error")
    assert slurm.array_accounting("500", 2, attempts=3) == {}


def test_a_canceled_waiting_run_after_a_requeue_is_the_last_run():
    text = (f"500_0|501|REQUEUED|2026-10-21T10:00:00|300|{TRES}\n"
            "500_0|501|CANCELLED by 123|Unknown|0|\n")  # the queue limit canceled the requeued, waiting run
    task = parse_array_sacct(text)[0]
    assert task.state == "CANCELLED"
    assert task.gpu_minutes == 5.0 and task.runs == 1
