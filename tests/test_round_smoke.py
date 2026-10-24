"""One small round on the CPU with generated images: the outputs and the records."""

import json

import torch

from pruning_example import check_round, records, run_round
from pruning_example.config import BASELINE_MODEL
from pruning_example.model import EXCLUDABLE_LAYERS, PRUNABLE_WEIGHTS, build_model
from pruning_example.pruning import keep_counts

CONFIG = {
    "sparsity": 0.9, "steps": 2, "finetune_epochs": 2, "lr": 0.01, "weight_decay": 0.0,
    "exclude_layers": ["conv1"], "reason": "test",
}


def test_round_writes_exact_records(project):
    (project / BASELINE_MODEL).parent.mkdir(parents=True)
    torch.save(build_model().state_dict(), project / BASELINE_MODEL)
    (project / "agent").mkdir()
    (project / "agent" / "next_config.json").write_text(json.dumps(CONFIG))

    run_round.main(["agent/next_config.json"])

    record = project / "state" / "rounds" / "round_01"
    result = records.read_json(record / "metrics.json")
    excluded = EXCLUDABLE_LAYERS["conv1"]
    assert result["total"] == PRUNABLE_WEIGHTS
    assert result["nonzero"] == excluded + keep_counts(PRUNABLE_WEIGHTS - excluded, 0.9, 2)[-1]
    assert 0.0 <= result["val_accuracy"] <= 1.0
    history = records.read_history(project / "state")
    assert [row["status"] for row in history] == ["ok"]
    assert records.read_json(record / "config.json") == CONFIG
    assert not records.integrity_problems(project)


def test_run_folders_do_not_clash(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)  # inside a job, run_folder reuses the folder of the job
    first = records.run_folder(tmp_path, "0123456789abcdef", "round")
    second = records.run_folder(tmp_path, "0123456789abcdef", "round")
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_a_requeued_job_gets_its_run_folder_again(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    first = records.run_folder(tmp_path, "0123456789abcdef", "round")
    assert first.name.endswith("_01234567_4242")
    assert records.run_folder(tmp_path, "0123456789abcdef", "round") == first


def test_check_round_finds_a_changed_record(project, capsys):
    test_round_writes_exact_records(project)
    log = project / "logs" / "prune-round_local.out"
    log.parent.mkdir(exist_ok=True)
    result = records.read_json(project / "state" / "rounds" / "round_01" / "metrics.json")
    log.write_text(f"val_accuracy={result['val_accuracy']:.4f}\n")
    assert check_round.main(["1"]) == 0

    result["nonzero"] += 1
    records.write_json(project / "state" / "rounds" / "round_01" / "metrics.json", result)
    assert check_round.main(["1"]) == 1
    assert "DIFF" in capsys.readouterr().out


def test_a_round_without_history_and_a_confirmation_with_seed_2(project):
    (project / BASELINE_MODEL).parent.mkdir(parents=True)
    torch.save(build_model().state_dict(), project / BASELINE_MODEL)
    (project / "agent").mkdir()
    (project / "agent" / "next_config.json").write_text(json.dumps(CONFIG))

    run_round.main(["agent/next_config.json", "--round", "3", "--no-history"])  # the multi-agent loop writes the row
    assert (project / "state" / "rounds" / "round_03" / "metrics.json").exists()
    assert records.read_history(project / "state") == []

    run_round.main(["state/rounds/round_03/config.json", "--round", "3", "--kind", "confirm", "--seed", "2"])
    result = records.read_json(project / "state" / "confirm" / "round_03_seed2" / "metrics.json")
    assert result["seed"] == 2 and result["kind"] == "confirm"
    assert not (project / "state" / "confirm" / "round_03").exists()  # seed 1 keeps its own folder
