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
