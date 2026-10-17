"""Tests of the round configuration schema and its extra rule."""

import json

import pytest

from pruning_example.config import ConfigError, load_experiment, load_schema, validate_round_config
from conftest import REPO

SCHEMA = load_schema(REPO)
EXAMPLE = json.loads((REPO / "configs" / "round_example.json").read_text())


def changed(**values):
    return {**EXAMPLE, **values}


def test_example_configuration_is_valid():
    assert validate_round_config(dict(EXAMPLE), SCHEMA) == EXAMPLE


@pytest.mark.parametrize("bad", [
    changed(sparsity=1.5),
    changed(sparsity=0.9995),
    changed(sparsity=0.2),
    changed(steps=0),
    changed(finetune_epochs=21),
    changed(lr=0.5),
    changed(exclude_layers=["conv1", "conv1"]),
    changed(exclude_layers=["layer1"]),
    changed(seed=3),
    changed(reason="x" * 601),
])
def test_schema_rejects_bad_values(bad):
    with pytest.raises(ConfigError):
        validate_round_config(bad, SCHEMA)


def test_missing_key_is_rejected():
    bad = dict(EXAMPLE)
    del bad["reason"]
    with pytest.raises(ConfigError):
        validate_round_config(bad, SCHEMA)


def test_finetune_epochs_must_be_at_least_steps():
    with pytest.raises(ConfigError, match="at least steps"):
        validate_round_config(changed(steps=5, finetune_epochs=4), SCHEMA)


def test_experiment_file_can_point_to_a_test_copy(project, monkeypatch):
    test_copy = project / "calibration" / "small_limits.yaml"
    test_copy.parent.mkdir()
    test_copy.write_text((project / "experiment.yaml").read_text().replace("tokens: 1000000", "tokens: 100000"))
    assert load_experiment(project)["budget"]["tokens"] == 1_000_000
    monkeypatch.setenv("EXPERIMENT_FILE", "calibration/small_limits.yaml")
    assert load_experiment(project)["budget"]["tokens"] == 100_000
