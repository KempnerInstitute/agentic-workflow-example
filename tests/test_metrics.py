"""Tests of the fixed evaluator."""

import pytest

from pruning_example import metrics
from pruning_example.model import EXCLUDABLE_LAYERS, PRUNABLE_WEIGHTS, build_model, prunable_layers


def test_prunable_weights_of_resnet18():
    model = build_model()
    layers = dict(prunable_layers(model))
    assert len(layers) == 21
    assert metrics.weight_counts(model)["total"] == PRUNABLE_WEIGHTS
    for name, count in EXCLUDABLE_LAYERS.items():
        assert layers[name].weight.numel() == count


def test_accuracy_threshold():
    target = {"val_accuracy": 0.905, "plateau_rounds": 3}
    assert metrics.passes(0.905, target)
    assert not metrics.passes(0.9049, target)


def test_sparse_size_is_bitmask_plus_fp16_values():
    assert metrics.sparse_size_bytes(16, 3) == 2 + 6
    assert metrics.sparse_size_bytes(17, 0) == 3


def test_wilson_interval_contains_the_accuracy():
    low, high = metrics.wilson_interval(0.9, 5000)
    assert low < 0.9 < high
    assert high - low == pytest.approx(0.0166, abs=0.001)
