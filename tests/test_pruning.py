"""Tests of the pruning schedule: step sizes, exact weight counts, and permanent zeros."""

import pytest

from pruning_example import metrics
from pruning_example.model import EXCLUDABLE_LAYERS, PRUNABLE_WEIGHTS, build_model
from pruning_example.pruning import epoch_blocks, keep_counts, make_permanent, prune_to, pruned_layers, unpruned


def test_keep_counts_are_exact_and_do_not_increase():
    assert keep_counts(1000, 0.9, 3) == [464, 215, 100]
    assert keep_counts(1000, 0.9, 1) == [100]
    counts = keep_counts(PRUNABLE_WEIGHTS, 0.98, 10)
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == PRUNABLE_WEIGHTS - 10_941_065  # ceil(0.98 * 11,164,352) weights removed


@pytest.mark.parametrize("epochs, steps, blocks", [(10, 1, [10]), (10, 3, [3, 3, 4]), (5, 5, [1, 1, 1, 1, 1])])
def test_epoch_blocks(epochs, steps, blocks):
    assert epoch_blocks(epochs, steps) == blocks
    assert sum(epoch_blocks(epochs, steps)) == epochs


def test_schedule_reaches_each_count_exactly():
    model = build_model()
    layers = pruned_layers(model, ["conv1"])
    total = sum(module.weight.numel() for _, module in layers)
    assert total == PRUNABLE_WEIGHTS - EXCLUDABLE_LAYERS["conv1"]
    keeps = keep_counts(total, 0.98, 3)
    for keep in keeps:
        prune_to(layers, keep)
        assert unpruned(layers) == keep
    make_permanent(layers)
    counts = metrics.weight_counts(model)
    assert counts["nonzero"] == keeps[-1] + EXCLUDABLE_LAYERS["conv1"]
    assert counts["layer_density"]["conv1"] == 1.0
    assert not any(hasattr(module, "weight_mask") for _, module in layers)
