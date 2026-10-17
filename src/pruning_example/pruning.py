"""Global magnitude pruning in equal-ratio steps, with fine-tuning after each step."""

from __future__ import annotations

import math
import os
from fractions import Fraction
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune

from .model import prunable_layers


def pruned_layers(model, exclude) -> list[tuple[str, nn.Module]]:
    """Return the prunable layers that the configuration does not exclude."""
    excluded = set(exclude)
    return [(name, module) for name, module in prunable_layers(model) if name not in excluded]


def keep_counts(total: int, sparsity: float, steps: int) -> list[int]:
    """Return the weights to keep after each step: floor(total * (1 - sparsity) ** (i / steps)).

    The last count is exact: total - ceil(total * sparsity), with the sparsity read as a decimal.
    A small tolerance protects the other counts from floating-point error.
    """
    counts = [math.floor(total * (1 - sparsity) ** (i / steps) + 1e-9) for i in range(1, steps)]
    counts.append(total - math.ceil(total * Fraction(str(sparsity))))
    for i in range(len(counts) - 2, -1, -1):
        counts[i] = max(counts[i], counts[i + 1])
    return counts


def epoch_blocks(epochs: int, steps: int) -> list[int]:
    """Divide the epochs into one block for each step. The last block gets the extra epochs."""
    base = epochs // steps
    return [base] * (steps - 1) + [epochs - base * (steps - 1)]


def unpruned(layers) -> int:
    """Return the number of weights that no mask removes."""
    return sum(
        int(module.weight_mask.sum()) if hasattr(module, "weight_mask") else module.weight.numel()
        for _, module in layers
    )


def prune_to(layers, keep: int) -> None:
    """Remove the smallest-magnitude weights over all layers together, until `keep` weights stay."""
    remove = unpruned(layers) - keep
    if remove <= 0:
        return
    scores = {}
    for _, module in layers:
        # Use weight_orig times the mask: module.weight can be older than the last optimizer step.
        weight = module.weight_orig * module.weight_mask if hasattr(module, "weight_mask") else module.weight
        scores[(module, "weight")] = weight.detach().abs()
    prune.global_unstructured(
        [(module, "weight") for _, module in layers],
        pruning_method=prune.L1Unstructured,
        importance_scores=scores,
        amount=remove,
    )


def add_masks(layers) -> None:
    """Add masks of ones, so that a checkpoint with masks can load into the model."""
    for _, module in layers:
        if not hasattr(module, "weight_mask"):
            prune.identity(module, "weight")


def make_permanent(layers) -> None:
    """Make the zeros permanent and remove the masks. Call this only after the last fine-tuning."""
    for _, module in layers:
        if hasattr(module, "weight_mask"):
            prune.remove(module, "weight")


def optimizer_and_schedule(model, lr, weight_decay, momentum, total_steps, warmup_steps=0):
    """Return SGD with Nesterov momentum, and a linear warmup followed by a cosine decay to zero."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay
    )

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, loader, optimizer, schedule, device: torch.device) -> float:
    """Train for one epoch and return the mean loss."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = torch.zeros((), device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = loss_fn(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        total += loss.detach()
    return float(total) / max(1, len(loader))


def save_atomic(obj, path: Path) -> None:
    """Write to a temporary file, then rename it, so a stopped job cannot leave a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
