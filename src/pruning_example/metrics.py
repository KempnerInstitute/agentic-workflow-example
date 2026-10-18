"""The fixed evaluator: accuracy, weights left, layer density, and sparse size.

The agent cannot change this file. The round job and check_round.py use it.
"""

from __future__ import annotations

import math

import torch

from .model import prunable_layers


def effective_weight(module) -> torch.Tensor:
    """Return the weight that the layer uses: the original weight times its mask, if it has a mask."""
    if hasattr(module, "weight_mask"):
        return (module.weight_orig * module.weight_mask).detach()
    return module.weight.detach()


def weight_counts(model) -> dict:
    """Count the nonzero prunable weights, in total and for each layer."""
    layers = {}
    total = nonzero = 0
    for name, module in prunable_layers(model):
        weight = effective_weight(module)
        count = int(torch.count_nonzero(weight))
        layers[name] = round(count / weight.numel(), 6)
        total += weight.numel()
        nonzero += count
    return {"total": total, "nonzero": nonzero, "weights_left": nonzero / total, "layer_density": layers}


def passes(val_accuracy: float, target: dict) -> bool:
    """Return True if a model keeps the accuracy threshold."""
    return val_accuracy >= target["val_accuracy"]


def sparse_size_bytes(total: int, nonzero: int) -> int:
    """Return the size of a bitmask for all weights plus fp16 values for the nonzero weights."""
    return math.ceil(total / 8) + 2 * nonzero


def wilson_interval(p: float, n: int, z: float = 1.96) -> list[float]:
    """Return the 95% Wilson interval of an accuracy p measured on n images."""
    if n == 0:
        return [0.0, 1.0]
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(center - half, 5), round(center + half, 5)]


@torch.no_grad()
def accuracy(model, loader, device: torch.device) -> float:
    """Return the fraction of correct predictions. No augmentation."""
    model.eval()
    correct = seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += labels.numel()
    return correct / seen
