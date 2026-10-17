"""ResNet-18 for CIFAR-10, and the layers that a round can prune."""

from __future__ import annotations

from torch import nn
from torchvision.models import resnet18

NUM_CLASSES = 10
PRUNABLE_WEIGHTS = 11_164_352  # the weights of the 20 Conv2d layers and the Linear layer
EXCLUDABLE_LAYERS = {"conv1": 1_728, "fc": 5_120}


def build_model() -> nn.Module:
    """Return ResNet-18 for 32x32 images: a 3x3 first convolution and no max pool."""
    model = resnet18(weights=None, num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def prunable_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return the Conv2d and Linear layers in model order. Biases and BatchNorm are not prunable."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
