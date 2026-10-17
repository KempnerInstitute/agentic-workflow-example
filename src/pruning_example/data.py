"""CIFAR-10 data: the shared testbed copy, the fixed validation split, and generated data for tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
FAKE = "fake"  # DATA_DIR=fake gives generated images, for tests without the real data


def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class TransformSubset(Dataset):
    """Some items of a dataset, with their own transform."""

    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        return self.transform(image), label


class FakeImages(Dataset):
    """Random images and labels. The tests and CI use this data."""

    def __init__(self, n: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 32, 32, generator=generator)
        self.labels = torch.randint(0, 10, (n,), generator=generator)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], int(self.labels[i])


def split_hash(val_indices) -> str:
    text = ",".join(str(i) for i in sorted(val_indices))
    return hashlib.sha256(text.encode()).hexdigest()


def load_split(path: Path) -> dict:
    with open(path) as f:
        split = json.load(f)
    if split_hash(split["val_indices"]) != split["sha256"]:
        raise ValueError(f"{path}: the hash does not match the validation indices")
    return split


def train_val_sets(data_dir: str, split_path: Path):
    """Return the training set and the validation set. The testbed folder stays read-only."""
    if data_dir == FAKE:
        return FakeImages(128, seed=1), FakeImages(64, seed=2)
    base = datasets.CIFAR10(data_dir, train=True, download=False)  # one integrity check of the files
    val = load_split(split_path)["val_indices"]
    val_set = set(val)
    train = [i for i in range(len(base)) if i not in val_set]
    return TransformSubset(base, train, train_transform()), TransformSubset(base, val, eval_transform())


def cifar_test_set(data_dir: str):
    """Return the 10,000 test images. Only final_eval.py uses them."""
    if data_dir == FAKE:
        return FakeImages(64, seed=3)
    return datasets.CIFAR10(data_dir, train=False, download=False, transform=eval_transform())


def worker_count() -> int:
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(0, min(8, cpus - 1))


def make_loader(dataset, batch_size: int, device: torch.device, shuffle: torch.Generator | None = None):
    """Return a data loader. A generator gives a training loader in shuffled order."""
    workers = 0 if isinstance(dataset, FakeImages) else worker_count()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle is not None,
        generator=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
