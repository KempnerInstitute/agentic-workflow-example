"""Write data/split.json: a stratified validation split with 500 images for each class.

Run this command one time, on a compute node: python -m pruning_example.make_split
"""

from __future__ import annotations

import numpy as np
from torchvision import datasets

from . import config, data, records


def main() -> None:
    paths = config.paths_from_env()
    labels = np.array(datasets.CIFAR10(paths.data_dir, train=True, download=False).targets)
    rng = np.random.default_rng(0)
    val = []
    for label in range(10):
        val.extend(rng.choice(np.flatnonzero(labels == label), size=500, replace=False).tolist())
    val = sorted(int(i) for i in val)
    split = {"seed": 0, "per_class": 500, "sha256": data.split_hash(val), "val_indices": val}
    records.write_json(paths.project / config.SPLIT_PATH, split, indent=None)
    print(f"wrote {config.SPLIT_PATH}: {len(val)} validation images, sha256 {split['sha256'][:12]}")


if __name__ == "__main__":
    main()
