"""Entry point of the baseline job: train the dense model one time.

The job saves a checkpoint after each epoch. After a requeue, the job continues from the checkpoint.
"""

from __future__ import annotations

import time

import torch

from . import config, data, metrics, records
from .model import build_model
from .pruning import optimizer_and_schedule, save_atomic, train_epoch


def main() -> None:
    paths = config.paths_from_env()
    settings = config.load_experiment(paths.project)["baseline"]
    records.check_integrity(paths.project)
    commit = records.commit_hash(paths.project)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = paths.project / config.SPLIT_PATH
    train, val = data.train_val_sets(paths.data_dir, split_path)
    split_sha = "fake" if paths.data_dir == data.FAKE else data.load_split(split_path)["sha256"]
    shuffle = torch.Generator()
    train_loader = data.make_loader(train, settings["batch_size"], device, shuffle)
    val_loader = data.make_loader(val, 512, device)

    folder = records.run_folder(paths.run_root, commit, "baseline")
    checkpoint = folder / "checkpoint.pt"
    model = build_model().to(device)
    per_epoch = len(train_loader)
    optimizer, schedule = optimizer_and_schedule(
        model, settings["lr"], settings["weight_decay"], settings["momentum"],
        settings["epochs"] * per_epoch, settings["warmup_epochs"] * per_epoch,
    )
    start = 0
    if checkpoint.exists():  # SLURM requeued the job
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"])
        start = saved["epoch"]
        print(f"resumed from epoch {start}", flush=True)

    meta = records.run_meta(commit, 0, split_sha, "baseline", folder)
    began = time.time()
    for epoch in range(start, settings["epochs"]):
        shuffle.manual_seed(epoch)
        loss = train_epoch(model, train_loader, optimizer, schedule, device)
        save_atomic(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": schedule.state_dict(), "epoch": epoch + 1},
            checkpoint,
        )
        print(f"epoch {epoch + 1}/{settings['epochs']} loss={loss:.4f}", flush=True)

    val_accuracy = metrics.accuracy(model, val_loader, device)
    save_atomic(model.state_dict(), paths.project / config.BASELINE_MODEL)
    result = {
        "val_accuracy": val_accuracy,
        "val_interval": metrics.wilson_interval(val_accuracy, len(val)),
        "val_images": len(val),
        "epochs": settings["epochs"],
        "train_minutes_this_run": round((time.time() - began) / 60, 2),
        **metrics.weight_counts(model),
    }
    meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    for target in (folder, paths.project / "state" / "baseline"):
        records.write_json(target / "metrics.json", result)
        records.write_json(target / "run_meta.json", meta)
    checkpoint.unlink(missing_ok=True)
    print(f"val_accuracy={val_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
