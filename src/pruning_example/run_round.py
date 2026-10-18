"""Entry point of the round job: prune, fine-tune, and evaluate one configuration.

Usage: python -m pruning_example.run_round CONFIG [--round N] [--kind round|confirm] [--seed S]

The job saves a checkpoint after each epoch. After a requeue, the job continues from the checkpoint.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

from . import config, data, metrics, records
from .model import build_model
from .pruning import (
    add_masks, epoch_blocks, keep_counts, make_permanent, optimizer_and_schedule,
    prune_to, pruned_layers, save_atomic, train_epoch, unpruned,
)


def parse(argv=None):
    parser = argparse.ArgumentParser(description="Prune, fine-tune, and evaluate one configuration.")
    parser.add_argument("config", type=Path, help="JSON file with one round configuration")
    parser.add_argument("--round", type=int, default=None, help="round number (default: the next round)")
    parser.add_argument("--kind", choices=["round", "confirm"], default="round")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def finish_records(paths, args, round_no, record_dir, round_config) -> None:
    """Make sure that history.tsv has the row of a complete round. A requeue can stop a job between steps."""
    if args.kind != "round":
        return
    if any(int(row["round"]) == round_no for row in records.read_history(paths.state)):
        return
    result = records.read_json(record_dir / "metrics.json")
    records.append_history(paths.state, records.history_row(round_no, "ok", round_config, result))


def main(argv=None) -> None:
    args = parse(argv)
    paths = config.paths_from_env()
    experiment = config.load_experiment(paths.project)
    settings = experiment["round"]
    records.check_integrity(paths.project)
    commit = records.commit_hash(paths.project)
    folder = records.run_folder(paths.run_root, commit, args.kind)
    if (folder / "config.json").exists():  # SLURM requeued the job: keep the first configuration and number
        round_config = records.read_json(folder / "config.json")
        round_no = records.read_json(folder / "round.json")["round"]
    else:
        round_config = config.load_round_config(args.config, config.load_schema(paths.project))
        round_no = args.round or records.next_round(paths.state)
        records.write_json(folder / "round.json", {"round": round_no, "kind": args.kind})
        records.write_json(folder / "config.json", round_config)
    record_dir = paths.state / ("rounds" if args.kind == "round" else "confirm") / f"round_{round_no:02d}"
    if (record_dir / "metrics.json").exists():
        done = records.read_json(record_dir / "run_meta.json")
        if done["job_id"] != records.job_id():
            raise SystemExit(f"{record_dir} already has a result from job {done['job_id']}")
        finish_records(paths, args, round_no, record_dir, round_config)
        print(f"round {round_no} is already complete", flush=True)
        return

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = paths.project / config.SPLIT_PATH
    train, val = data.train_val_sets(paths.data_dir, split_path)
    split_sha = "fake" if paths.data_dir == data.FAKE else data.load_split(split_path)["sha256"]
    shuffle = torch.Generator()
    train_loader = data.make_loader(train, settings["batch_size"], device, shuffle)
    val_loader = data.make_loader(val, 512, device)

    model = build_model()
    model.load_state_dict(torch.load(paths.project / config.BASELINE_MODEL, map_location="cpu"))
    model.to(device)
    layers = pruned_layers(model, round_config["exclude_layers"])
    total = sum(module.weight.numel() for _, module in layers)
    keeps = keep_counts(total, round_config["sparsity"], round_config["steps"])
    blocks = epoch_blocks(round_config["finetune_epochs"], round_config["steps"])

    def new_optimizer(block: int):
        return optimizer_and_schedule(
            model, round_config["lr"], round_config["weight_decay"], settings["momentum"],
            blocks[block] * len(train_loader),
        )

    checkpoint = folder / "checkpoint.pt"
    block, done, optimizer, schedule = 0, 0, None, None
    if checkpoint.exists():  # SLURM requeued the job
        saved = torch.load(checkpoint, map_location=device)
        add_masks(layers)
        model.load_state_dict(saved["model"])
        block, done = saved["block"], saved["done"]
        if done < blocks[block]:
            optimizer, schedule = new_optimizer(block)
            optimizer.load_state_dict(saved["optimizer"])
            schedule.load_state_dict(saved["scheduler"])
        else:
            block, done = block + 1, 0
        print(f"resumed at step {block + 1}, epoch {done}", flush=True)

    meta = records.run_meta(commit, args.seed, split_sha, args.kind, folder)
    began = time.time()
    while block < round_config["steps"]:
        if optimizer is None:
            prune_to(layers, keeps[block])
            optimizer, schedule = new_optimizer(block)
        for epoch in range(done, blocks[block]):
            shuffle.manual_seed(100_000 * args.seed + 1_000 * block + epoch)
            loss = train_epoch(model, train_loader, optimizer, schedule, device)
            save_atomic(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": schedule.state_dict(), "block": block, "done": epoch + 1},
                checkpoint,
            )
            print(f"step {block + 1}/{round_config['steps']} epoch {epoch + 1}/{blocks[block]} "
                  f"loss={loss:.4f} kept={unpruned(layers)}", flush=True)
        block, done, optimizer, schedule = block + 1, 0, None, None

    make_permanent(layers)
    counts = metrics.weight_counts(model)
    val_accuracy = metrics.accuracy(model, val_loader, device)
    target = experiment["target"]
    result = {
        "round": round_no,
        "kind": args.kind,
        "seed": args.seed,
        "val_accuracy": val_accuracy,
        "val_interval": metrics.wilson_interval(val_accuracy, len(val)),
        "val_images": len(val),
        **counts,
        "passes": metrics.passes(val_accuracy, target),
        "sparse_size_mb": round(metrics.sparse_size_bytes(counts["total"], counts["nonzero"]) / 1e6, 3),
        "dense_size_mb": round(4 * counts["total"] / 1e6, 3),
        "finetune_minutes_this_run": round((time.time() - began) / 60, 2),
    }
    save_atomic(model.state_dict(), folder / "model.pt")
    meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    records.write_json(folder / "metrics.json", result)
    records.write_json(folder / "run_meta.json", meta)
    record_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "run_meta.json", "metrics.json"):
        shutil.copyfile(folder / name, record_dir / name)
    finish_records(paths, args, round_no, record_dir, round_config)
    checkpoint.unlink(missing_ok=True)
    print(f"round {round_no}: val_accuracy={val_accuracy:.4f} weights_left={counts['weights_left']:.6f} "
          f"nonzero={counts['nonzero']} passes={result['passes']}", flush=True)


if __name__ == "__main__":
    main()
