"""One-time test evaluation of one model.

Usage: python -m pruning_example.final_eval --round N [--kind round|confirm] [--seed S]

The command refuses a second evaluation of the same model, and it logs each call.
The loop script runs this command in the final steps. The agent cannot run it.
--seed selects a confirmation round with seed 2 or more; seed 1 keeps the paths of a single confirmation round.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from . import config, data, metrics, records
from .model import build_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one model on the test set, one time.")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--kind", choices=["round", "confirm"], default="round")
    parser.add_argument("--seed", type=int, default=None, help="seed of a confirmation round (default: 1)")
    args = parser.parse_args(argv)

    paths = config.paths_from_env()
    paths.logs.mkdir(parents=True, exist_ok=True)
    with open(paths.logs / "final_eval.log", "a") as log:
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} job={records.job_id()} "
                  f"kind={args.kind} round={args.round}{'' if args.seed is None else f' seed={args.seed}'}\n")
    out = records.test_eval_path(paths.state, args.kind, args.round, args.seed)
    if out.exists():
        print(f"Refused: {out} exists. The test set evaluates each model one time only.")
        return 2

    record_dir = records.record_dir(paths.state, args.kind, args.round, args.seed)
    meta = records.read_json(record_dir / "run_meta.json")
    torch.set_num_threads(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1)
    model = build_model()
    model.load_state_dict(torch.load(Path(meta["run_folder"]) / "model.pt", map_location="cpu", weights_only=True))
    test = data.cifar_test_set(paths.data_dir)
    test_accuracy = metrics.accuracy(model, data.make_loader(test, 500, torch.device("cpu")), torch.device("cpu"))
    result = {
        "kind": args.kind,
        "round": args.round,
        "test_accuracy": test_accuracy,
        "test_interval": metrics.wilson_interval(test_accuracy, len(test)),
        "test_images": len(test),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.seed is not None:
        result["seed"] = args.seed
    records.write_json(out, result)
    print(f"test evaluation, {args.kind} {args.round}: test_accuracy={test_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
