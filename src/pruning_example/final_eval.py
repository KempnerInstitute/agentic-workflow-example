"""One-time test evaluation of one model.

Usage: python -m pruning_example.final_eval --round N [--kind round|confirm]

The command refuses a second evaluation of the same model, and it logs each call.
The loop script runs this command in the final steps. The agent cannot run it.
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
    args = parser.parse_args(argv)

    paths = config.paths_from_env()
    paths.logs.mkdir(parents=True, exist_ok=True)
    with open(paths.logs / "final_eval.log", "a") as log:
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} job={records.job_id()} "
                  f"kind={args.kind} round={args.round}\n")
    out = paths.state / "test_eval" / f"{args.kind}_{args.round:02d}.json"
    if out.exists():
        print(f"Refused: {out} exists. The test set evaluates each model one time only.")
        return 2

    record_dir = paths.state / ("rounds" if args.kind == "round" else "confirm") / f"round_{args.round:02d}"
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
    records.write_json(out, result)
    print(f"{args.kind} round {args.round}: test_accuracy={test_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
