"""Check the records of one round against its log and its checkpoint.

Usage: python -m pruning_example.check_round N
The check-round skill runs this command. The exit status is 0 only if all checks pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from . import config, metrics, records
from .model import EXCLUDABLE_LAYERS, PRUNABLE_WEIGHTS, build_model
from .pruning import keep_counts


def run_checks(round_no: int) -> list[tuple[str, bool, str]]:
    paths = config.paths_from_env()
    target = config.load_experiment(paths.project)["target"]
    folder = paths.state / "rounds" / f"round_{round_no:02d}"
    result = records.read_json(folder / "metrics.json")
    meta = records.read_json(folder / "run_meta.json")
    round_config = records.read_json(folder / "config.json")
    checks = []

    rows = [row for row in records.read_history(paths.state) if int(row["round"]) == round_no]
    checks.append(("one row in history.tsv", len(rows) == 1, f"{len(rows)} rows"))
    if rows:
        row = rows[0]
        checks.append(("history.tsv: validation accuracy",
                       row["val_accuracy"] == f"{result['val_accuracy']:.4f}", row["val_accuracy"]))
        checks.append(("history.tsv: nonzero weights", int(row["nonzero"]) == result["nonzero"], row["nonzero"]))

    logs = sorted(paths.logs.glob(f"*_{meta['job_id']}.out"))
    checks.append(("round log found", bool(logs), ", ".join(log.name for log in logs) or "no log"))
    if logs:
        expected = f"val_accuracy={result['val_accuracy']:.4f}"
        checks.append(("round log: validation accuracy", expected in logs[0].read_text(errors="replace"), expected))

    model = build_model()
    model.load_state_dict(torch.load(Path(meta["run_folder"]) / "model.pt", map_location="cpu"))
    counts = metrics.weight_counts(model)
    checks.append(("checkpoint: nonzero weights", counts["nonzero"] == result["nonzero"], str(counts["nonzero"])))

    excluded = sum(EXCLUDABLE_LAYERS[name] for name in round_config["exclude_layers"])
    expected_nonzero = excluded + keep_counts(PRUNABLE_WEIGHTS - excluded, round_config["sparsity"],
                                              round_config["steps"])[-1]
    checks.append(("configuration: expected nonzero weights", counts["nonzero"] == expected_nonzero,
                   f"expected {expected_nonzero}"))
    decision = metrics.meets_target(result["val_accuracy"], counts["nonzero"], target)
    checks.append(("target decision", decision == result["meets_target"], f"meets_target={decision}"))
    return checks


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or not argv[0].isdigit():
        print("Usage: python -m pruning_example.check_round N")
        return 2
    checks = run_checks(int(argv[0]))
    for name, ok, detail in checks:
        print(f"{'OK  ' if ok else 'DIFF'}  {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
