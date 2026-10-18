---
name: check-round
description: Use when a round of the pruning study has finished, before you trust its result. Runs the project's check command for one round and reports each difference between the metrics, the history row, the round log, and the checkpoint.
---

# Check a round

Run the check command for round `$ARGUMENTS`. If no round number is given, use the last round in `state/history.tsv`.

```bash
.venv/bin/python -m pruning_example.check_round $ARGUMENTS
```

The command compares:

- `state/rounds/round_NN/metrics.json` with the row of the round in `state/history.tsv`.
- The validation accuracy with the round log in `logs/`.
- The nonzero weights with a new count from the checkpoint.
- The nonzero weights with the count that the configuration gives.
- The pass decision with the accuracy threshold in `experiment.yaml`.

Report each line that starts with `DIFF`. If all lines start with `OK`, say that the records agree.

Do not change any file. If a check fails, do not try to repair the records. Tell the user which check failed.
