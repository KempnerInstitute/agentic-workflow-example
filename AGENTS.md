# Agent instructions

This project prunes ResNet-18 on CIFAR-10 on the Kempner AI cluster. Read this file before each task.

## All sessions

- **Goal.** Find the configuration with the fewest weights left that keeps the accuracy threshold.
- **Accuracy threshold.** Validation accuracy of at least 0.905 (`target.val_accuracy` in `experiment.yaml`). A round that keeps it passes.
- **Stop.** The loop script stops after `target.plateau_rounds` complete rounds in a row that find no passing model with fewer weights, or at a limit.
- **Prunable weights.** The 20 Conv2d layers and the Linear layer: 11,164,352 weights. `conv1` has 1,728 weights, and `fc` has 5,120. Excluded layers stay dense, and they still count.
- **Weights left.** Nonzero prunable weights divided by 11,164,352. With sparsity s and no excluded layers, a round keeps 11,164,352 - ceil(11,164,352 x s) weights.
- **A round.** The round job prunes in `steps` equal-ratio steps with global magnitude pruning, and it fine-tunes after each step. The learning rate restarts at `lr` in each fine-tuning block and decays with a cosine curve.
- **Files to read.** The experiment file (`experiment.yaml`, or the file that your prompt names), `configs/round.schema.json`, `state/baseline/metrics.json`, and `history.tsv` and `notes.md` in the state folder (`state/`, or the folder that your prompt names).
- **Data.** Use only validation results. The test set is for the final steps only.
- **Files.** Do not change tracked files. The round job stops if a tracked file changed.

## Supervised round

- Write the configuration only to `agent/next_config.json`. Use the keys and ranges in `configs/round.schema.json`.
- Submit the round as a background command: `sbatch --wait slurm/round.sbatch agent/next_config.json`.
- Run SLURM commands on their own, without pipes, redirects, `cd`, or `$(...)`. The sandbox needs this.
- Query only your own jobs, with `squeue -j <id>` and `sacct -j <id>`. Do not run `squeue` without `-j`.
- To check a finished round, run `/check-round N`.
- If a task needs more access, stop and ask.

## Agent step in the loop

- Return the next configuration only in the required JSON format, with a short reason.
- Do not try to run commands or write files. The loop script runs the round.
- Do not repeat a configuration that already ran. Learn from the rows and notes of the earlier rounds.

## Roles in the multi-agent loop

- Each role step has a role prompt from `prompts/multi/roles/`: the planner, the critic, the writer, or the auditor. Follow it.
- The multi-agent loop uses `experiment_multi.yaml` and the state folder `state_multi/`. Read only the state folder that your prompt names, and `state/baseline/metrics.json`.
- A wave has up to 3 rounds that run at the same time. The loop script stops after `target.plateau_rounds` complete rounds in a row, in round order, without a smaller passing model.
- Return only the required JSON or text. Do not try to run commands or write files. The loop script runs the rounds.
