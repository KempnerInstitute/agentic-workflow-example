#!/bin/bash
#SBATCH --job-name=prune-round
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=a100|h100|h200|rtx6000pro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Run one round for each task of an array job. The loop scripts submit it:
#   sbatch --array=0-2%3 slurm/round_array.sbatch TASKS_FILE
# Each line of TASKS_FILE, with no header, has 4 fields separated by tabs:
#   configuration path, round number, kind (round or confirm), seed
# Task i runs line i+1. %j gives each task its own job ID, so each task has its own log.
# kempner_requeue can preempt a task. The task saves a checkpoint after each epoch and continues after a requeue.
# Keep --time equal to round.time_minutes in the experiment file.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster.env
IFS=$'\t' read -r config round kind seed < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
echo "start $(date -Is) on $(hostname -s), task ${SLURM_ARRAY_TASK_ID}, restart ${SLURM_RESTART_COUNT:-0}"
exec .venv/bin/python -m pruning_example.run_round "$config" --round "$round" --kind "$kind" --seed "$seed" --no-history
