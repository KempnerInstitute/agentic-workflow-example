#!/bin/bash
#SBATCH --job-name=prune-round
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=a100|h100|h200|rtx6000pro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Run one round. Submit from the project folder:
#   sbatch slurm/round.sbatch CONFIG [--round N] [--kind round|confirm] [--seed S]
# kempner_requeue can preempt the job. The job saves a checkpoint after each epoch
# and continues after a requeue. Keep --time equal to round.time_minutes in experiment.yaml.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster.env
echo "start $(date -Is) on $(hostname -s), restart ${SLURM_RESTART_COUNT:-0}"
exec .venv/bin/python -m pruning_example.run_round "$@"
