#!/bin/bash
#SBATCH --job-name=prune-multi-loop
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/%x_%j.out

# Run the multi-agent loop, then the final steps. Submit from the project folder, on a login node:
#   sbatch --time=01:30:00 slurm/multi_loop.sbatch --plan-only   # plan wave 1, then stop so that you can read it
#   sbatch slurm/multi_loop.sbatch                              # wave 1 and the next waves, then the final steps
#   sbatch slurm/multi_loop.sbatch --finish                     # only the final steps, after an early stop
#   sbatch slurm/multi_loop.sbatch --audit-only                 # the auditor and check_report, for STATE_DIR
# Read experiment_multi.yaml first: it sets the roles, the models, the effort levels, and the budget of this job.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset ANTHROPIC_API_KEY CLAUDE_CODE_EFFORT_LEVEL   # print mode uses your Harvard login; the experiment file sets the effort
source cluster.env
export STATE_DIR="${STATE_DIR:-state_multi}"
export EXPERIMENT_FILE="${EXPERIMENT_FILE:-experiment_multi.yaml}"
export PATH="$HOME/.local/bin:$PATH"
CLAUDE_VERSION="$(claude --version)"
export CLAUDE_VERSION
echo "start $(date -Is) on $(hostname -s), ${CLAUDE_VERSION}, state folder ${STATE_DIR}, experiment file ${EXPERIMENT_FILE}"
exec .venv/bin/python -m pruning_example.multi_loop "$@"
