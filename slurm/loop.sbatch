#!/bin/bash
#SBATCH --job-name=prune-loop
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/%x_%j.out

# Run the unattended phase, then the final steps. Submit from the project folder, on a login node:
#   sbatch slurm/loop.sbatch            # the rounds, then the final steps
#   sbatch slurm/loop.sbatch --finish   # only the final steps, after an early stop
# Read experiment.yaml first: it sets the budget of this job.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset ANTHROPIC_API_KEY   # print mode must use your Harvard login, not a personal API key
source cluster.env
export PATH="$HOME/.local/bin:$PATH"
CLAUDE_VERSION="$(claude --version)"
export CLAUDE_VERSION
echo "start $(date -Is) on $(hostname -s), Claude Code ${CLAUDE_VERSION}"
exec .venv/bin/python -m pruning_example.loop "$@"
