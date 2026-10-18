# Agentic Workflow Example

A Claude Code agent prunes a ResNet-18 model for CIFAR-10 on the Kempner AI cluster. The goal is the smallest model that keeps a test accuracy of at least 90%. The chapter "End-to-End Agentic Workflow Example" of the [Kempner Institute Computing Handbook](https://handbook.eng.kempnerinstitute.harvard.edu/) describes this example step by step.

## What the example does

1. You train the dense baseline one time.
2. You supervise round 1 in an interactive Claude Code session. You approve each agent action.
3. You start the loop job. For each round, the loop script asks the agent for the next configuration, checks the configuration, submits one round job, and records the result.
4. The loop script stops when 3 rounds in a row find no smaller model that keeps the accuracy threshold, or at a limit. Then it runs the final steps: a confirmation round with a second seed, a one-time test evaluation, a summary, and a draft report.
5. You review the result.

In the unattended phase, the agent has read-only tools. It cannot run commands, write files, or submit jobs. The loop script, which is committed code, does these tasks.

## Before you start

- A Kempner AI cluster account, and a SLURM account that can use `kempner_requeue` and `shared`.
- Claude Code, signed in with the Claude access that HUIT provides through Harvard SSO. Leave `ANTHROPIC_API_KEY` unset.
- [uv](https://docs.astral.sh/uv/).
- The data is the shared copy of CIFAR-10 on the cluster. CIFAR-10 is public data (Level 1).

## Set up

Clone the repository into lab storage, set your values, and build the environment on a compute node:

```bash
cd /n/holylabs/LABS/<lab_name>/Users/<username>
git clone https://github.com/KempnerInstitute/agentic-workflow-example.git
cd agentic-workflow-example
cp cluster.env.example cluster.env      # set your account and your paths
source cluster.env
mkdir -p logs && chmod 700 logs          # the hook writes the command log here
salloc --partition=test --time=0-01:00 --mem=16G --cpus-per-task=4
uv sync --locked --extra gpu             # about 5 GB of packages
exit
```

`cluster.env` puts the uv cache and the Python interpreter in lab storage. The agent sandbox hides your home folder, so `.venv/bin/python` must not link to a Python in your home folder.

Train the baseline:

```bash
source cluster.env
sbatch slurm/baseline.sbatch             # about 2 minutes on one GPU
```

`state/baseline/metrics.json` then shows a validation accuracy of about 0.94.

## Round 1 (supervised)

1. On a login node, start tmux. In tmux, run `salloc --partition=test --time=0-02:00 --mem=16G --cpus-per-task=4`.
2. In the project folder, run `source cluster.env`. Read `AGENTS.md`, `.claude/settings.json`, `.claude/hooks/guard.sh`, the skills, and the job scripts.
3. Run `claude --permission-mode default`. Accept the trust prompt only after step 2. Run `/status` and check the organization.
4. Give this prompt: "Read AGENTS.md and state/baseline/metrics.json. Propose a cautious first round that tests the pipeline, explain it, and write it to agent/next_config.json. Then run `sbatch --wait slurm/round.sbatch agent/next_config.json` as a background command."
5. Approve the write. Read the `sbatch` command, then approve it.
6. When the job ends, type `/check-round 1`. Compare the numbers with `state/rounds/round_01/metrics.json`.
7. Type `/exit`, and end the interactive job with `exit`.

## The unattended phase

Read `experiment.yaml` first: it sets the budget. Then start the loop job on a login node, in the project folder:

```bash
source cluster.env
sbatch slurm/loop.sbatch
```

Watch the run with `tail -f logs/prune-loop_<job_id>.out` and `state/history.tsv`. To stop the run, run `scancel <job_id>` on the loop job. The loop script cancels its own round job and writes `state/summary.json`. To run the final steps later, run `sbatch slurm/loop.sbatch --finish`.

## Limits

| Limit | Value |
|---|---|
| Accuracy threshold | Validation accuracy of at least 0.905. The loop looks for the fewest weights that keep it, and stops after 3 rounds in a row without a smaller passing model. |
| Tokens | 1,000,000 for the unattended phase, with 10% kept for the report step. All four token types count, cache reads too. |
| Agent step | `--max-turns 8`, `--max-budget-usd 1.00`, read-only tools, `dontAsk` |
| Rounds | 15 round attempts, round 1 included |
| GPU | 60 GPU minutes for the unattended phase. One round job at a time, 1 GPU, 10 minutes at most. |
| Queue | A job can wait at most 45 minutes in the queue |
| Loop job | `shared` partition, 2 CPUs, 8 GB, 6 hours |

## Results

| File | Content |
|---|---|
| `state/summary.json` | Stop reason, totals, the selected round, both seeds, validation and test accuracy with 95% intervals |
| `state/final_report.md` | The draft report of the agent. Check each number against the other files. |
| `state/history.tsv` | One row for each round attempt |
| `state/ledger.tsv` | Tokens of each agent step, and GPU minutes of each job |
| `state/rounds/round_NN/` | The configuration, the metrics, and the run record of each round |
| `results/state/round_NN_model.pt` | The selected model |

The round checkpoints are in `RUN_ROOT` on scratch storage. FASRC deletes scratch files after 90 days, so the loop script copies the selected model to `results/`.

## Guardrails

| Guardrail | Supervised round | Unattended phase |
|---|---|---|
| Permission mode | `default` (manual) | `dontAsk` |
| Tools | All built-in tools | Read, Grep, and Glob; no MCP tools |
| Writes | The agent asks first. Deny rules block edits of tracked files and output folders. | None |
| Shell | Sandbox on, SLURM commands outside it, `scancel` and `rm -r` blocked by a hook | No shell |
| Job submission | `sbatch` asks | The loop script submits |
| Reads | Only the project folder | The same |
| Integrity | The round job and the loop script stop if a tracked file changed | The same |

## Test the loop with small limits

Put test copies in `local/`, which Git ignores. `EXPERIMENT_FILE` and `STATE_DIR` point the jobs to them:

```bash
mkdir -p local
cp experiment.yaml local/small.yaml       # then lower the limits in the copy
STATE_DIR=local/state_small EXPERIMENT_FILE=local/small.yaml sbatch slurm/loop.sbatch
```

The loop needs a round 1 in the state folder. `state/summary.json` records the experiment file and its hash.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests use generated images, so they need no GPU and no data. CI runs them with the `cpu` extra.

## Files

```text
AGENTS.md, CLAUDE.md      agent instructions
experiment.yaml           target, recipe, and limits
cluster.env.example       your account and paths
.claude/                  settings, hook, and skills
configs/                  round configuration schema and an example
prompts/                  prompts of the agent step and the report step
data/split.json           fixed validation split
slurm/                    baseline, round, and loop jobs
src/pruning_example/      code: data, model, evaluator, pruning, records, loop, final steps
tests/                    pytest tests
```

## Tested with

Claude Code 2.1.302, uv 0.8.19, Python 3.12.11, PyTorch 2.14.1 with CUDA 13.0, torchvision 0.29.1, and NVIDIA driver 610.57.04, on A100, H100, H200, and RTX PRO 6000 GPUs, in October 2026.

## Adapt the example

- **Change** the task: the data, the model, the metric, the round configuration schema, the target, and the limits.
- **Keep** the fixed evaluator, the read-only agent step, the loop script that submits the jobs, the limits outside the model, the run records, the confirmation round, and the one-time test evaluation.

## Sources

The skills in `.claude/skills/reproducible-research/` and `.claude/skills/testing-and-continuous-integration/`, and the hook `.claude/hooks/guard.sh`, come from the Kempner Institute Computing Handbook.

## License

MIT. Copyright (c) 2026 Harvard University. See `LICENSE`. To cite this example, see `CITATION.cff`.
