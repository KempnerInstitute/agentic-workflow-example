# Agentic Workflow Example

A Claude Code agent prunes a ResNet-18 model for CIFAR-10 on the Kempner AI cluster. The goal is the smallest model that keeps a test accuracy of at least 90%. The chapter [End-to-End Agentic Workflow Example](https://handbook.eng.kempnerinstitute.harvard.edu/s5b_agentic_ai_workflows/end_to_end_example.html) of the [Kempner Institute Computing Handbook](https://handbook.eng.kempnerinstitute.harvard.edu/) describes this example step by step.

## What the example does

1. You train the dense baseline one time.
2. You supervise round 1 in an interactive Claude Code session. You approve each agent action.
3. You start the loop job. For each round, the loop script asks the agent for the next configuration, checks the configuration, submits one round job, and records the result.
4. The loop script stops when 3 rounds in a row find no smaller model that keeps the accuracy threshold, or at a limit. Then it runs the final steps: a confirmation round with a second seed, a one-time test evaluation, a summary, and a draft report.
5. You review the result.

In the unattended phase, the agent has read-only tools. It cannot run commands, write files, or submit jobs. The loop script, which is committed code, does these tasks.

A second version of the loop uses four agent roles and runs 3 rounds at a time; see [Multi-agent version](#multi-agent-version).

## Before you start

- A Kempner AI cluster account, and a SLURM account that can use `kempner_requeue` and `shared`.
- Claude Code from the native installer, because the loop job looks for `claude` in `~/.local/bin`. Sign in with the Claude access that HUIT provides through Harvard SSO, and leave `ANTHROPIC_API_KEY` unset.
- [uv](https://docs.astral.sh/uv/).
- The data is the shared copy of CIFAR-10 on the cluster. CIFAR-10 is public data (Level 1).

## Set up

Clone the repository into lab storage, and copy the settings file:

```bash
mkdir -p /n/holylabs/LABS/<lab_name>/Users/<username>
cd /n/holylabs/LABS/<lab_name>/Users/<username>
git clone https://github.com/KempnerInstitute/agentic-workflow-example.git
cd agentic-workflow-example
cp cluster.env.example cluster.env
```

Edit `cluster.env`: set your SLURM account (`sshare -U -u $USER` lists your accounts), and your lab name and user name in the paths. Then load the file, and build the environment on a compute node:

```bash
source cluster.env && echo "$SBATCH_ACCOUNT $RUN_ROOT"   # check the values
mkdir -p logs && chmod 700 logs          # the hook writes logs/agent-commands.log here
salloc --partition=test --time=0-01:00 --mem=16G --cpus-per-task=4
uv sync --locked --extra gpu             # about 5 GB of packages
readlink -f .venv/bin/python             # must not be in your home directory
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
| Agent step | `claude-opus-5-5` at the `medium` effort level, `--max-turns 8`, `--max-budget-usd 1.00`, read-only tools, `dontAsk`. The loop script stops if a step used another model. |
| Rounds | 15 round attempts, round 1 included |
| GPU | 60 GPU minutes for the unattended phase. One round job at a time, 1 GPU, 10 minutes at most. |
| Queue | A job can wait at most 45 minutes in the queue |
| Loop job | `shared` partition, 2 CPUs, 8 GB, 6 hours |

## Results

| File | Content |
|---|---|
| `state/summary.json` | Stop reason, totals, the selected round, both seeds, validation and test accuracy with 95% intervals |
| `state/selection.json` | The selected round, saved before the confirmation round, so that `--finish` keeps it |
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
| Reads | File tools: only the project folder. Sandboxed commands: not the home folder. | Only the project folder, and not this README, which holds the results of our runs |
| Integrity | The round job and the loop script stop if a tracked file changed | The same |

## Skills

The project has three skills in `.claude/skills/`. Claude Code loads a skill when a task needs it, or when you type its name with a slash, for example `/check-round 1`.

| Skill | Source | Use |
|---|---|---|
| `check-round` | Written for this project | Checks the records of one round. It runs `.venv/bin/python -m pruning_example.check_round N`, which compares the metrics, the history row, and the round log, and counts the weights in the checkpoint again. |
| `reproducible-research` | The Coding Agent Skills page of the handbook | Practices for run records, seeds, environments, and data. Useful when you change the code. |
| `testing-and-continuous-integration` | The Coding Agent Skills page of the handbook | Practices for tests and CI. Useful when you change the code or the tests. |

- In this example, only the interactive session of round 1 can use the skills. The agent step in the loop has only the Read, Grep, and Glob tools, so it cannot load a skill.
- In our run, round 1 used `check-round`. The rounds did not use the two handbook skills.
- Read each skill before you trust the project folder, because Claude Code follows its instructions.

To update the two handbook skills, download them again, and commit the change. The round job stops if a tracked file changed.

```bash
base=https://handbook.eng.kempnerinstitute.harvard.edu/skills
for skill in reproducible-research testing-and-continuous-integration; do
  curl -fsSL "$base/$skill/SKILL.md" -o ".claude/skills/$skill/SKILL.md"
done
git diff --stat .claude/skills
```

## Multi-agent version

The multi-agent loop runs the same task with four agent roles. Each role step is one read-only `claude -p` call with its own model and effort level. The loop script still checks each output, submits the jobs, and writes all records.

| Role | Model | Effort | Task |
|---|---|---|---|
| Planner | `claude-opus-5-5` | `xhigh` | Plans each wave: 3 configurations, each with a hypothesis, a predicted validation accuracy, and a pass probability |
| Critic | `claude-sonnet-5-5` | `high` | Accepts or rejects each configuration before any GPU job runs |
| Writer | `claude-sonnet-5-5` | `medium` | Writes the draft report |
| Auditor | `claude-opus-5-5` | `max` | Checks each claim of the draft report against the records |

`experiment_multi.yaml` sets these values; change them there. A checker uses a different model from the role that it checks. With `roles.critic.enabled: false`, the loop is one agent that plans waves of 3. The role prompts are in `prompts/multi/roles/`.

Run it after the setup, the baseline, and round 1 above:

```bash
source cluster.env
sbatch --time=01:30:00 slurm/multi_loop.sbatch --plan-only   # plan wave 1, then stop
cat state_multi/waves/wave_01/plan.md                        # read the plan and the critic's verdicts
sbatch slurm/multi_loop.sbatch                               # run wave 1 and the next waves
```

To plan wave 1 again, remove `state_multi/waves/wave_01/`, and run the `--plan-only` job again. Watch the run with `tail -f logs/prune-multi-loop_<job_id>.out`. To stop the run, run `scancel <job_id>` on the loop job. The loop script cancels its array job. To run the final steps later, run `sbatch slurm/multi_loop.sbatch --finish`.

What changes from the single-agent loop:

- **Waves.** Each wave is one array job of up to 3 round jobs (`slurm/round_array.sbatch`). The loop script writes the history rows of a wave in round order.
- **Checks.** The code checks each configuration first: the schema, and no repeat. Then the critic checks it. A rejected configuration gets one revision. If the revision is rejected too, it is dropped.
- **Limits.** 1,000,000 tokens, with 20% kept for the writer and the auditor. 60 GPU minutes: the loop script starts a round only if the round's 10-minute time limit still fits. 45 round attempts. The loop stops after 6 rounds in a row without a smaller passing model.
- **Final steps.** Two confirmation rounds, with seeds 1 and 2, as one array job. The summary gives the mean validation accuracy and the mean test accuracy of the 3 seeds. Then the writer step, the auditor step, `check_report` (the numbers of the report against the records), and the checks of `check_round` for the selected round.
- **Records.** `state_multi/`, with `waves/wave_NN/` (the plan, the tasks, and the outcome), `hypotheses.tsv` (each prediction and its result), `audit.json`, and a ledger with the model and the effort level of each step.

To check the report of the single-agent loop the same way, run `STATE_DIR=state sbatch slurm/multi_loop.sbatch --audit-only`. To compare the two designs, run each one in its own clone that holds only the baseline and round 1. Then the agents of one design cannot read the records of the other.

## Test the loop with small limits

Put test copies in `local/`, which Git ignores. `EXPERIMENT_FILE` and `STATE_DIR` point the jobs to them:

```bash
mkdir -p local
cp experiment.yaml local/small.yaml       # then lower the limits in the copy
STATE_DIR=local/state_small EXPERIMENT_FILE=local/small.yaml sbatch slurm/loop.sbatch
```

The loop needs a round 1 in the state folder. The summary in the state folder records the experiment file and its hash.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests use generated images, so they need no GPU and no data. CI runs them with the `cpu` extra.

## Files

```text
AGENTS.md, CLAUDE.md      agent instructions
experiment.yaml           accuracy threshold, recipe, and limits
experiment_multi.yaml     roles, models, effort levels, waves, and limits of the multi-agent loop
cluster.env.example       your account and paths
.claude/                  settings, hook, and skills
configs/                  schemas of a round configuration, a wave plan, a review, and an audit
prompts/                  prompts of the agent step and the report step
prompts/multi/            role prompts and task prompts of the multi-agent loop
data/split.json           fixed validation split
slurm/                    baseline, round, round array, loop, and multi-agent loop jobs
src/pruning_example/      code: data, model, evaluator, pruning, records, loops, final steps, checks
tests/                    pytest tests
```

## Our run

One full run on October 17, 2026. Round 1 was supervised. The loop job ran rounds 2 to 15 and stopped at the round limit.

| Round | Sparsity | Steps | Epochs | Learning rate | Excluded | Validation accuracy | Weights left |
|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 1 | 2 | 0.01 | none | 0.9320 | 50% |
| 2 | 0.85 | 2 | 4 | 0.01 | none | 0.9356 | 15% |
| 3 | 0.95 | 3 | 6 | 0.01 | conv1 | 0.9180 | 5.0% |
| 5 | 0.98 | 5 | 10 | 0.01 | conv1 | 0.9074 | 2.0% |
| 7 | 0.988 | 5 | 20 | 0.01 | conv1 | 0.9052 | 1.2% |
| 8 | 0.989 | 4 | 20 | 0.01 | conv1 | 0.8976 (below) | 1.1% |
| 10 | 0.9895 | 5 | 20 | 0.03 | conv1 | 0.9154 | 1.07% |
| 15 | 0.9927 | 5 | 20 | 0.08 | conv1 | 0.9076 | 0.75% |

From round 9, the agent raised the learning rate by 0.01 in each round. This kept the threshold at higher sparsities.

| Result | Value |
|---|---|
| Selected model | Round 15: 83,215 nonzero weights (0.75%), sparse size 1.56 MB (dense: 44.7 MB) |
| Test accuracy | 0.9045 (95% interval 0.899 to 0.910) |
| Confirmation round (seed 1) | Validation 0.9050, test 0.9032 |
| Tokens | 323,072, estimated cost $1.11 |
| GPU minutes | 16.1 for the rounds, 1.0 for the confirmation round |
| Time of the loop job | 66 minutes |

The confirmation round passed exactly at the threshold, and both test values are close to 0.900. Read the result as a lead to check, not as a finding.

## Tested with

Claude Code 2.1.302, uv 0.8.19, Python 3.12.11, PyTorch 2.14.1 with CUDA 13.0, torchvision 0.29.1, and NVIDIA driver 610.57.04, on A100, H100, H200, and RTX PRO 6000 GPUs, in October 2026.

The multi-agent version has passed its unit tests. Its tests on the cluster and its runs are still to come.

## Adapt the example

- **Change** the task: the data, the model, the metric, the round configuration schema, the accuracy threshold, and the limits.
- **Keep** the fixed evaluator, the read-only agent step, the loop script that submits the jobs, the limits outside the model, the run records, the confirmation round, and the one-time test evaluation.

## Sources

The skills in `.claude/skills/reproducible-research/` and `.claude/skills/testing-and-continuous-integration/`, and the hook `.claude/hooks/guard.sh`, come from the Kempner Institute Computing Handbook.

## License

MIT. Copyright (c) 2026 Harvard University. See `LICENSE`. To cite this example, see `CITATION.cff`.
