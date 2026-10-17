# Agentic Workflow Example

A Claude Code agent prunes a ResNet-18 model for CIFAR-10 on the Kempner AI cluster. The goal is a small model with a test accuracy of at least 90%.

You supervise round 1. Then a loop job runs more rounds without a person, until a model meets the target or the run reaches a limit. The token budget of the unattended phase is 1,000,000 tokens.

This repository is under development. The Kempner Institute Computing Handbook will describe the full workflow.

## License

MIT. Copyright (c) 2026 Harvard University. See `LICENSE`.
