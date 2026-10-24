"""Run claude -p for an unattended agent step, and count its tokens.

The single-agent loop runs the agent step and the report step. The multi-agent loop runs the role steps.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

TOKEN_FIELDS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
READ_ONLY_TOOLS = "Read,Grep,Glob"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# The README holds example results. An unattended step must find its own way, so it cannot read them.
UNATTENDED_SETTINGS = {"permissions": {"deny": ["Read(./README.md)"]}}


class AgentError(RuntimeError):
    """claude -p failed, or it printed no JSON result."""


class AgentFormatError(AgentError):
    """The JSON result does not have a required field. The loop script stops."""


class ModelCheckError(AgentError):
    """The JSON result does not show the configured model. The loop script stops."""


def tokens_used(result: dict) -> int:
    """Add the four token types of all models in modelUsage. Cache reads also count."""
    usage = result.get("modelUsage")
    if not isinstance(usage, dict) or not usage:
        raise AgentFormatError("the JSON result of claude -p has no modelUsage field")
    try:
        return sum(int(model[field]) for model in usage.values() for field in TOKEN_FIELDS)
    except (KeyError, TypeError, ValueError) as error:
        raise AgentFormatError(f"modelUsage has no token count: {error}") from None


def check_model(result: dict, model: str | None) -> None:
    """Raise ModelCheckError if the step did not use the configured model.

    Claude Code replaces a model that the organization does not allow with the default model,
    and it gives only a warning. An alias such as opus names no fixed model, so it is not checked.
    """
    if not model or not model.startswith("claude-"):
        return
    used = sorted((result.get("modelUsage") or {}).keys())
    # The name itself, with a date such as -20261001, or with a suffix such as [1m]. claude-opus-5 does not
    # match claude-opus-5-5, which is another model.
    same = re.compile(re.escape(model) + r"(-\d{8})?(\[[^\]]*\])?")
    if not any(same.fullmatch(name) for name in used):
        raise ModelCheckError(f"the step used {', '.join(used) or 'no model'}, not the configured model {model}")


def command(prompt: str, caps: dict, schema_text: str | None = None) -> list[str]:
    """Return the claude -p command: read-only tools, dontAsk, no MCP tools, the caps, and the optional
    model, effort level, and role prompt."""
    args = [
        "claude", "-p", prompt,
        "--permission-mode", "dontAsk",
        "--tools", READ_ONLY_TOOLS,
        "--disallowedTools", "mcp__*",
        "--max-turns", str(caps["max_turns"]),
        "--max-budget-usd", f"{caps['max_budget_usd']:.2f}",
        "--output-format", "json",
        "--settings", json.dumps(UNATTENDED_SETTINGS),
    ]
    if caps.get("model"):
        args += ["--model", str(caps["model"])]
    if caps.get("effort"):
        args += ["--effort", str(caps["effort"])]
    if caps.get("role_prompt"):
        args += ["--append-system-prompt-file", str(caps["role_prompt"])]
    if schema_text is not None:
        args += ["--json-schema", schema_text]
    return args


def step_env(caps: dict) -> dict:
    """Return the environment of a step.

    Without ANTHROPIC_API_KEY, print mode uses your Harvard login. When the step sets an effort level,
    CLAUDE_CODE_EFFORT_LEVEL is removed too, because it would override --effort.
    """
    drop = {"ANTHROPIC_API_KEY"}
    if caps.get("effort"):
        drop.add("CLAUDE_CODE_EFFORT_LEVEL")
    return {name: value for name, value in os.environ.items() if name not in drop}


class Agent:
    """Runs claude -p in the project folder. stop() ends a running call."""

    def __init__(self, project):
        self.project = project
        self.process = None

    def run(self, prompt: str, caps: dict, schema_text: str | None = None) -> dict:
        process = subprocess.Popen(
            command(prompt, caps, schema_text), cwd=self.project, env=step_env(caps),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.process = process
        try:
            out, err = process.communicate(timeout=60 * caps["timeout_minutes"])
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.process = None
            raise AgentError("claude -p did not end in time") from None
        self.process = None
        try:
            result = json.loads(out)
        except json.JSONDecodeError:
            raise AgentError(f"claude -p exited with {process.returncode} and printed no JSON: "
                             f"{err.strip()[:300]}") from None
        result["exit_code"] = process.returncode
        return result

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
