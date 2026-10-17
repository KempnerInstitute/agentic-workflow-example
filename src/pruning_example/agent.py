"""Run claude -p for an agent step or for the report step, and count its tokens."""

from __future__ import annotations

import json
import os
import subprocess

TOKEN_FIELDS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
READ_ONLY_TOOLS = "Read,Grep,Glob"


class AgentError(RuntimeError):
    """claude -p failed, or it printed no JSON result."""


class AgentFormatError(AgentError):
    """The JSON result does not have a required field. The loop script stops."""


def tokens_used(result: dict) -> int:
    """Add the four token types of all models in modelUsage. Cache reads also count."""
    usage = result.get("modelUsage")
    if not isinstance(usage, dict) or not usage:
        raise AgentFormatError("the JSON result of claude -p has no modelUsage field")
    try:
        return sum(int(model[field]) for model in usage.values() for field in TOKEN_FIELDS)
    except (KeyError, TypeError, ValueError) as error:
        raise AgentFormatError(f"modelUsage has no token count: {error}") from None


def command(prompt: str, caps: dict, schema_text: str | None = None) -> list[str]:
    """Return the claude -p command: read-only tools, dontAsk, no MCP tools, and the caps."""
    args = [
        "claude", "-p", prompt,
        "--permission-mode", "dontAsk",
        "--tools", READ_ONLY_TOOLS,
        "--disallowedTools", "mcp__*",
        "--max-turns", str(caps["max_turns"]),
        "--max-budget-usd", f"{caps['max_budget_usd']:.2f}",
        "--output-format", "json",
    ]
    if schema_text is not None:
        args += ["--json-schema", schema_text]
    return args


class Agent:
    """Runs claude -p in the project folder. stop() ends a running call."""

    def __init__(self, project):
        self.project = project
        self.process = None

    def run(self, prompt: str, caps: dict, schema_text: str | None = None) -> dict:
        env = {name: value for name, value in os.environ.items() if name != "ANTHROPIC_API_KEY"}
        process = subprocess.Popen(
            command(prompt, caps, schema_text), cwd=self.project, env=env,
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
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
