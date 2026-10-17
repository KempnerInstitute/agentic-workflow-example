#!/bin/bash
# .claude/hooks/guard.sh: block job cancellation and recursive deletes.
# The tool call arrives as JSON on stdin; exit code 2 blocks it.
# Source: the Hooks section of the Kempner Institute Computing Handbook.
command -v jq >/dev/null || { echo "guard.sh needs jq; blocking." >&2; exit 2; }
cmd=$(jq -r '.tool_input.command // empty')
if [[ "$cmd" =~ scancel|clustertool[[:space:]]+jobs[[:space:]]+cancel ]] ||
   [[ "$cmd" =~ rm[[:space:]](.*[[:space:]])?(-[[:alpha:]]*[rR]|--r) ]] ||
   [[ "$cmd" =~ find[[:space:]].*(-delete|-exec(dir)?[[:space:]]+rm)|xargs[[:space:]](.*[[:space:]])?rm([[:space:]]|$) ]] ||
   [[ "$cmd" =~ git[[:space:]]+(-C[[:space:]]+[^[:space:]]+[[:space:]]+)?clean ]]; then
  echo "Blocked by project hook: run this yourself if you mean it." >&2
  exit 2
fi
exit 0
