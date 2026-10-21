"""Check the numbers of a draft report against the records of a state folder.

Usage: python -m pruning_example.check_report [REPORT]

The command lists each number of the report that it cannot find in the records. It checks each number
with a decimal point, each percentage, and each whole number of 4 or more digits. It skips small whole
numbers, such as round numbers and seeds, and years. A record value matches if the report number is
a rounding of it, also after a factor of 100 for percentages. The list is a hint for you, not a verdict:
a correct number can be missing from the records, for example a difference of two values.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from . import config

# A number in prose: 83,215 or 0.9076 or 0.75%. Version numbers such as 2.1.302 do not match.
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)(?!\.\d)")
TABLES = ("history.tsv", "hypotheses.tsv", "ledger.tsv")
JSON_GLOBS = ("summary.json", "rounds/*/metrics.json", "confirm/*/metrics.json", "test_eval/*.json")


def numbers_in(value, found: list[float]) -> None:
    """Add each number in a JSON value to found."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        found.append(float(value))
    elif isinstance(value, str):
        try:
            found.append(float(value))
        except ValueError:
            pass
    elif isinstance(value, dict):
        for item in value.values():
            numbers_in(item, found)
    elif isinstance(value, list):
        for item in value:
            numbers_in(item, found)


def record_values(state: Path) -> list[float]:
    """Return the numbers in the records of a state folder."""
    found: list[float] = []
    for pattern in JSON_GLOBS:
        for path in sorted(state.glob(pattern)):
            try:
                numbers_in(json.loads(path.read_text()), found)
            except (OSError, json.JSONDecodeError):
                continue
    for name in TABLES:
        path = state / name
        if path.exists():
            with open(path, newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    numbers_in(list(row.values()), found)
    return found


def matches(whole: str, fraction: str | None, values: list[float]) -> bool:
    """Return True if the number is a rounding of a record value, or of 100 times a record value."""
    number = float(whole.replace(",", "") + ("." + fraction if fraction else ""))
    half_step = 0.5 * 10 ** -(len(fraction) if fraction else 0) + 1e-12
    return any(abs(value - number) <= half_step or abs(100 * value - number) <= half_step for value in values)


def checked(whole: str, fraction: str | None, percent: str) -> bool:
    """Return True for the numbers that the check covers."""
    if percent and whole == "95" and not fraction:
        return False  # the level of the intervals, as in "95% interval"
    if fraction or percent:
        return True
    digits = whole.replace(",", "")
    return len(digits) >= 4 and not (len(digits) == 4 and 1900 <= int(digits) <= 2100)  # skip years


def unmatched_numbers(text: str, state: Path) -> list[dict]:
    """Return the numbers of the report that the records do not show, with their line numbers."""
    values = record_values(state)
    missing = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for match in NUMBER.finditer(line):
            whole, fraction, percent = match.groups()
            if checked(whole, fraction, percent) and not matches(whole, fraction, values):
                missing.append({"number": match.group(0), "line": line_no})
    return missing


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    paths = config.paths_from_env()
    report = Path(argv[0]) if argv else paths.state / "final_report.md"
    if not report.exists():
        print(f"There is no report: {report}")
        return 2
    missing = unmatched_numbers(report.read_text(), paths.state)
    for item in missing:
        print(f"line {item['line']}: {item['number']} is not in the records")
    print(f"{len(missing)} number(s) not found in {paths.state.name}/" if missing else "Each checked number is in the records.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
