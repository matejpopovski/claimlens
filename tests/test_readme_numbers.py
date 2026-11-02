"""Every decimal in the README prose must trace to a committed results file."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DECIMAL = re.compile(r"(?<![\d.\w-])(\d+\.\d+)(%?)(?![\d.]*\.\d|\w)")


def result_values() -> list[float]:
    values: list[float] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, float | int) and not isinstance(node, bool):
            values.append(float(node))

    for path in (REPO / "results").rglob("*.json"):
        walk(json.loads(path.read_text()))
    return values


def prose(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def untraceable(text: str, values: list[float]) -> list[str]:
    missing = []
    for match in DECIMAL.finditer(text):
        token, percent = match.groups()
        digits = len(token.split(".")[1])
        written = float(token)
        scale = (1.0, 100.0) if percent or written > 1 else (1.0,)
        if not any(round(v * s, digits) == written for v in values for s in scale):
            missing.append(match.group(0))
    return missing


def test_every_readme_decimal_is_in_results():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert untraceable(prose(readme), result_values()) == []


def test_the_check_notices_an_invented_number():
    assert untraceable("reached 0.123 and 45.6%", [0.5]) == ["0.123", "45.6%"]
    assert untraceable("reached 0.500 and 50.0%", [0.5]) == []


def test_versions_and_code_blocks_are_not_treated_as_results():
    assert untraceable("playwright 1.49.1 and fastapi 0.115.0", []) == []
    assert untraceable(prose("```\nnum = 0.123\n```"), []) == []
