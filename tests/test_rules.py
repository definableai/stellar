"""The rules that keep core small, stdlib-only and provider-blind.

Run: uv run python tests/test_rules.py
"""

import ast
import re
import sys
from pathlib import Path

CORE = sorted((Path(__file__).parent.parent / "core").glob("*.py"))
BUDGET = 2000
PRODUCTS = re.compile(
    r"\b(mcp|anthropic|openai|litellm|claude|gpt|a2a|acp)\b", re.IGNORECASE
)


def import_roots(source: str) -> list[str]:
    """The top-level package name of every import in the source."""
    roots = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots.append("core" if node.level else (node.module or "").split(".")[0])
    return roots


def test_core_fits_the_budget() -> None:
    total = sum(len(f.read_text().splitlines()) for f in CORE)
    assert total <= BUDGET, f"core is {total} lines, budget is {BUDGET}"
    print(f"  core: {total}/{BUDGET} lines")


def test_core_imports_stdlib_and_itself_only() -> None:
    for f in CORE:
        for root in import_roots(f.read_text()):
            ok = root == "core" or root in sys.stdlib_module_names
            assert ok, f"core/{f.name} imports {root}"


def test_core_names_no_products() -> None:
    for f in CORE:
        found = PRODUCTS.search(f.read_text())
        assert not found, f"core/{f.name} says {found.group()!r}"


if __name__ == "__main__":
    assert CORE, "no core/*.py files found"
    test_core_fits_the_budget()
    test_core_imports_stdlib_and_itself_only()
    test_core_names_no_products()
    print("test_rules: all ok")
