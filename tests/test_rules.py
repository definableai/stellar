"""The rules that keep core small, stdlib-only (httpx in llm.py) and provider-blind.

Run: uv run python tests/test_rules.py
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORE = sorted((ROOT / "core").glob("*.py"))
BUDGET = 2000
PRODUCTS = re.compile(
    r"\b(mcp|anthropic|openai|litellm|claude|gpt|a2a|acp)\b", re.IGNORECASE
)
ADAPTERS = {"models": {"httpx"}, "hooks": set(), "drivers": set()}


def imports(source: str) -> list[str]:
    """Every import's module name, dotted. A relative import answers in its dots."""
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level or (node.module or ""))
    return names


def root(name: str) -> str:
    """The top-level package one dotted import belongs to; dots come back whole."""
    return name if name.startswith(".") else name.split(".")[0]


def test_core_fits_the_budget() -> None:
    total = sum(len(f.read_text().splitlines()) for f in CORE)
    assert total <= BUDGET, f"core is {total} lines, budget is {BUDGET}"
    print(f"  core: {total}/{BUDGET} lines")


def test_core_imports_stdlib_and_itself_only() -> None:
    for f in CORE:
        for name in imports(f.read_text()):
            top = root(name)
            ok = (top in (".", "core") or top in sys.stdlib_module_names
                  or (f.name == "llm.py" and top == "httpx"))
            assert ok, f"core/{f.name} imports {top}"


def test_core_names_no_products() -> None:
    for f in CORE:
        found = PRODUCTS.search(f.read_text())
        assert not found, f"core/{f.name} says {found.group()!r}"


def test_adapters_import_core_and_stdlib_only() -> None:
    """models/, hooks/ and drivers/ speak core and stdlib — never each other.

    A provider is a folder under models/; its files speak each other,
    relatively, and nobody else.
    """
    for folder, extra in ADAPTERS.items():
        files = sorted((ROOT / folder).rglob("*.py"))
        assert files, f"no {folder}/*.py files found"
        for f in files:
            for name in imports(f.read_text()):
                top = root(name)
                ok = (top in extra or top == "core"
                      or top in sys.stdlib_module_names
                      or (top == "." and f.parent != ROOT / folder))
                assert ok, f"{f.relative_to(ROOT)} imports {name}"


if __name__ == "__main__":
    assert CORE, "no core/*.py files found"
    test_core_fits_the_budget()
    test_core_imports_stdlib_and_itself_only()
    test_core_names_no_products()
    test_adapters_import_core_and_stdlib_only()
    print("test_rules: all ok")
