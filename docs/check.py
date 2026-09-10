"""The docs' own test: the table and the pages agree, links land, python parses.

    python3 docs/check.py

Every page docs.json lists has a file under content/; every file is listed,
unless its frontmatter says `hidden: true`; every page has a title and a
description; every absolute link points at a page; every ```python fence
compiles (top-level await allowed; add `nocheck` to a fence to skip it).
One line per problem and exit 1, else one line and exit 0.
"""

import ast
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTENT = HERE / "content"
FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
FENCE = re.compile(r"^[ \t]*```(\w+)([^\n]*)\n(.*?)^[ \t]*```", re.M | re.S)
LINK = re.compile(r"\]\(([^)\s]+)\)|href=\"([^\"]+)\"")
NAV_KEYS = ("tabs", "groups", "pages", "anchors", "dropdowns")


def pages(node):
    """Every page string under a navigation node, in reading order."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for child in node:
            yield from pages(child)
    elif isinstance(node, dict):
        for key in NAV_KEYS:
            yield from pages(node.get(key, []))


def frontmatter(text):
    """The `key: value` lines between the dashes, or None without them."""
    m = FRONT.match(text)
    if not m:
        return None
    pairs = (line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in pairs}


def problems():
    nav = json.loads((HERE / "docs.json").read_text())["navigation"]
    listed = list(pages(nav))
    files = {p.relative_to(CONTENT).with_suffix("").as_posix(): p
             for p in sorted(CONTENT.rglob("*.mdx"))} if CONTENT.is_dir() else {}
    seen = set()
    for page in listed:
        if page in seen:
            yield f"docs.json: {page} is listed twice"
        seen.add(page)
        if page not in files:
            yield f"docs.json: {page} has no content/{page}.mdx"
    for name, path in files.items():
        rel = path.relative_to(HERE)
        text = path.read_text()
        fm = frontmatter(text)
        if fm is None:
            yield f"{rel}: no frontmatter"
            continue
        for key in ("title", "description"):
            if not fm.get(key):
                yield f"{rel}: frontmatter has no {key}"
        if name not in seen and fm.get("hidden") != "true":
            yield f"{rel}: not in docs.json (say `hidden: true` if that is on purpose)"
        for m in LINK.finditer(FENCE.sub("", text)):      # a link inside a fence is not a link
            href = m.group(1) or m.group(2)
            if href.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not href.startswith("/"):
                yield f"{rel}: link {href!r} is relative; use an absolute site path"
                continue
            target = href.split("#")[0].strip("/") or "introduction"
            if target not in files:
                yield f"{rel}: link {href!r} points at no page"
        for m in FENCE.finditer(text):
            lang, meta, code = m.groups()
            if lang != "python" or "nocheck" in meta:
                continue
            line = text.count("\n", 0, m.start()) + 1
            try:
                compile(textwrap.dedent(code), f"{rel}:{line}", "exec",
                        flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as e:
                yield f"{rel}:{line}: python fence does not parse: {e.msg} (fence line {e.lineno})"
    yield from ()


def main() -> int:
    found = list(problems())
    for line in found:
        print(line)
    if found:
        print(f"{len(found)} problem(s)")
        return 1
    count = sum(1 for _ in CONTENT.rglob("*.mdx")) if CONTENT.is_dir() else 0
    print(f"ok: {count} pages, every link lands, every python fence parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
