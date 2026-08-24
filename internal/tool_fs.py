"""File primitives, rooted: read / write / edit / list one directory.

    agent = Agent(llm, tools=[*fs_tools("workspace")])

Every ``path`` is relative to ``root`` and resolved before use, so
``..`` and symlinks are judged by where they land, not how they spell
it; anything outside ``root`` is a readable error. Text only, UTF-8,
line endings verbatim — ``edit_file`` matches strings, so a CRLF file
stays CRLF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import Tool, ToolSpec

MAX_LINES = 2000        # read_file default and ceiling
MAX_ENTRIES = 500       # list_files entries before the "… N more" tail
_SKIP = {".git", "__pycache__"}

_PATH = {"type": "string",
         "description": "File path, relative to the root directory."}
_READ = {"type": "object", "required": ["path"], "properties": {
    "path": _PATH,
    "offset": {"type": "integer",
               "description": "First line to read, 1-based (0 = start of file)."},
    "limit": {"type": "integer",
              "description": f"Max lines to return (default {MAX_LINES})."}}}
_WRITE = {"type": "object", "required": ["path", "content"], "properties": {
    "path": _PATH,
    "content": {"type": "string",
                "description": "Full new contents; overwrites any existing file."}}}
_EDIT = {"type": "object", "required": ["path", "old", "new"], "properties": {
    "path": _PATH,
    "old": {"type": "string", "description": "Exact text to replace — must "
                                             "be unique unless replace_all."},
    "new": {"type": "string", "description": "Text to put in its place."},
    "replace_all": {"type": "boolean",
                    "description": "Replace every occurrence (default false)."}}}
_LIST = {"type": "object", "properties": {
    "path": {"type": "string",
             "description": "Directory to list, relative to the root (default '.')."},
    "glob": {"type": "string",
             "description": "Pattern matched under it, e.g. '*.py' (default '**/*')."}}}


def _lines(text: str) -> list[str]:
    # \n only: \f, \v and U+2028 are text to an editor, not line breaks
    lines = text.split("\n")
    if lines[-1] == "":     # a trailing newline terminates a line, it doesn't add one
        lines.pop()
    return lines


def _slurp(p: Path, name: str) -> str:
    try:
        return p.read_text(encoding="utf-8", newline="")
    except UnicodeDecodeError:
        raise ValueError(
            f"{name} is not UTF-8 text — these tools are text-only") from None


def _listed(root: Path, p: Path) -> str | None:
    """Root-relative posix path if ``p`` is a listable file, else None."""
    q = p.resolve()   # a glob escapes too (`../*`): judge where it landed
    if not (q.is_file() and q.is_relative_to(root)):
        return None
    rel = q.relative_to(root)
    return None if _SKIP & set(rel.parts) else rel.as_posix()


def fs_tools(root: str | Path) -> list[Tool]:
    """The four file tools, rooted at ``root`` — the directory every path
    resolves under, created if missing. Required: where an agent keeps
    its files is the developer's call, never core's guess."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def jail(rel: str) -> Path:
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f"paths stay under {root}: {rel!r}")
        return p

    def read_file(cctx: Any, path: str = "", offset: int = 0,
                  limit: int = MAX_LINES) -> str:
        """Read a UTF-8 text file, or a window of its lines."""
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        lines = _lines(_slurp(jail(path), path))
        start = max(offset - 1, 0)
        if lines and start >= len(lines):
            raise ValueError(
                f"offset {offset} is past the end of {path} ({len(lines)} lines)")
        if not lines:
            return "(empty file)"
        window = lines[start:start + min(limit, MAX_LINES)]
        left = len(lines) - (start + len(window))
        return "\n".join(window) + (f"\n… {left} more lines" if left > 0 else "")

    def write_file(cctx: Any, path: str = "", content: str = "") -> str:
        """Write a text file whole, creating parent directories."""
        p = jail(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")
        n = len(_lines(content))
        return f"Wrote {n} line{'s' * (n != 1)} to {path}"

    def edit_file(cctx: Any, path: str = "", old: str = "", new: str = "",
                  replace_all: bool = False) -> str:
        """Replace an exact string in a text file."""
        if not old:
            raise ValueError("old must not be empty")
        if old == new:
            raise ValueError("new must differ from old")
        p = jail(path)
        src = _slurp(p, path)
        hits = src.count(old)
        if hits == 0:
            raise ValueError(f"old not found in {path} (0 matches)")
        if hits > 1 and not replace_all:
            raise ValueError(f"old is not unique in {path} ({hits} matches) — "
                             "add surrounding context or pass replace_all")
        p.write_text(src.replace(old, new, -1 if replace_all else 1),
                     encoding="utf-8", newline="")
        n = hits if replace_all else 1
        return f"Edited {path} ({n} replacement{'s' * (n != 1)})"

    def list_files(cctx: Any, path: str = ".", glob: str = "**/*") -> str:
        """List files under a directory, as paths relative to the root."""
        base = jail(path)
        if not base.is_dir():
            raise ValueError(f"not a directory: {path!r}")
        try:
            found = sorted({r for p in base.glob(glob) if (r := _listed(root, p))})
        except (NotImplementedError, ValueError) as ex:
            raise ValueError(f"bad glob {glob!r}: {ex}") from None
        if len(found) > MAX_ENTRIES:
            found = found[:MAX_ENTRIES] + [f"… {len(found) - MAX_ENTRIES} more"]
        return "\n".join(found) or "(no files)"

    return [Tool(spec=ToolSpec(fn.__name__, fn.__doc__ or "", params),
                 handler=fn, parallel_safe=parallel_safe)
            for fn, params, parallel_safe in [
                (read_file, _READ, True), (write_file, _WRITE, False),
                (edit_file, _EDIT, False), (list_files, _LIST, True)]]
