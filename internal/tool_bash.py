"""Bash, rooted: one shell command per call, cwd = the given root.

    agent = Agent(llm, tools=[bash_tool("workspace")])

NOT a sandbox. Commands run with this process's full permissions and
can walk straight out of the root (``cd /``, absolute paths, network) —
the root is a home, not a jail. Same trust model as examples/cc: gate
``bash`` with internal.hook_approval before pointing it at anything you
would not run by hand.
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

from core import Tool, ToolSpec

MAX_OUTPUT = 30_000     # chars of output kept before truncation
MAX_TIMEOUT = 600.0     # seconds; the ceiling on the caller's timeout

_SCHEMA = {"type": "object", "required": ["command"], "properties": {
    "command": {"type": "string",
                "description": "Shell command; runs from the root directory, "
                               "stdout and stderr merged."},
    "timeout": {"type": "number",
                "description": "Seconds before the command is killed "
                               f"(default 120, max {MAX_TIMEOUT:g})."}}}


def bash_tool(root: str | Path) -> Tool:
    """The bash tool, rooted at ``root`` — the working directory every
    command starts in, created if missing. Required: a shell's cwd is
    the developer's call, never core's guess."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    async def bash(cctx: Any, command: str = "", timeout: float = 120) -> str:
        """Run a shell command and return its merged stdout and stderr."""
        # ponytail: one subprocess per call — cwd/env do not persist across
        # calls. A long-lived shell + sentinel framing if that ever matters.
        if not command.strip():
            raise ValueError("command must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0 (seconds)")
        limit = min(timeout, MAX_TIMEOUT)
        proc = await asyncio.create_subprocess_shell(
            command, cwd=root, start_new_session=True,  # own group: kill its children
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        chunks: list[bytes] = []

        async def drain() -> None:
            while chunk := await proc.stdout.read(65536):
                chunks.append(chunk)

        reader = asyncio.ensure_future(drain())
        note = ""
        try:
            await asyncio.wait_for(asyncio.shield(reader), limit)
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # the whole tree, not just sh
            except ProcessLookupError:
                pass
            note = f"\n(command timed out after {limit:g}s and was killed)"
            await asyncio.wait({reader}, timeout=1.0)  # a process that escaped
            reader.cancel()                            # holds stdout — don't wait
            # ponytail: private, but asyncio exposes no public way to drop our
            # end of a pipe an escapee still holds open. Leaks an fd otherwise.
            proc._transport.close()
        await proc.wait()
        text = b"".join(chunks).decode(errors="replace")   # keep whatever it printed
        if len(text) > MAX_OUTPUT:
            over = len(text) - MAX_OUTPUT
            text = text[:MAX_OUTPUT] + f"\n… truncated ({over} more chars)"
        if proc.returncode and not note:
            text += f"\n(exit code {proc.returncode})"
        return (text + note) or "(no output)"

    return Tool(spec=ToolSpec("bash", bash.__doc__ or "", _SCHEMA),
                handler=bash, parallel_safe=False)
