"""The kernel: the agent's hands on itself. One adapter, four tools.

    adapter_list()          the mirror: what am I made of
    adapter_load(path)      import a workspace .py, mount it
    adapter_unload(name)    drop it — inverses unwind LIFO
    adapter_reload(name)    drop + fresh import (the edit loop)

    agent.use(kernel, workspace="external")   # just the tools
    boot(agent, "external")                   # + every workspace *.py

Self-modification is a tool call: the same before_tool hooks that gate
bash gate adapter_load (wire your approval rules to ``adapter_*``), and
the session log records every mount as an ordinary tool step. The
kernel is itself an adapter — don't mount it and the agent is frozen;
an agent may even drop it (auditable self-lockdown).

Trust model — read before deploying. Adapter code runs IN-PROCESS with
full interpreter privileges; this is not a sandbox. The security
boundary is (1) who can write the workspace directory and (2) the
before_tool gate on ``adapter_load`` and on file writes into the
workspace. Loading is path-jailed to the workspace: never point it at
a directory unreviewed third parties can write, and keep the workspace
under git so every self-change is a reviewable diff.
"""

from __future__ import annotations

from itertools import count
from pathlib import Path
from types import ModuleType
from typing import Any

from .adapter import Scope
from .agent import Agent
from .tools import Tool, ToolSpec

_seq = count(1)


def load_module(path: Path) -> ModuleType:
    """Execute a .py fresh — no sys.modules, no bytecode cache (importlib's
    .pyc check misses same-size same-second edits, which is exactly what an
    agent's quick edit-reload loop produces). Must define setup(ctx)."""
    mod = ModuleType(f"stellar_adapter_{path.stem}_{next(_seq)}")
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    if not callable(getattr(mod, "setup", None)):
        raise TypeError(f"{path.name} defines no setup(ctx)")
    return mod


def _name(ws: Path, path: Path) -> str:
    # workspace-relative, so tool/x.py and hook/x.py never collide
    return path.relative_to(ws).with_suffix("").as_posix()


def _mount_file(agent: Agent, ws: Path, path: Path) -> Scope:
    return agent.use(load_module(path).setup,
                     name=_name(ws, path), source=str(path))


def kernel(ctx: Any) -> None:
    """The four self-composition tools. Config: ``workspace=`` — the only
    directory adapter_load may read. Required: where an agent keeps its
    own parts is the developer's call, never core's guess."""
    agent: Agent = ctx.agent
    if not (workspace := ctx.config.get("workspace")):
        raise ValueError("kernel needs workspace=<dir>: "
                         "agent.use(kernel, workspace=...)")
    ws = Path(workspace).resolve()

    def jail(rel: str) -> Path:
        p = (ws / rel).resolve()
        if not (p.is_relative_to(ws) and p.suffix == ".py"):
            raise ValueError(f"adapters load only from {ws} (*.py files): {rel!r}")
        return p

    def adapter_list(cctx: Any) -> list[dict[str, Any]]:
        """List every mounted adapter: name, source file, contributions."""
        return [{"name": s.name, "source": s.source, "contributed": s.notes}
                for s in agent.adapters.values()]

    def adapter_load(cctx: Any, path: str = "") -> str:
        """Load an adapter file from the workspace and mount it."""
        p = jail(path)
        if _name(ws, p) in agent.adapters:   # before exec: module body runs once
            raise ValueError(
                f"{_name(ws, p)!r} already mounted; adapter_reload picks up edits")
        scope = _mount_file(agent, ws, p)
        return f"mounted {scope.name!r}: {', '.join(scope.notes) or 'nothing registered'}"

    def adapter_unload(cctx: Any, name: str = "") -> str:
        """Unmount an adapter; its registrations unwind newest-first."""
        errors = agent.drop(name)
        return (f"unloaded {name!r}"
                + (f" (inverse errors: {[e.message for e in errors]})"
                   if errors else ""))

    def adapter_reload(cctx: Any, name: str = "") -> str:
        """Re-import a file-backed adapter from disk and remount it."""
        source = agent.adapters[name].source   # KeyError -> readable tool error
        if not source:
            raise ValueError(f"{name!r} is not file-backed; unload it instead")
        mod = load_module(Path(source))   # import FIRST: a broken edit is a
        agent.drop(name)                  # readable error, the old version stays
        scope = agent.use(mod.setup, name=name, source=source)
        return f"reloaded {scope.name!r}: {', '.join(scope.notes) or 'nothing registered'}"

    path_arg = {"type": "object", "required": ["path"], "properties": {
        "path": {"type": "string",
                 "description": "Adapter file (.py defining setup(ctx)), "
                                "relative to the workspace."}}}
    name_arg = {"type": "object", "required": ["name"], "properties": {
        "name": {"type": "string", "description": "A mounted adapter's name."}}}
    none_arg = {"type": "object", "properties": {}}
    for fn, params in [(adapter_list, none_arg), (adapter_load, path_arg),
                       (adapter_unload, name_arg), (adapter_reload, name_arg)]:
        # composition mutations serialize the whole tool batch
        ctx.tool(Tool(spec=ToolSpec(fn.__name__, fn.__doc__ or "", params),
                      handler=fn, parallel_safe=False))


def boot(agent: Agent, workspace: str | Path,
         *, mount_kernel: bool = True) -> list[Scope]:
    """Mount the kernel plus every workspace ``*.py`` sorted by name
    (prefix ``00-``, ``10-`` to order mounts). Subdirectories are mounted
    too, named by relative path (``tool/x.py`` mounts as ``tool/x``), so
    an agent may file its parts under hooks/, llm/, tool/ — its choice;
    a leading ``_`` on a file or folder skips it. The directory
    IS the manifest: move a file out to disable it. Missing dir = empty
    self. A broken file fails the boot loudly; files mounted before it
    stay."""
    ws = Path(workspace).resolve()
    scopes = []
    if mount_kernel:
        scopes.append(agent.use(kernel, workspace=str(ws)))
    if ws.is_dir():
        scopes += [_mount_file(agent, ws, p) for p in sorted(ws.rglob("*.py"))
                   if not any(part.startswith("_")
                              for part in p.relative_to(ws).parts)]
    return scopes

