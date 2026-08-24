"""The kernel: the agent's hands on itself. One adapter, four tools.

    adapter_list()          the mirror: what am I made of
    adapter_load(path)      import a workspace .py, mount it
    adapter_unload(name)    drop it — inverses unwind LIFO
    adapter_reload(name)    drop + fresh import (the edit loop)

    agent.use(kernel, workspace=".stellar/adapters")   # just the tools
    boot(agent, ".stellar/adapters")   # kernel + every workspace *.py

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

Self-check: uv run python -m internal.kernel
"""

from __future__ import annotations

from itertools import count
from pathlib import Path
from types import ModuleType
from typing import Any

from core import Agent, Scope, Tool, ToolSpec

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


def _mount_file(agent: Agent, path: Path, name: str | None = None) -> Scope:
    return agent.use(load_module(path).setup,
                     name=name or path.stem, source=str(path))


def kernel(ctx: Any) -> None:
    """The four self-composition tools. Config: ``workspace=`` (the only
    directory adapter_load may read; default .stellar/adapters)."""
    agent: Agent = ctx.agent
    ws = Path(ctx.config.get("workspace", ".stellar/adapters")).resolve()

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
        scope = _mount_file(agent, jail(path))
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
        agent.drop(name)
        scope = _mount_file(agent, Path(source), name)
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


def boot(agent: Agent, workspace: str | Path = ".stellar/adapters",
         *, mount_kernel: bool = True) -> list[Scope]:
    """Mount the kernel plus every workspace ``*.py`` sorted by name
    (prefix ``00-``, ``10-`` to order mounts). The directory IS the
    manifest: move a file out to disable it. Missing dir = empty self."""
    ws = Path(workspace).resolve()
    scopes = []
    if mount_kernel:
        scopes.append(agent.use(kernel, workspace=str(ws)))
    if ws.is_dir():
        scopes += [_mount_file(agent, p) for p in sorted(ws.glob("*.py"))
                   if not p.name.startswith("_")]
    return scopes


# ---- self-check --------------------------------------------------------

CALC_V1 = '''
from core import Tool, ToolSpec

def setup(ctx):
    ctx.tool(Tool(ToolSpec("calc", "Add a and b.",
                           {"type": "object", "required": ["a", "b"],
                            "properties": {"a": {"type": "number"},
                                           "b": {"type": "number"}}}),
                  handler=lambda cctx, a, b: a + b))
'''


async def _selfcheck() -> None:
    import tempfile

    from core import Agent, LLMDelta, LLMReply, Message, ToolCall

    class Scripted:
        def __init__(self, replies):
            self.replies = list(replies)

        async def stream(self, messages, tools, **params):
            reply = self.replies.pop(0)
            if reply.message.content:
                yield LLMDelta(text=reply.message.content)
            yield reply

    def call(cid, name, args):
        return LLMReply(message=Message(
            role="assistant", tool_calls=[ToolCall(cid, name, args)]),
            stop_reason="tool_use")

    def text(t):
        return LLMReply(message=Message(role="assistant", content=t))

    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / "calc.py").write_text(CALC_V1)

        # boot: kernel + the workspace file, both mounted
        agent = Agent(Scripted([]), max_steps=8)
        scopes = boot(agent, ws)
        assert [s.name for s in scopes] == ["kernel", "calc"]
        assert "adapter_load" in agent.tools and "calc" in agent.tools

        # the full loop: agent inspects itself, unloads calc, reloads it,
        # calls it — every self-change is an ordinary, logged tool step
        agent.llm = Scripted([
            call("c1", "adapter_list", {}),
            call("c2", "adapter_unload", {"name": "calc"}),
            call("c3", "adapter_load", {"path": "calc.py"}),
            call("c4", "calc", {"a": 2, "b": 3}),
            text("done"),
        ])
        result = await agent.run("evolve")
        assert result.status == "completed"
        results = [m.tool_result for m in result.messages if m.role == "tool"]
        assert not any(r.is_error for r in results), results
        assert results[3].content == 5

        # jail: escapes and non-.py fail as readable tool errors
        agent.llm = Scripted([call("c5", "adapter_load",
                                   {"path": "../evil.py"}), text("ok")])
        result = await agent.run("try escape")
        bad = [m.tool_result for m in result.messages if m.role == "tool"][0]
        assert bad.is_error and "load only from" in bad.content

        # reload picks up an edit
        (ws / "calc.py").write_text(CALC_V1.replace("a + b", "a * b"))
        agent.llm = Scripted([call("c6", "adapter_reload", {"name": "calc"}),
                              call("c7", "calc", {"a": 2, "b": 3}), text("ok")])
        result = await agent.run("reload")
        results = [m.tool_result for m in result.messages if m.role == "tool"]
        assert results[1].content == 6

        # self-lockdown: dropping the kernel removes the four tools
        agent.drop("kernel")
        assert "adapter_load" not in agent.tools

    print("kernel selfcheck ok")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_selfcheck())
