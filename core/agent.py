"""The Agent: the loop. This is the whole heart.

    run/start
    repeat (up to max_steps):
        fire before_llm hooks                 (hook/start..end each)
        text/start
            stream LLM -> text/delta*         (stop checked per chunk)
        text/end
        fire after_llm hooks
        no tool calls? -> break
        for each tool call (sequential or parallel):
            tool/start
            fire before_tool hooks            (may short-circuit)
            execute handler                   (may emit tool/delta*)
            fire after_tool hooks             (may rewrite result)
            tool/end
    run/end
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Callable, Iterable, Mapping

from .adapter import Ctx, Scope
from .events import StepKind, StepPhase
from .hooks import Hook, HookFn, HookPoint, Hooks, LLMHookContext, ToolHookContext
from .llm import LLM, LLMDelta, LLMReply
from .run import RunContext, RunHandle, RunResult, RunStatus
from .session import Session
from .tools import Tool, ToolCallContext, call_maybe_async, validate_args
from .tracer import Tracer
from .types import ErrorInfo, Message, ToolCall, ToolResult, Usage, new_id

START, DELTA, END = StepPhase.START, StepPhase.DELTA, StepPhase.END


class Agent:
    def __init__(
        self,
        llm: LLM,
        *,
        tools: Iterable[Tool] = (),
        hooks: Hooks | Iterable[Hook]
        | Mapping[HookPoint, Iterable[HookFn]] | None = None,
        tracers: Iterable[Tracer] = (),
        system: str | None = None,
        max_steps: int = 8,
        parallel_tools: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(llm, LLM):   # runtime_checkable: stream must exist
            raise TypeError(
                "Agent needs an LLM adapter: an object with "
                "async stream(messages, tools, **params) -> LLMDelta/LLMReply "
                "— see core/llm.py")
        self.llm = llm
        self.tools: dict[str, Tool] = {t.spec.name: t for t in tools}
        self.hooks = hooks if isinstance(hooks, Hooks) else Hooks(hooks)
        self.tracers = list(tracers)
        self.system = system
        self.max_steps = max_steps
        self.parallel_tools = parallel_tools
        self.params = params or {}
        self.adapters: dict[str, Scope] = {}

    # ---- public API ----------------------------------------------------

    def run(
        self,
        input: str | Message | Iterable[Message],
        *,
        session: Session | None = None,
        history: Iterable[Message] | None = None,
        state: dict[str, Any] | None = None,
        run_id: str | None = None,
        params: dict[str, Any] | None = None,   # per-run LLM param overrides
    ) -> RunHandle:
        """Start a run; events stream on the returned handle. ``await
        agent.run(...)`` yields the RunResult; keep the handle un-awaited
        to stream instead. Needs a running asyncio loop. With ``session=``
        the request derives from the log and appends through it."""
        if session is not None:
            if history is not None:
                raise ValueError("pass session or history, not both")
            new = self._input_messages(input)
            for m in new:            # validate the whole batch before any
                json.dumps(m.to_dict())  # append lands (atomic input)
            for m in new:
                session.append(m)
            messages = self._build_messages(session.messages(), None)
        else:
            messages = self._build_messages(input, history)
        handle = RunHandle(run_id, self.tracers)
        # state: `is not None`, not truthiness — a caller-held dict (even
        # empty) is shared identity across runs (worker turn scratch)
        ctx = RunContext(run_id=handle.run_id, handle=handle, agent=self,
                         state=state if state is not None else {})
        merged = {**self.params, **(params or {}),   # dicts deep-merge one level
                  **{k: {**self.params[k], **v} for k, v in (params or {}).items()
                     if isinstance(v, dict) and isinstance(self.params.get(k), dict)}}
        handle._task = asyncio.create_task(
            self._run(handle, ctx, messages, merged, session))
        return handle

    def use(self, setup: Callable[[Ctx], Any], /, name: str | None = None,
            source: str | None = None, **config: Any) -> Scope:
        """Mount an adapter — a ``setup(ctx)`` function: registrations
        record inverses; a raising setup mounts nothing (core/adapter.py)."""
        if not callable(setup):
            raise TypeError(f"not an adapter: {setup!r}")
        name = name or getattr(setup, "__name__", "adapter")
        if name == "setup":   # a module's entrypoint: name by the module
            name = getattr(setup, "__module__", "adapter").rsplit(".", 1)[-1]
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} already mounted")
        scope = Scope(name, source)
        scope._remount = (setup, config)   # drop() rebuilds those above
        try:
            setup(Ctx(self, scope, config))
        except BaseException as ex:
            for e in scope.dispose():      # partial-unwind failures stay visible
                ex.add_note(f"inverse failed during unwind: {e.type}: {e.message}")
            raise
        self.adapters[name] = scope
        return scope

    def drop(self, name: str) -> list[ErrorInfo]:
        """Unmount. Adapters mounted after ``name`` unwind first and
        remount on the new base — recorded priors never go stale.
        Inverse failures are returned, never raised; a raising remount
        propagates, leaving it and later adapters unmounted (loud)."""
        if name not in self.adapters:
            raise KeyError(name)
        order = list(self.adapters)
        above = [self.adapters[n] for n in order[order.index(name) + 1:]]
        errors: list[ErrorInfo] = []
        for s in reversed(above):
            errors += self.adapters.pop(s.name).dispose()
        errors += self.adapters.pop(name).dispose()
        for s in above:
            setup, config = s._remount
            self.use(setup, name=s.name, source=s.source, **config)
        return errors

    # ---- internals -------------------------------------------------------

    def _input_messages(
        self, input: str | Message | Iterable[Message]) -> list[Message]:
        if isinstance(input, str):
            return [Message(role="user", content=input)]
        return [input] if isinstance(input, Message) else list(input)

    def _build_messages(
        self, input: str | Message | Iterable[Message],
        history: Iterable[Message] | None) -> list[Message]:
        sys_ = [Message(role="system", content=self.system)] if self.system else []
        return [*sys_, *(history or []), *self._input_messages(input)]

    async def _run(
        self, h: RunHandle, ctx: RunContext, messages: list[Message],
        params: dict[str, Any], session: Session | None,
    ) -> None:
        run_step = new_id("run")
        usage = Usage()
        output: str | None = None
        status: RunStatus = "completed"
        stop_reason: str | None = None
        error: ErrorInfo | None = None
        steps = 0

        def record(msg: Message) -> None:
            """Every new message lands in the transcript and the session log."""
            messages.append(msg)
            if session is not None:
                session.append(msg)

        await h.emit(StepKind.RUN, START,
                     {"messages": len(messages), "run_id": h.run_id}, run_step)
        try:
            for step_index in range(self.max_steps):
                steps = step_index + 1

                if h._inbox:                    # mid-run steering (handle.send)
                    for m in h._inbox:
                        record(m)
                    h._inbox.clear()

                # -- before_llm hooks (may mutate messages/tools) ----------
                hctx = LLMHookContext(
                    run=ctx,
                    messages=messages,
                    tools=[t.spec for t in self.tools.values()],
                )
                await self._fire(h, "before_llm", hctx)
                messages = hctx.messages   # hooks may rebind, not just mutate

                # -- LLM step ----------------------------------------------
                sid = new_id("text")
                await h.emit(StepKind.TEXT, START, {"step": step_index}, sid)
                reply: LLMReply | None = None
                partial = False
                parts: list[str] = []   # for stop-mid-stream synthesis
                stream = self.llm.stream(list(messages), hctx.tools, **params)
                try:
                    async for item in stream:
                        if isinstance(item, LLMReply):
                            reply = item
                        elif isinstance(item, LLMDelta) and item.text:
                            if item.channel == "text":
                                parts.append(item.text)
                            await h.emit(StepKind.TEXT, DELTA,
                                         {"text": item.text, "channel": item.channel,
                                          "index": item.index}, sid)
                        if h.stop_requested:
                            partial = reply is None
                            break
                finally:
                    aclose = getattr(stream, "aclose", None)
                    if aclose:
                        await aclose()

                if reply is None:  # stopped mid-stream: synthesize the partial
                    reply = LLMReply(
                        message=Message(role="assistant",
                                        content="".join(parts) or None,
                                        meta={"interrupted": True}),
                        stop_reason="stopped",
                    )
                usage = ctx.usage = usage + reply.usage
                ctx.last_usage = reply.usage
                await h.emit(StepKind.TEXT, END, {
                    "text": reply.message.content,
                    "tool_calls": [c.to_dict() for c in reply.message.tool_calls],
                    "stop_reason": reply.stop_reason, "partial": partial,
                    "usage": reply.usage.to_dict(),
                }, sid)

                # -- after_llm hooks (may mutate reply) ---------------------
                hctx.reply = reply.message
                await self._fire(h, "after_llm", hctx)
                record(reply.message)
                output = (reply.message.content
                          if isinstance(reply.message.content, str) else None)

                if h.stop_requested:
                    status, stop_reason = "stopped", h.stop_reason
                    break
                if not reply.message.tool_calls:
                    break

                # -- tool steps ---------------------------------------------
                calls = reply.message.tool_calls
                bad_args: dict[str, str] = (
                    reply.message.meta.get("invalid_tool_args") or {})
                # ponytail: one unsafe tool serializes the whole batch
                if (self.parallel_tools and len(calls) > 1
                        and all(t is None or t.parallel_safe
                                for t in (self.tools.get(c.name) for c in calls))):
                    settled = await asyncio.gather(
                        *(self._exec_tool(h, ctx, c, malformed=c.id in bad_args)
                          for c in calls),
                        return_exceptions=True,
                    )
                    # hook failure: fail closed after siblings settle —
                    # their real results stay in the transcript
                    exc = next((m for m in settled if isinstance(m, BaseException)),
                               None)
                    tool_msgs = [m for m in settled if isinstance(m, Message)]
                else:
                    exc = None
                    tool_msgs = [await self._exec_tool(
                        h, ctx, c, malformed=c.id in bad_args) for c in calls]
                for m in tool_msgs:
                    record(m)
                if exc is not None:
                    raise exc

                if h.stop_requested:
                    status, stop_reason = "stopped", h.stop_reason
                    break
            else:
                status, stop_reason = "truncated", "max_steps"
        except Exception as ex:
            status = "error"
            error = ErrorInfo.from_exc(ex, "run", traceback.format_exc(limit=8))

        try:
            if h._inbox:        # late sends still land in the transcript
                for m in h._inbox:
                    record(m)
                h._inbox.clear()
        except Exception as ex:  # a failing append must never hang the handle
            status = "error"
            error = error or ErrorInfo.from_exc(ex, "run", traceback.format_exc(limit=8))
        h._inbox_closed = True   # sends past this point report unaccepted
        await h.emit(StepKind.RUN, END, {
            "status": status, "steps": steps, "usage": usage.to_dict(),
            "stop_reason": stop_reason,
            "error": error.to_dict() if error else None,
        }, run_step)
        h._finish(RunResult(
            run_id=h.run_id, status=status,
            messages=messages, output=output, usage=usage,
            error=error, stop_reason=stop_reason,
        ))

    async def _exec_tool(self, h: RunHandle, ctx: RunContext, call: ToolCall,
                         *, malformed: bool = False) -> Message:
        """One tool call: start -> before hooks -> exec -> after hooks -> end.
        Never raises: failures become ToolResult(is_error=True) so the
        LLM can see and recover. Hook failures still abort the run
        (fail closed) — they propagate out of _fire."""
        sid = new_id("tool")
        await h.emit(StepKind.TOOL, START, {
            "call_id": call.id, "name": call.name, "arguments": call.arguments,
        }, sid)
        async def emit_delta(payload: dict[str, Any]) -> Any:
            return await h.emit(StepKind.TOOL, DELTA,
                                {"call_id": call.id, **payload}, sid)

        tctx = ToolHookContext(run=ctx, call=call, emit_delta=emit_delta)
        await self._fire(h, "before_tool", tctx)

        err: ErrorInfo | None = None
        if tctx.result is None:  # not short-circuited by a hook
            tool = self.tools.get(call.name)
            problems = ([] if tool is None or malformed or not tool.validate
                        else validate_args(tool.spec.parameters, call.arguments))
            if malformed:        # adapter flagged unparseable argument JSON
                err = ErrorInfo("MalformedArguments",
                                "invalid or truncated tool-call JSON", "tool")
                tctx.result = ToolResult(
                    call.id, call.name,
                    "Malformed tool arguments (invalid or truncated JSON). "
                    "Re-issue the call.", is_error=True)
            elif tool is None:
                err = ErrorInfo("UnknownTool", f"Unknown tool: {call.name}", "tool")
                tctx.result = ToolResult(
                    call.id, call.name, err.message, is_error=True
                )
            elif problems:       # schema guardrail: readable, model-facing
                err = ErrorInfo("InvalidArguments", "; ".join(problems), "tool")
                tctx.result = ToolResult(
                    call.id, call.name,
                    f"Invalid arguments: {'; '.join(problems)}", is_error=True)
            else:
                cctx = ToolCallContext(run=ctx, call=call, emit_delta=emit_delta)
                exec_task = asyncio.ensure_future(
                    call_maybe_async(tool.handler, cctx, **call.arguments)
                )
                stop_task = asyncio.ensure_future(h._stop.wait())
                done, _ = await asyncio.wait(
                    {exec_task, stop_task}, timeout=tool.timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                if exec_task in done:
                    stop_task.cancel()
                    try:
                        tctx.result = ToolResult(call.id, call.name, exec_task.result())
                    except Exception as ex:
                        err = ErrorInfo.from_exc(ex, "tool")
                        tctx.result = ToolResult(
                            call.id, call.name,
                            f"{err.type}: {err.message}", is_error=True,
                        )
                else:  # stop requested mid-execution, or tool.timeout hit
                    exec_task.cancel()
                    stop_task.cancel()
                    stopped = stop_task in done
                    why = "cancelled" if stopped else f"timeout after {tool.timeout}s"
                    err = ErrorInfo("Cancelled" if stopped else "Timeout", why, "tool")
                    tctx.result = ToolResult(call.id, call.name, why, is_error=True)

        await self._fire(h, "after_tool", tctx)
        result = tctx.result
        await h.emit(StepKind.TOOL, END, {
            "call_id": call.id, "name": call.name,
            "result": result.content, "is_error": result.is_error,
            **({"error": err.to_dict()} if err else {}),
        }, sid)
        return Message(role="tool", tool_result=result)

    async def _fire(self, h: RunHandle, point: HookPoint,
                    hctx: LLMHookContext | ToolHookContext) -> None:
        """Fire all hooks at a point, each its own step. Snapshot: a
        hook mutating composition must not skip a sibling."""
        for fn in list(self.hooks.get(point)):
            sid = new_id("hook")
            name = getattr(fn, "__qualname__", repr(fn))
            await h.emit(StepKind.HOOK, START, {"point": point, "hook": name}, sid)
            try:
                await call_maybe_async(fn, hctx)
            except Exception as ex:
                await h.emit(
                    StepKind.HOOK, END,
                    {"point": point, "hook": name,
                     "error": ErrorInfo.from_exc(ex, "hook").to_dict()},
                    sid,
                )
                raise  # fail closed: a broken guardrail aborts the run
            await h.emit(StepKind.HOOK, END, {"point": point, "hook": name}, sid)
