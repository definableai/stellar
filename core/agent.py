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

Everything else in the package exists to serve these ~200 lines.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any, AsyncIterator, Iterable, Mapping

from .events import StepKind, StepPhase
from .hooks import HookFn, HookPoint, Hooks, LLMHookContext, ToolHookContext
from .llm import LLM, LLMDelta, LLMReply
from .run import RunContext, RunHandle, RunResult
from .tools import Tool, ToolCallContext, call_maybe_async
from .tracer import Tracer
from .types import ErrorInfo, Message, ToolResult, Usage, new_id

START, DELTA, END = StepPhase.START, StepPhase.DELTA, StepPhase.END


class Agent:
    def __init__(
        self,
        llm: LLM,
        *,
        tools: Iterable[Tool] = (),
        hooks: Hooks | Mapping[HookPoint, Iterable[HookFn]] | None = None,
        tracers: Iterable[Tracer] = (),
        system: str | None = None,
        max_steps: int = 8,
        parallel_tools: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.tools: dict[str, Tool] = {t.spec.name: t for t in tools}
        self.hooks = hooks if isinstance(hooks, Hooks) else Hooks(hooks)
        self.tracers = list(tracers)
        self.system = system
        self.max_steps = max_steps
        self.parallel_tools = parallel_tools
        self.params = params or {}

    # ---- public API ----------------------------------------------------

    def run(
        self,
        input: str | Message | Iterable[Message],
        *,
        history: Iterable[Message] | None = None,
        state: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunHandle:
        """Start a run. Returns immediately; events stream on the handle.

        ``await agent.run(...)`` yields the RunResult; keep the handle
        un-awaited to stream events instead.
        Must be called inside a running asyncio event loop.
        """
        messages = self._build_messages(input, history)
        handle = RunHandle(run_id, self.tracers)
        ctx = RunContext(run_id=handle.run_id, handle=handle, state=state or {})
        handle._task = asyncio.create_task(self._run(handle, ctx, messages))
        return handle

    # ---- internals -------------------------------------------------------

    def _build_messages(self, input, history) -> list[Message]:
        messages: list[Message] = []
        if self.system:
            messages.append(Message(role="system", content=self.system))
        if history:
            messages.extend(history)
        if isinstance(input, str):
            messages.append(Message(role="user", content=input))
        elif isinstance(input, Message):
            messages.append(input)
        else:
            messages.extend(input)
        return messages

    async def _run(
        self, h: RunHandle, ctx: RunContext, messages: list[Message]
    ) -> None:
        run_step = new_id("run")
        usage = Usage()
        output: str | None = None
        status: str = "completed"
        stop_reason: str | None = None
        error: ErrorInfo | None = None
        steps = 0

        await h.emit(StepKind.RUN, START,
                     {"messages": len(messages), "run_id": h.run_id}, run_step)
        try:
            for step_index in range(self.max_steps):
                steps = step_index + 1

                if h._inbox:                    # mid-run steering (handle.send)
                    messages.extend(h._inbox)
                    h._inbox.clear()

                # -- before_llm hooks (may mutate messages/tools) ----------
                hctx = LLMHookContext(
                    run=ctx,
                    messages=messages,
                    tools=[t.spec for t in self.tools.values()],
                )
                await self._fire(h, "before_llm", hctx)

                # -- LLM step ----------------------------------------------
                sid = new_id("text")
                await h.emit(StepKind.TEXT, START, {"step": step_index}, sid)
                reply: LLMReply | None = None
                partial = False
                stream = self.llm.stream(list(messages), hctx.tools, **self.params)
                try:
                    async for item in stream:
                        if isinstance(item, LLMReply):
                            reply = item
                        elif isinstance(item, LLMDelta) and item.text:
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

                if reply is None:  # stopped mid-stream: synthesize from deltas
                    text = "".join(
                        e.payload["text"] for e in h._buffer
                        if e.step_id == sid and e.phase is DELTA
                        and e.payload["channel"] == "text"
                    )
                    reply = LLMReply(
                        message=Message(role="assistant", content=text or None),
                        stop_reason="stopped",
                    )
                usage = usage + reply.usage
                await h.emit(StepKind.TEXT, END, {
                    "text": reply.message.content,
                    "tool_calls": [c.to_dict() for c in reply.message.tool_calls],
                    "stop_reason": reply.stop_reason, "partial": partial,
                    "usage": reply.usage.to_dict(),
                }, sid)

                # -- after_llm hooks (may mutate reply) ---------------------
                hctx.reply = reply.message
                await self._fire(h, "after_llm", hctx)
                messages.append(reply.message)
                output = reply.message.content

                if h.stop_requested:
                    status, stop_reason = "stopped", h.stop_reason
                    break
                if not reply.message.tool_calls:
                    break

                # -- tool steps ---------------------------------------------
                calls = reply.message.tool_calls
                # ponytail: one unsafe tool serializes the whole batch
                if (self.parallel_tools and len(calls) > 1
                        and all(t is None or t.parallel_safe
                                for t in (self.tools.get(c.name) for c in calls))):
                    tool_msgs = await asyncio.gather(
                        *(self._exec_tool(h, ctx, c) for c in calls)
                    )
                else:
                    tool_msgs = [await self._exec_tool(h, ctx, c) for c in calls]
                messages.extend(tool_msgs)

                if h.stop_requested:
                    status, stop_reason = "stopped", h.stop_reason
                    break
            else:
                stop_reason = "max_steps"
        except Exception as ex:
            status = "error"
            error = ErrorInfo.from_exc(ex, "run", traceback.format_exc(limit=8))

        if h._inbox:            # late sends still land in the transcript
            messages.extend(h._inbox)
            h._inbox.clear()
        await h.emit(StepKind.RUN, END, {
            "status": status, "steps": steps, "usage": usage.to_dict(),
            "stop_reason": stop_reason,
            "error": error.to_dict() if error else None,
        }, run_step)
        h._finish(RunResult(
            run_id=h.run_id, status=status,  # type: ignore[arg-type]
            messages=messages, output=output, usage=usage,
            error=error, stop_reason=stop_reason,
        ))

    async def _exec_tool(self, h: RunHandle, ctx: RunContext, call) -> Message:
        """One tool call: start -> before hooks -> exec -> after hooks -> end.

        Never raises: failures become ToolResult(is_error=True) so the
        LLM can see and recover from them. Hook failures still abort
        the run (fail closed) — they propagate out of _fire.
        """
        sid = new_id("tool")
        await h.emit(StepKind.TOOL, START, {
            "call_id": call.id, "name": call.name, "arguments": call.arguments,
        }, sid)
        tctx = ToolHookContext(run=ctx, call=call)
        await self._fire(h, "before_tool", tctx)

        err: ErrorInfo | None = None
        if tctx.result is None:  # not short-circuited by a hook
            tool = self.tools.get(call.name)
            if tool is None:
                err = ErrorInfo("UnknownTool", f"Unknown tool: {call.name}", "tool")
                tctx.result = ToolResult(
                    call.id, call.name, err.message, is_error=True
                )
            else:
                cctx = ToolCallContext(
                    run=ctx,
                    call=call,
                    emit_delta=lambda payload: h.emit(
                        StepKind.TOOL, DELTA, {"call_id": call.id, **payload}, sid
                    ),
                )
                exec_task = asyncio.ensure_future(
                    call_maybe_async(tool.handler, cctx, **call.arguments)
                )
                stop_task = asyncio.ensure_future(h._stop.wait())
                done, _ = await asyncio.wait(
                    {exec_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
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
                else:  # stop requested mid-execution
                    exec_task.cancel()
                    err = ErrorInfo("Cancelled", "cancelled", "tool")
                    tctx.result = ToolResult(
                        call.id, call.name, "cancelled", is_error=True
                    )

        await self._fire(h, "after_tool", tctx)
        result = tctx.result
        await h.emit(StepKind.TOOL, END, {
            "call_id": call.id, "name": call.name,
            "result": result.content, "is_error": result.is_error,
            **({"error": err.to_dict()} if err else {}),
        }, sid)
        return Message(role="tool", tool_result=result)

    async def _fire(self, h: RunHandle, point: HookPoint, hctx: Any) -> None:
        """Fire all hooks at a point; each is its own hook step."""
        for fn in self.hooks.get(point):
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
