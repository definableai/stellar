"""A chat that survives kill -9 — the conversation *is* a Session file.

Every message lands in one append-only JSONL log the moment it happens:
your prompt, the model's reply, each tool call, each result. Kill the
process anywhere in that sequence and rerun: ``Session.load`` drops a
torn tail, answers tool calls the log never got results for (marked
``repaired``), and hands the loop a valid transcript to keep going from.

    uv run python -m examples.durable_chat            # offline demo, no network

    export OPENAI_API_KEY=...
    uv run python -m examples.durable_chat live "hi, I'm Ana"
    uv run python -m examples.durable_chat live "what's my name?"   # remembers

Live mode appends to ./chat.jsonl — delete it to start a fresh chat.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from core import (Agent, LLMReply, Message, Session, StepKind, StepPhase,
                  ToolCall)
from internal.tools.schema import tool

CHAT = Path("chat.jsonl")


@tool()
def line_count(ctx, path: str) -> str:
    """Count the lines in a text file.

    Args:
        path: file to count
    """
    return f"{path}: {len(Path(path).read_text().splitlines())} lines"


def build(llm, **kw) -> Agent:
    return Agent(llm=llm, tools=[line_count], **kw,
                 system="You are a terse assistant with perfect recall of this chat.")


def open_chat(path: Path) -> Session:
    """The whole resume story: an existing log loads (repairing itself)."""
    return (Session.load(path) if path.exists() and path.stat().st_size
            else Session(path))

# ---- live ---------------------------------------------------------------


async def live(prompt: str) -> None:
    """One turn appended to ./chat.jsonl. Run it again — it remembers."""
    from internal.llm.openai import OpenAILLM

    agent = build(OpenAILLM(model="gpt-5.6-luna"))   # key check before we touch the log
    with open_chat(CHAT) as s:
        print(f"[{s.id}] resuming {len(s)} messages\n")
        handle = agent.run(prompt, session=s)
        async for e in handle:
            if (e.kind is StepKind.TEXT and e.phase is StepPhase.DELTA
                    and e.payload["channel"] == "text"):
                print(e.payload["text"], end="", flush=True)
        print(f"\n\n[{(await handle).status}] {len(s)} messages in {CHAT}")

# ---- offline demo -------------------------------------------------------


class ScriptedLLM:
    """Canned replies, one per step; keeps every request it was handed."""

    def __init__(self, *replies: LLMReply) -> None:
        self.replies, self.seen = list(replies), []

    async def stream(self, messages, tools, **params):
        self.seen.append(list(messages))
        yield self.replies.pop(0)


def says(text: str | None = None, calls=()) -> LLMReply:
    return LLMReply(Message(role="assistant", content=text, tool_calls=list(calls)))


async def demo() -> None:
    with tempfile.TemporaryDirectory() as d:
        chat, notes = Path(d) / "chat.jsonl", Path(d) / "notes.txt"
        notes.write_text("alpha\nbeta\ngamma\n")

        # -- process 1: dies between the tool call and its result ---------
        # The loop appends each message as it happens, so a SIGKILL here
        # leaves the call on disk unanswered — written by hand because the
        # demo has to outlive its own crash.
        with Session(chat) as s:
            s.append(Message(role="user", content=f"how long is {notes.name}?"))
            s.append(Message(role="assistant", tool_calls=[
                ToolCall("c1", "line_count", {"path": str(notes)})]))

        # -- process 2: load repairs the dangling call, then just continue -
        with Session.load(chat) as s:
            assert [m.role for m in s.messages()] == ["user", "assistant", "tool"]
            fixed = s.messages()[-1]
            assert fixed.meta.get("repaired") and fixed.tool_result.is_error
            assert fixed.tool_result.call_id == "c1"    # the call the crash orphaned
            assert "interrupted" in fixed.tool_result.content

            llm = ScriptedLLM(
                says(calls=[ToolCall("c2", "line_count", {"path": str(notes)})]),
                says(f"{notes.name} has 3 lines."))
            r = await build(llm).run("that died — try again", session=s)

        assert r.status == "completed" and "3 lines" in r.output
        # the retry request carried the repaired history, system prompt first
        assert [m.role for m in llm.seen[0]] == [
            "system", "user", "assistant", "tool", "user"]

        # -- process 3: nothing in memory, everything still remembered ----
        with Session.load(chat) as s:
            llm2 = ScriptedLLM(says(f"You asked how long {notes.name} is: 3 lines."))
            r2 = await build(llm2).run("what did I ask you first?", session=s)

        assert r2.status == "completed" and notes.name in r2.output
        assert len(llm2.seen[0]) == 9        # system + all 8 messages on disk

        with Session.load(chat) as s:        # the log is the whole truth
            assert [m.role for m in s.messages()] == [
                "user", "assistant", "tool",                 # crash, repaired
                "user", "assistant", "tool", "assistant",    # the retry
                "user", "assistant"]                         # and the turn after
    print("durable chat demo ok")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["live"]:
        asyncio.run(live(" ".join(argv[1:]) or "hello"))
    else:
        asyncio.run(demo())
