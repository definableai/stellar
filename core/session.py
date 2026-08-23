"""Durable sessions: the append-only log a long-lived agent lives on.

One JSONL file per session; the log is the source of truth for the
conversation. A run started with ``agent.run(..., session=s)`` derives
its request messages from the log and appends every new message through
it, so the durable transcript and the live one can never diverge.

    {"v": 0, "id": "session_ab12cd34ef56"}                  <- header
    {"seq": 0, "type": "message", "data": {...Message...}}
    {"seq": 1, "type": "message", "data": {...Message...}}

Rules:
    * Append-only. Compaction/pruning are *views* (before_llm hooks
      mutate the derived request list); the log is never rewritten.
    * Messages only, no deltas — a crash loses at most the partial text
      of the step in flight. ponytail: message granularity; chunk-level
      persistence only if token-exact crash recovery ever matters.
    * The system message is agent config, not conversation — it is
      prepended by the loop and never stored.
    * ``load()`` repairs: a crash mid-tool-batch leaves dangling tool
      calls; they get synthetic error results (marked ``repaired`` in
      meta) so the next request is always a valid transcript.
    * Message content must be JSON-serializable; ``append`` is
      all-or-nothing, so a bad payload fails loudly instead of letting
      memory and disk diverge.
    * Single writer per file, enforced by an advisory flock on the
      append handle — a second open (other process or this one) raises.
      One run at a time within that writer: two concurrent runs on one
      session interleave into a provider-invalid transcript. ponytail:
      no run guard; Worker serializes runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

from .types import Message, ToolResult, new_id

FORMAT_VERSION = 0


class SessionError(ValueError):
    """Corrupt, incompatible, or already-claimed session file."""


def _open_locked(p: Path) -> IO[str]:
    """Append handle holding an exclusive advisory lock: a second writer
    (any process, or a second load in this one) fails loudly instead of
    interleaving records. Released by close(). ponytail: no-op where
    fcntl is missing (Windows)."""
    f = p.open("a", encoding="utf-8")
    try:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        pass
    except OSError:
        f.close()
        raise SessionError(f"session file locked by another writer: {p}") from None
    return f


class Session:
    def __init__(self, path: str | Path | None = None, id: str | None = None) -> None:
        """New session. ``path=None`` is in-memory (tests, ephemeral runs)."""
        self.id = id or new_id("session")
        self.path = Path(path) if path else None
        self._messages: list[Message] = []
        self._seq = 0
        self._file: IO[str] | None = None
        if self.path:
            if self.path.exists() and self.path.stat().st_size:
                raise SessionError(f"session file exists, use Session.load(): {self.path}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = _open_locked(self.path)
            self._write({"v": FORMAT_VERSION, "id": self.id})

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        """Rebuild from disk, repair dangling tool calls, ready to continue."""
        p = Path(path)
        raw = p.read_bytes()  # bytes: a write torn mid-codepoint must not
        # kill the decode of the good prefix. b"\n" is the only record
        # separator (splitlines() would also split on U+2028/U+2029/U+0085,
        # which json.dumps emits raw inside strings).
        lines = raw.split(b"\n")
        torn = None if lines[-1] == b"" else lines[-1]  # no trailing \n: crash mid-write
        lines = lines[:-1]
        header = cls._parse(lines[0], 1) if lines else None
        if not isinstance(header, dict) or header.get("v") != FORMAT_VERSION:
            raise SessionError(f"bad session header: {p}")
        s = cls.__new__(cls)
        s.id, s.path, s._messages, s._seq, s._file = header["id"], p, [], 0, None
        for n, line in enumerate(lines[1:], start=2):
            rec = cls._parse(line, n)
            if rec.get("seq") != s._seq or rec.get("type") != "message":
                raise SessionError(f"line {n}: bad record (seq {rec.get('seq')}, "
                                   f"expected {s._seq}, type {rec.get('type')!r})")
            s._messages.append(Message.from_dict(rec["data"]))
            s._seq += 1
        if torn is not None:  # truncate, or the next append corrupts this line
            with p.open("r+b") as f:
                f.truncate(len(raw) - len(torn))
        s._file = _open_locked(p)
        s._repair()
        return s

    def append(self, message: Message) -> None:
        """All-or-nothing: on failure (closed session, non-JSON-serializable
        content) nothing lands in memory or on disk."""
        if self.path is not None and self._file is None:
            raise SessionError(f"session is closed: {self.path}")
        if self._file:
            self._write({"seq": self._seq, "type": "message", "data": message.to_dict()})
        self._messages.append(message)
        self._seq += 1

    def messages(self) -> list[Message]:
        """The conversation: fresh list, shared Message objects."""
        return list(self._messages)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._messages)

    # ---- internals ---------------------------------------------------

    def _write(self, obj: dict[str, Any]) -> None:
        assert self._file is not None
        self._file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        # ponytail: flush, not fsync — an OS crash may lose the tail; repair covers it
        self._file.flush()

    @staticmethod
    def _parse(line: bytes, n: int) -> dict[str, Any]:
        try:
            return json.loads(line)  # bad JSON and bad UTF-8 both ValueError
        except ValueError:
            raise SessionError(f"corrupt session line {n}") from None

    def _repair(self) -> None:
        """Close out a crash: synthesize error results for tool calls the
        tail assistant message made but the log never answered."""
        msgs, i, answered = self._messages, len(self._messages), set()
        while i and msgs[i - 1].role == "tool":
            i -= 1
            tr = msgs[i].tool_result
            if tr:
                answered.add(tr.call_id)
        if i and msgs[i - 1].role == "assistant":
            for call in msgs[i - 1].tool_calls:
                if call.id not in answered:
                    self.append(Message(
                        role="tool", meta={"repaired": True},
                        tool_result=ToolResult(
                            call.id, call.name,
                            "tool outcome unknown: session interrupted",
                            is_error=True),
                    ))
