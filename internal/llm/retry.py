"""RetryLLM — wrap any adapter with backoff on retryable failures.

    agent = Agent(RetryLLM(OpenAILLM(...), attempts=3))

Retries ``LLMError.retryable`` (408/429/5xx) and httpx transport errors
(timeouts, connection failures) with jittered exponential backoff.
Retries ONLY when nothing has been yielded yet — after the first delta,
errors propagate, because replaying would duplicate streamed output.
In practice that covers the real cases: rate limits and connect errors
fail before the first token.

Self-check: uv run python -m internal.llm.retry
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, AsyncIterator, Sequence

import httpx

from core import LLMDelta, LLMError, LLMReply, Message, ToolSpec


class RetryLLM:
    def __init__(self, inner: Any, attempts: int = 3,
                 base_delay: float = 1.0, max_delay: float = 30.0):
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], **params: Any
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        for attempt in range(self.attempts):
            yielded = False
            try:
                async for item in self.inner.stream(messages, tools, **params):
                    yielded = True
                    yield item
                return
            except LLMError as e:
                if yielded or not e.retryable or attempt == self.attempts - 1:
                    raise
            except httpx.TransportError:
                if yielded or attempt == self.attempts - 1:
                    raise
            await asyncio.sleep(min(self.max_delay, self.base_delay * 2 ** attempt)
                                * (0.5 + random.random() / 2))


if __name__ == "__main__":

    async def _selfcheck() -> None:
        class Flaky:
            """Fails n times with the given error, then succeeds."""

            def __init__(self, n: int, exc: Exception):
                self.n, self.exc, self.calls = n, exc, 0

            async def stream(self, messages, tools, **params):
                self.calls += 1
                if self.calls <= self.n:
                    raise self.exc
                yield LLMDelta(text="ok")
                yield LLMReply(message=Message(role="assistant", content="ok"))

        # retryable: fails twice, third succeeds
        inner = Flaky(2, LLMError(429, "slow down"))
        r = RetryLLM(inner, attempts=3, base_delay=0.01)
        got = [x async for x in r.stream([], [])]
        assert inner.calls == 3 and got[-1].message.content == "ok"

        # fatal 401: no retry
        inner = Flaky(5, LLMError(401, "bad key"))
        r = RetryLLM(inner, attempts=3, base_delay=0.01)
        try:
            _ = [x async for x in r.stream([], [])]
            raise AssertionError("expected LLMError")
        except LLMError as e:
            assert e.status == 401 and inner.calls == 1

        # exhausted attempts: raises after N tries
        inner = Flaky(9, LLMError(500, "boom"))
        r = RetryLLM(inner, attempts=2, base_delay=0.01)
        try:
            _ = [x async for x in r.stream([], [])]
            raise AssertionError("expected LLMError")
        except LLMError:
            assert inner.calls == 2

        # transport error retried
        inner = Flaky(1, httpx.ConnectError("refused"))
        r = RetryLLM(inner, attempts=2, base_delay=0.01)
        got = [x async for x in r.stream([], [])]
        assert inner.calls == 2 and got[-1].message.content == "ok"

        # partial output then error: NO retry (would duplicate deltas)
        class MidFail:
            calls = 0

            async def stream(self, messages, tools, **params):
                MidFail.calls += 1
                yield LLMDelta(text="partial")
                raise LLMError(529, "overloaded mid-stream")

        r = RetryLLM(MidFail(), attempts=3, base_delay=0.01)
        try:
            _ = [x async for x in r.stream([], [])]
            raise AssertionError("expected LLMError")
        except LLMError as e:
            assert e.status == 529 and MidFail.calls == 1

        print("retry wrapper self-check ok")

    asyncio.run(_selfcheck())
