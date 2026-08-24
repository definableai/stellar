"""RetryLLM — wrap any adapter with backoff on retryable failures.

    agent = Agent(RetryLLM(OpenAILLM(...), attempts=3))

Retries ``LLMError.retryable`` (408/429/5xx) and httpx transport errors
(timeouts, connection failures) with jittered exponential backoff.
Retries ONLY when nothing has been yielded yet — after the first delta,
errors propagate, because replaying would duplicate streamed output.
In practice that covers the real cases: rate limits and connect errors
fail before the first token.
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


def setup(ctx: Any) -> None:
    """Adapter shape (core/adapter.py): wraps the agent's CURRENT llm —
    mount it after the one it protects. Config = RetryLLM kwargs."""
    ctx.llm(RetryLLM(ctx.agent.llm, **ctx.config))
