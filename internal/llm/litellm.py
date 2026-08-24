"""LiteLLM adapter — 100+ providers through one optional dependency.

LiteLLM speaks the OpenAI chat format on both sides, so this adapter
reuses the openai module's request transformation and maps the streamed
chunks through ReplyBuilder. Install the optional dep to use it:

    uv add litellm

    agent = Agent(LiteLLM(model="anthropic/claude-opus-5"))
    agent = Agent(LiteLLM(model="gemini/gemini-2.5-pro"))
    agent = Agent(LiteLLM(model="ollama/llama3", api_base="http://..."))

Provider params pass through verbatim (constructor defaults, per-agent,
per-run); API keys come from each provider's usual env var — that is
litellm's department. Reasoning streams as ``channel="reasoning"``
where the provider exposes ``reasoning_content``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, AsyncIterator, Callable, Sequence

from core import LLMDelta, LLMError, LLMReply, Message, ReplyBuilder, ToolSpec
from internal.llm.openai import _to_openai


def _get(o: Any, key: str, default: Any = None) -> Any:
    """Chunks are objects in current litellm, plain dicts in older
    versions and some shims — read both, never drop data silently."""
    if isinstance(o, Mapping):
        return o.get(key, default)
    return getattr(o, key, default)


# ---- adapter ----------------------------------------------------------


class LiteLLM:
    def __init__(self, model: str,
                 acompletion: Callable[..., Any] | None = None,
                 **defaults: Any):
        # acompletion= is the test seam; real use resolves litellm lazily
        # so importing this module never requires the package.
        self.model = model
        self.defaults = defaults
        self._acompletion = acompletion

    def _fn(self) -> Callable[..., Any]:
        if self._acompletion is None:
            try:
                import litellm
            except ImportError:
                raise ImportError("the LiteLLM adapter needs the optional "
                                  "dependency: uv add litellm") from None
            self._acompletion = litellm.acompletion
        return self._acompletion

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec],
        **params: Any,
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **{**self.defaults, **params},
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": s.to_dict()}
                               for s in tools]
        try:
            resp = await self._fn()(**kwargs)
        except Exception as ex:      # litellm normalizes provider errors
            raise LLMError(getattr(ex, "status_code", 500), str(ex)) from ex

        b = ReplyBuilder()
        async for chunk in resp:
            u = _get(chunk, "usage")
            if u:
                b.usage(input_tokens=_get(u, "prompt_tokens", 0) or 0,
                        output_tokens=_get(u, "completion_tokens", 0) or 0)
            choices = _get(chunk, "choices") or []
            if not choices:
                continue
            choice = choices[0]
            b.finish(_get(choice, "finish_reason"))
            d = _get(choice, "delta")
            if d is None:
                continue
            if _get(d, "reasoning_content"):
                yield b.reasoning(_get(d, "reasoning_content"))
            if _get(d, "content"):
                yield b.text(_get(d, "content"))
            for i, tc in enumerate(_get(d, "tool_calls") or []):
                idx = _get(tc, "index")     # absent index: position in the
                idx = i if idx is None else idx  # delta, so calls don't merge
                fn = _get(tc, "function")
                b.tool_call(idx, id=_get(tc, "id") or "",
                            name=(_get(fn, "name") or "") if fn else "")
                args = _get(fn, "arguments") if fn else None
                if args:
                    yield b.tool_args(idx, args)
        yield b.reply()
