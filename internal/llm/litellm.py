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

Self-check (no network, no litellm install needed — acompletion= seam):

    uv run python -m internal.llm.litellm
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


# ---- self-check -------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from types import SimpleNamespace as NS

    def _delta(**kw: Any) -> Any:
        d = NS(content=None, reasoning_content=None, tool_calls=None)
        d.__dict__.update(kw)
        return d

    async def _selfcheck() -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)

            async def gen():
                yield NS(choices=[NS(delta=_delta(reasoning_content="hmm"),
                                     finish_reason=None)], usage=None)
                yield NS(choices=[NS(delta=_delta(content="Hi"),
                                     finish_reason=None)], usage=None)
                yield NS(choices=[NS(delta=_delta(tool_calls=[
                    NS(index=0, id="c1",
                       function=NS(name="add", arguments='{"a": '))]),
                    finish_reason=None)], usage=None)
                yield NS(choices=[NS(delta=_delta(tool_calls=[
                    NS(index=0, id=None,
                       function=NS(name=None, arguments="2}"))]),
                    finish_reason="tool_calls")], usage=None)
                yield NS(choices=[],
                         usage=NS(prompt_tokens=7, completion_tokens=3))
            return gen()

        llm = LiteLLM(model="fake/model", acompletion=fake_acompletion)
        got = [x async for x in llm.stream(
            [Message(role="user", content="hi")], [ToolSpec(name="add")])]
        *deltas, reply = got
        assert [(d.text, d.channel) for d in deltas] == [
            ("hmm", "reasoning"), ("Hi", "text"),
            ('{"a": ', "tool_args"), ("2}", "tool_args")], deltas
        assert reply.message.content == "Hi"
        assert reply.message.tool_calls[0].name == "add"
        assert reply.message.tool_calls[0].arguments == {"a": 2}
        assert (reply.usage.input_tokens, reply.usage.output_tokens) == (7, 3)
        assert reply.stop_reason == "tool_calls"
        assert captured["model"] == "fake/model"
        assert captured["messages"] == [{"role": "user", "content": "hi"}]
        assert captured["tools"][0]["function"]["name"] == "add"
        assert captured["stream_options"] == {"include_usage": True}

        # dict-shaped chunks (older litellm / shims) parse identically;
        # two index-less calls in one delta stay two calls
        async def dict_acompletion(**kwargs: Any) -> Any:
            async def gen():
                yield {"choices": [{"delta": {"content": "Yo"},
                                    "finish_reason": None}]}
                yield {"choices": [{"delta": {"tool_calls": [
                    {"id": "d1", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "d2", "function": {"name": "b", "arguments": "{}"}},
                ]}, "finish_reason": "tool_calls"}]}
                yield {"choices": [], "usage": {"prompt_tokens": 1,
                                                "completion_tokens": 2}}
            return gen()

        dllm = LiteLLM(model="d", acompletion=dict_acompletion)
        *_, drep = [x async for x in dllm.stream(
            [Message(role="user", content="q")], [])]
        assert drep.message.content == "Yo"
        assert [(c.id, c.name) for c in drep.message.tool_calls] == \
            [("d1", "a"), ("d2", "b")]
        assert (drep.usage.input_tokens, drep.usage.output_tokens) == (1, 2)
        assert drep.stop_reason == "tool_calls"

        # provider error -> LLMError carrying the mapped status
        async def boom(**kwargs: Any) -> Any:
            e = RuntimeError("rate limited")
            e.status_code = 429
            raise e

        bad = LiteLLM(model="x", acompletion=boom)
        try:
            async for _ in bad.stream([Message(role="user", content="hi")], []):
                pass
            raise AssertionError("expected LLMError")
        except LLMError as e:
            assert e.status == 429 and e.retryable

        print("litellm adapter self-check ok")

    asyncio.run(_selfcheck())
