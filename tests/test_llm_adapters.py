"""LLM adapter self-checks against fake httpx transports (no network):
Anthropic Messages, OpenAI chat completions + responses, and the shared
block/result helpers.

Run: uv run python tests/test_llm_adapters.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    LLMError, LLMReply, Message, ToolCall, ToolResult, ToolSpec, file_block,
)
from internal.llm.anthropic import AnthropicLLM, _to_anthropic  # noqa: E402
from internal.llm.common import data_url, dump_result, map_blocks  # noqa: E402
from internal.llm.openai import (  # noqa: E402
    OpenAILLM, OpenAIResponsesLLM, _blocks, _r_blocks,
)


def test_common_helpers() -> None:
    same = {"text": lambda b: {"t": b.get("text", "")},
            "image": lambda b: {"i": 1}, "file": lambda b: {"f": 1}}
    assert map_blocks("plain", **same) == "plain"
    assert map_blocks(None, **same) == ""
    assert map_blocks([{"type": "text", "text": "x"}, {"type": "image"},
                       {"type": "file"}, {"text": "no-type"}], **same) == \
        [{"t": "x"}, {"i": 1}, {"f": 1}, {"t": "no-type"}]
    try:
        map_blocks([{"type": "video", "url": "u"}], **same)
        raise AssertionError("unknown block type must raise")
    except ValueError:
        pass
    assert data_url({"url": "http://x"}, "a/b") == "http://x"
    assert data_url({"data": "QUJD"}, "a/b") == "data:a/b;base64,QUJD"
    assert dump_result({"k": object()}).startswith('{"k": ')


async def test_anthropic() -> None:
    system, msgs = _to_anthropic([
        Message(role="system", content="be terse"),
        Message(role="user", content="add 1+2 and 3+4"),
    ])
    assert system == "be terse" and msgs == [{"role": "user", "content": "add 1+2 and 3+4"}]

    _, msgs2 = _to_anthropic([
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=[
            ToolCall(id="t1", name="add", arguments={"a": 1}),
            ToolCall(id="t2", name="add", arguments={"a": 3})]),
        Message(role="tool", tool_result=ToolResult("t1", "add", 3)),
        Message(role="tool", tool_result=ToolResult("t2", "add", 7, is_error=True)),
    ])
    assert [b["type"] for b in msgs2[1]["content"]] == ["tool_use", "tool_use"]
    results = msgs2[2]["content"]                 # coalesced into ONE user turn
    assert msgs2[2]["role"] == "user" and len(results) == 2
    assert results[1]["is_error"] and "tool_use_id" in results[0]

    sse = b"".join([
        b'event: message_start\n',
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":9}}}\n\n',
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"s1"}}\n\n',
        b'data: {"type":"content_block_stop","index":0}\n\n',
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}\n\n',
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
        b'data: {"type":"content_block_start","index":2,"content_block":'
        b'{"type":"tool_use","id":"t9","name":"add"}}\n\n',
        b'data: {"type":"content_block_delta","index":2,"delta":'
        b'{"type":"input_json_delta","partial_json":"{\\"a\\": "}}\n\n',
        b'data: {"type":"content_block_delta","index":2,"delta":'
        b'{"type":"input_json_delta","partial_json":"5}"}}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        b'"usage":{"output_tokens":11}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ])
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, content=sse, headers={"content-type": "text/event-stream"}))
    llm = AnthropicLLM(api_key="test", client=httpx.AsyncClient(
        transport=transport, base_url="http://fake"))
    got = [x async for x in llm.stream([Message(role="user", content="hi")],
                                       [ToolSpec(name="add")])]
    *deltas, reply = got
    assert [(d.text, d.channel, d.index) for d in deltas] == [
        ("hmm", "reasoning", 0), ("Hi", "text", 0),
        ('{"a": ', "tool_args", 2), ("5}", "tool_args", 2)], deltas
    assert reply.message.content == "Hi"          # reasoning stays out
    assert reply.message.tool_calls == [ToolCall(id="t9", name="add",
                                                 arguments={"a": 5})]
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (9, 11)
    assert reply.stop_reason == "tool_use"
    assert reply.message.meta["anthropic_thinking"] == [
        {"type": "thinking", "thinking": "hmm", "signature": "s1"}]
    _, echo = _to_anthropic([Message(role="user", content="hi"), reply.message])
    assert echo[1]["content"][0]["type"] == "thinking"   # echoed back first
    _, dropped = _to_anthropic([Message(role="user", content="x"),
                                Message(role="assistant")])
    assert len(dropped) == 1                             # empty assistant dropped

    captured: dict[str, Any] = {}

    def capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, content=sse,
                              headers={"content-type": "text/event-stream"})

    cllm = AnthropicLLM(api_key="t", client=httpx.AsyncClient(
        transport=httpx.MockTransport(capture), base_url="http://fake"))
    async for _ in cllm.stream([Message(role="system", content="sys"),
                                Message(role="user", content="hi")], []):
        pass
    body = captured["body"]
    assert body["system"] == [{"type": "text", "text": "sys",
                               "cache_control": {"type": "ephemeral"}}]
    assert body["messages"][-1]["content"][-1]["cache_control"] == \
        {"type": "ephemeral"}

    nllm = AnthropicLLM(api_key="t", cache=False, client=httpx.AsyncClient(
        transport=httpx.MockTransport(capture), base_url="http://fake"))
    async for _ in nllm.stream([Message(role="system", content="sys"),
                                Message(role="user", content="hi")], []):
        pass
    assert captured["body"]["system"] == "sys"                # untouched
    assert captured["body"]["messages"][-1]["content"] == "hi"

    err_t = httpx.MockTransport(lambda req: httpx.Response(
        401, content=b'{"error":{"type":"authentication_error"}}'))
    bad = AnthropicLLM(api_key="x", client=httpx.AsyncClient(
        transport=err_t, base_url="http://fake"))
    try:
        async for _ in bad.stream([Message(role="user", content="hi")], []):
            pass
        raise AssertionError("expected LLMError on 401")
    except LLMError as e:
        assert e.status == 401 and not e.retryable

    overload = b'data: {"type":"error","error":{"type":"overloaded_error"}}\n\n'
    ot = httpx.MockTransport(lambda req: httpx.Response(
        200, content=overload, headers={"content-type": "text/event-stream"}))
    ollm = AnthropicLLM(api_key="x", client=httpx.AsyncClient(
        transport=ot, base_url="http://fake"))
    try:
        async for _ in ollm.stream([Message(role="user", content="hi")], []):
            pass
        raise AssertionError("expected LLMError on overload")
    except LLMError as e:
        assert e.status == 529 and e.retryable


async def test_openai() -> None:
    try:
        _blocks([file_block(url="http://x/d.pdf")])
        raise AssertionError("chat file URL must raise")
    except ValueError:
        pass
    assert _blocks([file_block(data="QUJD")])[0]["file"][
        "file_data"].startswith("data:application/pdf")
    assert _r_blocks([file_block(url="http://x/d.pdf")]) == [
        {"type": "input_file", "file_url": "http://x/d.pdf"}]

    sse_body = b"".join([
        b'data: {"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n\n',
        b': keep-alive comment\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        b'"function":{"name":"add","arguments":"{\\"a\\": "}}]},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,'
        b'"completion_tokens_details":{"reasoning_tokens":2}}}\n\n',
        b'data: [DONE]\n\n',
    ])
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200, content=sse_body,
            headers={"content-type": "text/event-stream"}))
    llm = OpenAILLM(api_key="test", client=httpx.AsyncClient(
        transport=transport, base_url="http://fake"))
    got = [x async for x in llm.stream(
        [Message(role="user", content="hi")],
        [ToolSpec(name="add", parameters={"type": "object"})])]

    *deltas, reply = got
    assert [(d.text, d.channel) for d in deltas] == [
        ("hmm", "reasoning"), ("Hel", "text"), ("lo", "text"),
        ('{"a": ', "tool_args"), ("2}", "tool_args")], deltas
    assert isinstance(reply, LLMReply)
    assert reply.message.content == "Hello"       # reasoning stays out
    assert reply.message.tool_calls == [
        ToolCall(id="c1", name="add", arguments={"a": 2})]
    assert (reply.usage.input_tokens, reply.usage.output_tokens,
            reply.usage.reasoning_tokens) == (7, 3, 2)
    assert reply.stop_reason == "tool_calls"

    err_transport = httpx.MockTransport(
        lambda req: httpx.Response(401, content=b'{"error":"bad key"}'))
    bad = OpenAILLM(api_key="x", client=httpx.AsyncClient(
        transport=err_transport, base_url="http://fake"))
    try:
        async for _ in bad.stream([Message(role="user", content="hi")], []):
            pass
        raise AssertionError("expected LLMError on 401")
    except LLMError as e:
        assert e.status == 401 and "bad key" in e.body and not e.retryable

    resp_body = b"".join([
        b'data: {"type":"response.reasoning_summary_text.delta","delta":"think"}\n\n',
        b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
        b'data: {"type":"response.function_call_arguments.delta",'
        b'"delta":"{\\"a\\": 1}","output_index":1}\n\n',
        b'data: {"type":"response.output_item.done","item":{"type":"function_call",'
        b'"call_id":"c9","name":"add","arguments":"{\\"a\\": 1}"}}\n\n',
        b'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,'
        b'"output_tokens":9,"output_tokens_details":{"reasoning_tokens":4}}}}\n\n',
    ])
    rt = httpx.MockTransport(lambda req: httpx.Response(
        200, content=resp_body,
        headers={"content-type": "text/event-stream"}))
    rllm = OpenAIResponsesLLM(api_key="test", client=httpx.AsyncClient(
        transport=rt, base_url="http://fake"))
    got2 = [x async for x in rllm.stream([Message(role="user", content="hi")], [])]
    *d2, rep2 = got2
    assert [(d.text, d.channel, d.index) for d in d2] == [
        ("think", "reasoning", 0), ("Hi", "text", 0),
        ('{"a": 1}', "tool_args", 1)], d2
    assert rep2.message.content == "Hi"
    assert rep2.message.tool_calls == [ToolCall(id="c9", name="add",
                                                arguments={"a": 1})]
    assert (rep2.usage.input_tokens, rep2.usage.output_tokens,
            rep2.usage.reasoning_tokens) == (5, 9, 4)
    assert rep2.stop_reason == "tool_use"


async def main() -> None:
    test_common_helpers()
    await test_anthropic()
    await test_openai()
    print("test_llm_adapters: all ok")


if __name__ == "__main__":
    asyncio.run(main())
