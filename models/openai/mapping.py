"""Chat Completions, row by row: the wire on the right, core on the left.

Its reply has no block types — text arrives as a string, tool calls as a
list — so this wire has OUT and no IN: one in-row, tool_call. A tool call's
arguments travel as a JSON *string*: dumped on the way out, parsed on the
way back in.
"""

import json
from typing import cast

from core import ContractError, Message, Part, ToolCall
from core.llm import args_of

# ---- Part -> content part -----------------------------------------------


def text(p: Part) -> dict:
    return {"type": "text", "text": p.data}


def image(p: Part) -> dict:
    d = p.data                          # {"url": …} or {"media_type": …, "data": …}
    url = d["url"] if "url" in d else f"data:{d['media_type']};base64,{d['data']}"
    return {"type": "image_url", "image_url": {"url": url}}


def unsaid(p: Part) -> dict:
    """A part this wire has no shape for — loud, because a profile word promised it."""
    raise ContractError(
        f"Chat Completions has no part type {p.type!r}; say 'text' or 'image'")


OUT = {"text": text, "image": image}        # Part.type -> content part; else unsaid

# ---- Message -> message, Tool -> schema ---------------------------------


def function(c: ToolCall) -> dict:
    return {"id": c.id, "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.args)}}


def tool(schema: dict) -> dict:
    return {"type": "function", "function": schema}


def turn(m: Message) -> dict:
    """One notebook line as one message: the role, the words, the plumbing."""
    parts = cast(list[Part], m.content)
    if m.role == "assistant":
        # its own past may be off another wire — a thinking block from a Claude
        # turn — so drop what this one cannot say; a part you authored goes
        # through unsaid, because a profile word promised it
        parts = [p for p in parts if p.type in OUT]
    said = {"role": m.role,
            "content": (m.text if all(p.type == "text" for p in parts)
                        else [OUT.get(p.type, unsaid)(p) for p in parts])}
    if m.tool_calls:
        said["tool_calls"] = [function(c) for c in m.tool_calls]
    if m.tool_call_id:
        said["tool_call_id"] = m.tool_call_id
    return said

# ---- reply -> Parts -----------------------------------------------------

USAGE = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}


def tool_call(call: dict, invalid: dict) -> Part:
    """One tool call, POST-shaped: {"id", "function": {"name", "arguments"}}.

    Arguments that will not parse are filed in invalid under the call's id,
    and the ToolCall goes out with none — the loop answers the model itself.
    """
    args, junk = args_of(call["function"].get("arguments") or "")
    if junk is not None:
        invalid[call["id"]] = junk
    return Part("tool_call", ToolCall(call["id"], call["function"]["name"], args))


def meta(seen: dict, asked: str, invalid: dict) -> Part:
    """The one meta Part: what the turn cost, who answered, why it stopped."""
    used = seen.get("usage") or {}
    said = {"usage": {core: used.get(wire) for core, wire in USAGE.items()},
            "model": seen.get("model") or asked,
            "stop_reason": seen.get("finish_reason")}
    if invalid:
        said["invalid_args"] = invalid
    return Part("meta", said)
