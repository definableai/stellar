"""The Responses API, row by row: the wire on the right, core on the left.

A turn here is a list of *items*, not one message — words, calls and the
model's own reasoning side by side — so a notebook line can open into several,
and this wire has an OUT and an IN. The reasoning item is why the folder
exists: kept whole coming in, it goes back out exactly as it came. A tool
call's arguments travel as a JSON *string*, dumped out and parsed back in.
"""

import json
from typing import cast

from core import ContractError, Message, Part, ToolCall
from core.llm import args_of

# ---- Part -> content part -----------------------------------------------


def text(p: Part) -> dict:
    return {"type": "input_text", "text": p.data}


def image(p: Part) -> dict:
    d = p.data                          # {"url": …} or {"media_type": …, "data": …}
    url = d["url"] if "url" in d else f"data:{d['media_type']};base64,{d['data']}"
    return {"type": "input_image", "image_url": url, "detail": "auto"}


def unsaid(p: Part) -> dict:
    """A part this wire has no shape for — loud, because a profile word promised it."""
    raise ContractError(
        f"the Responses API has no part type {p.type!r}; say 'text' or 'image'")


OUT = {"text": text, "image": image}        # Part.type -> content part; else unsaid

# ---- output item -> Parts -----------------------------------------------


def words(item: dict, invalid: dict) -> list[Part]:
    """One output message as its words: an output_text apiece."""
    return [Part("text", c["text"]) for c in item.get("content") or ()
            if c.get("type") == "output_text"]


def called(item: dict, invalid: dict) -> list[Part]:
    """One function call; arguments that will not parse are filed in invalid."""
    args, junk = args_of(item.get("arguments") or "")
    if junk is not None:
        invalid[item["call_id"]] = junk
    return [Part("tool_call", ToolCall(item["call_id"], item["name"], args))]


def whole(item: dict, invalid: dict) -> list[Part]:
    """Anything else as it came — a reasoning item above all, so it can go back."""
    return [Part(item["type"], item)]


IN = {"message": words, "function_call": called,   # item type -> Parts; else whole
      "reasoning": whole}


def parts(item: dict, invalid: dict) -> list[Part]:
    """One output item as Parts — its own row, or the item kept whole."""
    return IN.get(item["type"], whole)(item, invalid)

# ---- Message -> items, Tool -> schema -----------------------------------


def function(c: ToolCall) -> dict:
    return {"type": "function_call", "call_id": c.id, "name": c.name,
            "arguments": json.dumps(c.args)}


def tool(schema: dict) -> dict:
    return {"type": "function", **schema}       # flat here: no nested "function"


def turn(m: Message) -> list[dict]:
    """One line as items — reasoning, then words, then the calls it led to."""
    content = cast(list[Part], m.content)
    if m.role == "tool":                # its pictures ride a user item, below
        return [{"type": "function_call_output", "call_id": m.tool_call_id,
                 "output": m.text}]
    if m.role != "assistant":
        return [{"role": m.role,
                 "content": [OUT.get(p.type, unsaid)(p) for p in content]}]
    # ponytail: only reasoning replays whole — another wire's thinking block
    # cannot be said here, and a hosted-tool item is not ours to replay
    items = [p.data for p in content if p.type == "reasoning"]
    if m.text:                          # the plain shape: no item id to invent
        items.append({"role": "assistant", "content": m.text})
    return items + [function(c) for c in m.tool_calls]


def notebook(messages: list[Message]) -> tuple[str, list[dict]]:
    """The system lines lifted out, every other line as items.

    A tool item says text only, so the image Parts off a run of tool lines ride
    in one user item after that run, captioned with the call id. Anything else
    a tool returned has no shape here and is dropped.
    """
    system: list[str] = []
    items: list[dict] = []
    shown: list[dict] = []          # captions and images, held till the run ends
    for m in messages:
        if m.role == "system":
            system.append(m.text)
            continue
        if shown and m.role != "tool":
            items.append({"role": "user", "content": shown})
            shown = []
        items += turn(m)
        if m.role == "tool":
            images = [p for p in cast(list[Part], m.content) if p.type == "image"]
            if images:
                caption = f"[image returned by tool call {m.tool_call_id}]"
                shown += [text(Part("text", caption))] + [image(p) for p in images]
    if shown:
        items.append({"role": "user", "content": shown})
    return "\n\n".join(system), items

# ---- reply -> meta ------------------------------------------------------

USAGE = ("input_tokens", "output_tokens")   # this wire names them as core does


def meta(seen: dict, asked: str, invalid: dict) -> Part:
    """The one meta Part: what the turn cost, who answered, why it stopped.

    The status is the stop reason, except "incomplete" — then the reason
    underneath it is the word.
    """
    used = seen.get("usage") or {}
    stop = seen.get("status")
    if stop == "incomplete":
        stop = (seen.get("incomplete_details") or {}).get("reason") or stop
    said = {"usage": {k: used.get(k) for k in USAGE},
            "model": seen.get("model") or asked,
            "stop_reason": stop}
    if invalid:
        said["invalid_args"] = invalid
    return Part("meta", said)
