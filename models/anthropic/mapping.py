"""Claude's wire, row by row: the Messages API on the right, core on the left.

A Part is a block (OUT, IN — many per message, keyed by type). A Message is
a turn (turn, notebook — one per message). A Tool is a tool schema (tool).
The reply's header is the meta Part (meta — one per reply).
"""

from typing import cast

from core import Message, Part, ToolCall

# ---- Part <-> block -----------------------------------------------------


def text(p: Part) -> dict:
    return {"type": "text", "text": p.data}


def image(p: Part) -> dict:
    d = p.data                          # {"url": …} or {"media_type": …, "data": …}
    source = ({"type": "url", "url": d["url"]} if "url" in d else
              {"type": "base64", "media_type": d["media_type"], "data": d["data"]})
    return {"type": "image", "source": source}


def passthrough(p: Part) -> dict:
    """A block core never opened — a document, its own thinking — as it came."""
    return {"type": p.type, **p.data}


def whole(b: dict) -> Part:
    """A block core does not fold — thinking, whatever comes next — kept whole."""
    return Part(b["type"], b)


OUT = {"text": text, "image": image}        # Part.type -> block; else passthrough
IN = {                                      # block type -> Part; else whole
    "text": lambda b: Part("text", b["text"]),
    "tool_use": lambda b: Part(
        "tool_call", ToolCall(b["id"], b["name"], b.get("input") or {})),
}

# ---- Message -> turn, Tool -> schema ------------------------------------


def tool_use(c: ToolCall) -> dict:
    return {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}


def tool_result(m: Message) -> dict:
    """A tool Message as one tool_result block: its text, or every block it holds."""
    parts = cast(list[Part], m.content)
    content = (m.text if all(p.type == "text" for p in parts)
               else [OUT.get(p.type, passthrough)(p) for p in parts])
    return {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": content}


def tool(schema: dict) -> dict:
    return {"name": schema["name"], "description": schema["description"],
            "input_schema": schema["parameters"]}


def turn(m: Message) -> dict:
    """A user or assistant Message: its parts, then what it asked for, one list."""
    parts = cast(list[Part], m.content)
    return {"role": m.role,
            "content": [OUT.get(p.type, passthrough)(p) for p in parts]
                       + [tool_use(c) for c in m.tool_calls]}


def notebook(messages: list[Message]) -> tuple[str, list[dict]]:
    """The whole notebook: system lines up top, results in a row in one user turn."""
    system: list[str] = []
    turns: list[dict] = []
    merging = False
    for m in messages:
        if m.role == "system":
            system.append(m.text)
        elif m.role == "tool":
            if not merging:
                turns.append({"role": "user", "content": []})
            turns[-1]["content"].append(tool_result(m))
            merging = True
        else:
            turns.append(turn(m))
            merging = False
    return "\n\n".join(system), turns

# ---- reply header -> meta -----------------------------------------------


def meta(raw: dict, asked: str, invalid: dict) -> Part:
    """The one meta Part: what the turn cost, who answered, why it stopped."""
    used = raw.get("usage") or {}
    said = {"usage": {k: used.get(k) for k in ("input_tokens", "output_tokens")},
            "model": raw.get("model") or asked,
            "stop_reason": raw.get("stop_reason")}
    if invalid:
        said["invalid_args"] = invalid
    return Part("meta", said)
