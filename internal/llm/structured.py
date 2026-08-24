"""extract() — schema-validated JSON from any LLM adapter, with retry.

    from internal.llm.structured import extract

    data = await extract(llm, schema={
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }, input="John Smith is 42.")

Whole-call retry loop: instructs JSON-only, parses (code fences tolerated),
validates against a schema subset (object type, required keys, primitive
property types — no jsonschema dependency), and on failure feeds the error
back to the model for another attempt. Provider-native enforcement
(``response_format`` etc.) composes: pass it via the adapter's defaults —
this validator is then the belt to that suspender.

ponytail: validates one level deep — nested object/array items unchecked;
bring jsonschema via ``validate=`` when that matters.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from core import LLMReply, Message

_TYPES = {"string": str, "integer": int, "number": (int, float),
          "boolean": bool, "array": list, "object": dict}


def _validate(schema: dict[str, Any], data: Any) -> str | None:
    """-> error string, or None if valid."""
    if not isinstance(data, dict):
        return f"expected a JSON object, got {type(data).__name__}"
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        return f"missing required keys: {missing}"
    for key, spec in (schema.get("properties") or {}).items():
        if key in data and spec.get("type") in _TYPES:
            expected = _TYPES[spec["type"]]
            ok = isinstance(data[key], expected)
            if spec.get("type") == "integer":   # bool is int in Python; and
                ok = ok and not isinstance(data[key], bool)
            if spec.get("type") == "number":
                ok = isinstance(data[key], (int, float)) and \
                    not isinstance(data[key], bool)
            if not ok:
                return f"key {key!r} should be {spec['type']}"
    return None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


async def extract(
    llm: Any,
    schema: dict[str, Any],
    input: "str | Sequence[Message]",
    attempts: int = 3,
    validate: Callable[[dict], str | None] | None = None,
    **params: Any,
) -> dict[str, Any]:
    msgs: list[Message] = [Message(
        role="system",
        content="Reply with ONLY a JSON object matching this schema — "
                "no prose, no markdown:\n" + json.dumps(schema))]
    if isinstance(input, str):
        msgs.append(Message(role="user", content=input))
    else:
        msgs.extend(input)

    err = "no attempts made"
    for _ in range(attempts):
        reply = None
        async for item in llm.stream(msgs, [], **params):
            if isinstance(item, LLMReply):
                reply = item
        text = (reply and reply.message.content) or ""
        try:
            data = json.loads(_strip_fences(text))
        except json.JSONDecodeError as ex:
            err = f"not valid JSON ({ex})"
        else:
            err = _validate(schema, data)
            if err is None and validate:
                err = validate(data) or None   # any falsy return counts as valid
            if err is None:
                return data
        msgs.append(reply.message if reply else
                    Message(role="assistant", content=text))
        msgs.append(Message(role="user",
                            content=f"Invalid: {err}. Reply with ONLY the "
                                    "corrected JSON object."))
    raise ValueError(f"no schema-valid output after {attempts} attempts: {err}")
