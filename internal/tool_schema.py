"""Schema inference: write a typed function, ``@tool`` does the rest.

Drop-in for ``core.tool`` — same decorator, but when ``parameters`` is
omitted it derives JSON Schema from the signature and Google-style
docstring (description = text before ``Args:``, per-param descriptions
from the ``Args:`` block):

    from internal.tool_schema import tool

    @tool()
    async def search(ctx, query: str, limit: int = 10):
        '''Search the index.

        Args:
            query: full-text query string
            limit: max results to return
        '''

Explicit ``parameters=`` always wins — use it for nested models or
anything inference can't express.

A helper, not an adapter — nothing to compose onto an agent, so no
``setup(ctx)``.
"""

from __future__ import annotations

import inspect
import re
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from core import Tool
from core import tool as core_tool

_PRIM = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _type_schema(t: Any) -> dict[str, Any]:
    if t in _PRIM:
        return {"type": _PRIM[t]}
    o = get_origin(t)
    if o is Annotated:                    # Annotated[T, "description"]
        inner, *meta = get_args(t)
        doc = next((m for m in meta if isinstance(m, str)), None)
        return {**_type_schema(inner),
                **({"description": doc} if doc else {})}
    if o is Literal:
        return {"enum": list(get_args(t))}
    if o in (list, set, tuple) or t is list:
        args = [a for a in get_args(t) if a is not Ellipsis]
        return {"type": "array", **({"items": _type_schema(args[0])} if args else {})}
    if o is dict or t is dict:
        return {"type": "object"}
    if o in (Union, types.UnionType):        # Optional[X] / X | None
        inner = [a for a in get_args(t) if a is not type(None)]
        if len(inner) == 1:
            return _type_schema(inner[0])
        return {"anyOf": [_type_schema(a) for a in inner]}
    return {}  # ponytail: unknown/nested types -> any; pass parameters= explicitly


def _parse_doc(fn: Any) -> tuple[str, dict[str, str]]:
    doc = inspect.getdoc(fn) or ""
    head, _, rest = doc.partition("Args:")
    rest = re.split(r"^\s*(?:Returns|Raises|Yields|Examples):", rest, maxsplit=1,
                    flags=re.M)[0]
    # ponytail: first line of each arg description only
    descs = dict(re.findall(r"^\s+(\w+)\s*(?:\([^)]*\))?:\s*(.+)$", rest, re.M))
    return head.strip(), descs


def infer_schema(fn: Any) -> tuple[str, dict[str, Any]]:
    """-> (description, JSON Schema) from signature + docstring."""
    desc, arg_docs = _parse_doc(fn)
    hints = get_type_hints(fn, include_extras=True)
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in list(inspect.signature(fn).parameters.values())[1:]:  # [0] is ctx
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        s = _type_schema(hints.get(p.name, Any))
        if p.name in arg_docs and "description" not in s:  # Annotated wins
            s = {**s, "description": arg_docs[p.name]}
        props[p.name] = s
        if p.default is p.empty:
            required.append(p.name)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return desc, schema


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    parallel_safe: bool = True,
    timeout: float | None = None,
):
    """Like ``core.tool`` but infers what you don't pass."""

    def wrap(fn: Any) -> Tool:
        inferred_desc, inferred_schema = infer_schema(fn)
        return core_tool(
            name=name or fn.__name__,
            description=description or inferred_desc,
            parameters=parameters or inferred_schema,
            parallel_safe=parallel_safe,
            timeout=timeout,
        )(fn)

    return wrap
