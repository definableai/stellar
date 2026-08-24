"""The adapter recipe. An LLM adapter is exactly three transformations:

    1. request/messages:  core Message[] -> provider wire format
    2. request/blocks:    core content Blocks -> provider block shapes
    3. response:          provider stream events -> ReplyBuilder calls

Nothing else. Every adapter file has the same sections, in this order:

    # ---- request: core -> wire ----   block shapes + _to_wire(), built
    #                                   from one small per-role function,
    #                                   each named for the provider quirk
    #                                   it encodes
    # ---- adapter ----                 class: payload + stream loop

``core.ReplyBuilder`` owns transformation 3's assembly; the helpers
below dedupe the block dispatch in transformation 2.

A helper, not an adapter — nothing to compose onto an agent, so no
``setup(ctx)``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

BlockFn = Callable[[dict[str, Any]], dict[str, Any]]


def map_blocks(content: Any, *, text: BlockFn, image: BlockFn,
               file: BlockFn) -> Any:
    """Dispatch core content blocks to per-shape functions. Strings and
    None pass through (providers take bare strings). An unrecognized
    block type raises — a silently-emptied payload is worse than a
    loud request failure."""
    if not isinstance(content, list):
        return content or ""
    out = []
    for b in content:
        kind = b.get("type")
        if kind == "image":
            out.append(image(b))
        elif kind == "file":
            out.append(file(b))
        elif kind == "text" or "text" in b:
            out.append(text(b))
        else:
            raise ValueError(f"unknown content block type: {kind!r}")
    return out


def data_url(b: dict[str, Any], default_media: str) -> str:
    """The block's url, or its base64 payload as a data: URL."""
    return b.get("url") or (f"data:{b.get('media_type', default_media)};"
                            f"base64,{b.get('data', '')}")


def dump_result(content: Any) -> str:
    """Tool results cross the wire as text; non-JSON types degrade to str."""
    return json.dumps(content, default=str)
