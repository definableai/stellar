"""check_model: three exchanges that say whether an adapter speaks core.

One plain reply, then a tool round trip. Exactly three model calls, always
in that order, so a canned adapter can script its answers against them.
"""

import asyncio

from core.agent import Agent
from core.contracts import ContractError
from core.tool import tool

PLAIN = "Say only the word ok."
ROUND = "Call the echo tool with text='hi', then say done."


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


def wrong(number: int, expected: str, messages) -> ContractError:
    """Which exchange broke, what it needed, and what came back instead."""
    seen = ", ".join(m.role + ("+tool_calls" if m.tool_calls else "") for m in messages)
    return ContractError(f"exchange {number}: expected {expected}; saw [{seen}]")


async def exchanges(model) -> None:
    """Talk to the model three times and read what lands in the notebook."""
    said = (await Agent(model).run(PLAIN)).messages
    if not said or said[-1].role != "assistant" or not said[-1].content:
        raise wrong(1, "an assistant message with content", said)

    said = (await Agent(model, [echo]).run(ROUND)).messages
    asked = next((i for i, m in enumerate(said) if m.tool_calls), None)
    if asked is None:
        raise wrong(2, "an assistant message carrying tool_calls — does send "
                       "yield them as tool_call Parts, and does encode put "
                       "run.agent.tools in the body?",
                    said)
    call = said[asked].tool_calls[0]
    if not isinstance(call.args, dict):
        kind = type(call.args).__name__
        raise wrong(2, f"tool_calls args to be a dict, not {kind}", said)
    ran = next(
        (i for i, m in enumerate(said[asked + 1:], asked + 1)
         if m.role == "tool" and m.tool_call_id == call.id),
        None,
    )
    if ran is None:
        raise wrong(2, f"a tool message answering {call.id!r}", said)
    if not any(m.role == "assistant" for m in said[ran + 1:]):
        raise wrong(3, "one more assistant message after the tool result", said)


def check_model(model) -> None:
    """Grade an adapter; raise if it fails. Call this from sync code only.

    This grades the reply side (the Parts send yields) and the loop fit.
    Nothing here can see whether encode built a body your provider will
    accept — your adapter's own tests must assert that directly.
    """
    asyncio.run(exchanges(model))
