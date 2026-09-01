"""check_model: three exchanges that say whether an adapter speaks core.

One plain reply, then a tool round trip. Exactly three model calls, always
in that order, so a canned adapter can script its answers against them.
"""

import asyncio

from core.agent import Agent
from core.contracts import ContractError
from core.tool import tool
from core.types import Message, Part

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


async def talk(model, prompt: str, *tools) -> tuple[list[Message], list[Part]]:
    """One throwaway agent, one run: the notebook it filled, the deltas it rang."""
    agent = Agent(model, tools)
    deltas: list[Part] = []
    agent.events.listen(lambda event: deltas.append(event.data), "model.delta")
    return (await agent.run(prompt)).messages, deltas


def adds_up(number: int, said: list[Message], deltas: list[Part]) -> None:
    """What was streamed has to add up to what landed in the notebook.

    A Model that streams nothing rings no deltas and is excused; every
    ProviderModel rings one per Part, so its stream is graded in order.
    """
    streamed = "".join(p.data for p in deltas if p.type == "text")
    folded = "".join(m.text for m in said if m.role == "assistant")
    if streamed and streamed != folded:
        raise wrong(number, "the text deltas to add up, in order, to the reply — "
                    f"streamed {streamed!r}, folded {folded!r}", said)
    if any(p.type == "meta" for p in deltas) and not any(m.meta for m in said):
        raise wrong(number, "the meta parts merged into message.meta", said)


async def exchanges(model) -> None:
    """Talk to the model three times and read what lands in the notebook."""
    said, deltas = await talk(model, PLAIN)
    if not said or said[-1].role != "assistant" or not said[-1].text:
        raise wrong(1, "an assistant message with text", said)
    if sum(m.role == "assistant" for m in said) != 1:
        raise wrong(1, "one model call and no tool calls — there is no toolbox", said)
    adds_up(1, said, deltas)

    said, deltas = await talk(model, ROUND, echo)
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
    if sum(m.role == "assistant" for m in said) != 2:
        raise wrong(3, "the model to stop after the tool result — one ask, "
                       "one answer, one goodbye; does encode carry the tool "
                       "result back?", said)
    adds_up(2, said, deltas)


def check_model(model) -> None:
    """Grade an adapter; raise if it fails. Call this from sync code only.

    This grades the reply side — the Parts send yields, the deltas they ring
    on the way — and the loop fit. Nothing here can see whether encode built
    a body your provider will accept: your adapter's own tests must assert
    that directly.
    """
    asyncio.run(exchanges(model))
