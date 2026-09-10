"""Output: the final answer, parsed — or asked for again."""

from core import Message, hook


def Output(parse, tries: int = 2):
    """Parses the final answer; a bad one goes back with the error, tries times.

    What parse gives back lands in run.extra["output"] — no key, every try failed.
    """

    @hook("model.post")
    async def validate(answer, run) -> Message | None:
        """A reply that asks for tools is not the final answer: left alone."""
        # ponytail: the re-asks happen inside the card, so the other model.post
        # cards (Budget) never see the rejects and run.step does not count them.
        if answer.tool_calls:
            return None
        for left in range(tries, -1, -1):
            try:
                run.extra["output"] = parse(answer.text)
                return answer                # good — the loop appends this one
            except Exception as ex:
                if not left:
                    return answer            # out of tries: the text, unparsed
                run.messages += [answer, Message(
                    "user", f"Not valid: {ex}. Answer again with only the output.")]
                answer = await run.agent.model.invoke(run)

    return validate
