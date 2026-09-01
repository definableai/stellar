"""Log: every event on the bus, as one line a person can read.

A listener, not a hook — it changes nothing, so it rides the data plane:

    agent.events.listen(Log())                        # everything, to print
    agent.events.listen(Log(lines.append), "tool")    # only the tool bells
    agent.events.listen(Log(), run_id=run.id)         # one conversation of many

Listeners are sync; a consumer that has to await takes events.stream().
"""

from core import Message, Part, ToolCall


def says(data) -> str:
    """One payload in a few words: what somebody watching would want to read."""
    if isinstance(data, Message):
        return " ".join(filter(None, [data.text, *map(says, data.tool_calls)]))
    if isinstance(data, ToolCall):
        return f"{data.name}({data.args})"
    if isinstance(data, Part):
        return data.data if data.type == "text" else f"<{data.type}>"
    if isinstance(data, list):
        return f"{len(data)} message" + "s" * (len(data) != 1)
    return "" if data is None else str(data)


def Log(write=print):
    """Hands every event it hears to one callable, `write(line)`."""

    def say(event) -> None:
        write(f"{event.name}: {says(event.data)}")

    return say
