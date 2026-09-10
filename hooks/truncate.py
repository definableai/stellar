"""Truncate: how much of one tool result the notebook has to carry."""

from core import Part, hook


def Truncate(max_chars: int = 20_000):
    """Cuts a tool result that would flood the context down to max_chars.

    Only text is cut, and it is cut once: all of it becomes one Part, and
    every other Part — an image, an adapter's own — rides along behind it
    untouched, in the order it came.
    """

    @hook("tool.post")
    async def cut(call, reply):
        """A reply that fits is left alone — None says so."""
        text = reply.text
        if len(text) <= max_chars:
            return None
        note = f"\n… [truncated, {len(text) - max_chars} more chars]"
        reply.content = [Part("text", text[:max_chars] + note)] + [
            p for p in reply.content if p.type != "text"]
        return reply

    return cut
