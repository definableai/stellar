"""Compact: the older notebook folded to one line once it outgrows the window."""

from dataclasses import replace

from core import Hooks, Message, Run, hook

PROMPT = "Summarise densely — every fact, decision, file path and open task:\n\n"


def Compact(max_tokens: int, keep: int = 4):
    """All but the last `keep` lines, traded for one summary line so the run goes on."""

    @hook("model.pre")
    async def fold(messages, run):
        """A list back replaces the notebook for good; None leaves it alone."""
        usage = next((m.meta.get("usage") or {} for m in reversed(messages)
                      if m.role == "assistant"), {})
        rest = [m for m in messages if m.role != "system"]   # system stays up top
        cut = max(len(rest) - keep, 0)
        while cut and rest[cut].role == "tool":    # a tool batch never leaves the
            cut -= 1                               # turn that asked for it
        if not cut or (usage.get("input_tokens") or 0) <= max_tokens:
            return None
        told = "\n".join(f"{m.role}: {m.text} " + " ".join(
            f"{c.name}({c.args})" for c in m.tool_calls) for m in rest[:cut])
        aside = Run(replace(run.agent, tools={}),   # no toolbox: a summary
                    run.id + ".compact",            # is words; its own id
                    [Message("user", PROMPT + told)], hooks=Hooks(Hooks()))
        summary = await run.agent.model.invoke(aside)   # empty parent: no cards ring
        note = Message("user", "Summary of the conversation so far:\n" + summary.text)
        return [m for m in messages if m.role == "system"] + [note] + rest[cut:]

    return fold
