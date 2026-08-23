"""Smallest possible agent: one inferred-schema tool, streamed events.

    export OPENAI_API_KEY=...
    uv run python -m examples.cc.main
"""

import asyncio

from core import Agent
from internal.llm.openai import OpenAIResponsesLLM
from internal.tools.schema import tool   # infers schema from signature + docstring

llm = OpenAIResponsesLLM(
    model="gpt-5.6-luna",
    reasoning={"summary": "auto"},  # streams channel="reasoning" deltas
)


@tool(name="file_write")
def write(ctx, path: str, content: str):
    """Write or create a file on the system.

    Args:
        path: destination file path
        content: full file contents to write
    """
    with open(path, "w") as f:
        f.write(content)
    return f"file written at: {path}"


async def main():
    agent = Agent(llm=llm, tools=[write])
    handle = agent.run(input="write fast.py: a minimal fastapi todo app")
    async for e in handle:
        print(e.payload)


if __name__ == "__main__":
    asyncio.run(main())
