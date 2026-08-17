from core.run import RunHandle
from core import Agent
from builtin.llm.openai import OpenAIResponsesLLM
from builtin.tools.schema import tool   # infers schema from signature + docstring
import asyncio

llm = OpenAIResponsesLLM(
    model="gpt-5.6-luna",
    api_key="***REMOVED***",
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

    return f"file written at : {path}"



async def main():

    a = Agent(llm=llm, tools=[write])

    # this will return a RunHandle
    handle: RunHandle = a.run(input="write a file name fast.py and write simple fastapi code for todo app")
    async for e in handle:
        print(e.payload)


asyncio.run(main())
