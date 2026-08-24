# external — your own adapters

Adapters you write live here. There are three kinds, named for what they
give the agent: `llm_*.py`, `tool_*.py`, `hook_*.py`.

Each file exposes one plain factory — call it, pass the result to
`Agent(...)`:

    from external.tool_skill import skill_tools

    agent = Agent(llm, tools=[*skill_tools("skills")])

`internal/` is the shipped catalog, same convention.
