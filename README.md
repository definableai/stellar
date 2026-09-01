# stellar core

A minified agent core: the infrastructure to run an agent, kept under 2000
readable lines (`wc -l core/*.py`) and built on the standard library alone.
Small enough to read top to bottom in one sitting, strong enough for hard tasks
when it is used well. Everything past the loop — providers, rule cards,
transports — is a small file outside `core/` that you can read, copy and change.

## The robot with one backpack

An agent is a robot with one backpack. Inside: a brain (`model`), a toolbox
(`tools`), a stack of rule cards (`hooks`), a notebook (`messages`) and one
spare pocket (`extra`).

The robot reads its rule cards before and after it thinks, and before and after
every tool it picks up. Everything that happens goes in the notebook. When it
stops asking for tools, it is done.

```
run_pre
  ┌ model_pre → the model answers, a model_delta per streamed Part → model_post
  │   the answer goes in the notebook; for each tool it asked for:
  │     tool_pre → the tool runs, a tool_delta per yielded Part → tool_post
  │     → the result goes in the notebook
  └ again, as long as the last answer asked for tools
run_post
```

**One `Agent` is one conversation — never share a live robot.** The brain, the
toolbox and the rule cards can be shared with other robots; the notebook, the
spare pocket and everything in flight are its own. Two runs on one `Agent` at
the same time will interleave their notebooks and trample `response`, `call`,
`result` and `delta`. Build a robot per conversation; it is one line.

Any part can be swapped by assignment, even mid-run: `check()` reads the whole
backpack again at the top of every step, so a bad swap is caught before it runs.

## The nine nouns

| noun | kind | the contract |
| --- | --- | --- |
| `Part` | data | one piece of a message: `type` names it, `data` holds it. Three types are reserved words core folds itself — `text`, `tool_call`, `meta`; everything else passes through untouched. |
| `ToolCall` | data | the model asking for one tool: `id`, `name`, `args`. |
| `Message` | data | one line in the notebook: `role`, `content`, `tool_calls`, `tool_call_id`, `meta`. `content` is always `list[Part]` — a str coerces at construction, and `.text` joins the text parts back. |
| `Model` | contract | the brain. Write an async `invoke`; hand back a `Message`. |
| `Tool` | contract | one thing the agent can do. Write an async `execute` — an async generator streams. |
| `Hook` | contract | a rule card. Write the moments you care about; the rest do nothing. |
| `Agent` | the bag | all of the above plus `messages`, `step` and `extra`. One method: `run()`. |
| `ContractError` | signal | you wired it wrong. The message carries the code to type instead. |
| `Stop` | signal | end the run cleanly. Raise it from any hook or tool. |

Everything is async and every method takes one argument, the agent. There are
no sync twins and no bridges: sync code is one `asyncio.to_thread` line the
implementer writes inside the async method.

`from core import …` also gives you the things that are not new nouns:
`ProviderModel` and `FakeModel` (both are Models), the `@tool` and `on()`
shortcuts, `check_model`, and the loop's own functions `run`, `fire` and
`check`.

## Quickstart

Build the wiring once. Build an `Agent` per conversation.

```python
import asyncio

from core import Agent, tool
from hooks.steps import Steps
from models.anthropic import Anthropic


@tool
def shout(word: str) -> str:
    """Shout a word."""
    return word.upper()


wiring = {"model": Anthropic(), "tools": [shout], "hooks": [Steps(8)]}


async def main() -> None:
    agent = Agent(**wiring)
    answer = await agent.run("shout hello")
    print(answer.text)                             # what the model said last

    await agent.run("again, louder")               # same robot, same notebook


asyncio.run(main())
```

`Anthropic()` reads `ANTHROPIC_API_KEY` from the environment unless you pass
`api_key=`. `run()` takes a str (wrapped into a user message), a `Message`
(appended as-is — images and all), or nothing (runs on the messages already
there — that is the resume-from-checkpoint path). It gives back the last
message, so `answer = await Agent(model).run("hi")` works in one line when
you have nothing to keep.

## Models

### Three ways in

**Wrap what you already have.** Subclass `Model` and write an async `invoke`.
A sync SDK is one `to_thread` line away:

```python
class MyModel(Model):
    async def invoke(self, agent) -> Message:
        said = await asyncio.to_thread(my_sdk.complete, agent.messages)
        return Message("assistant", said)
```

**No base class.** Any object whose own class defines an async `invoke(agent)`
passes `check()` and runs — nothing checks your ancestry. A sync `invoke` is
refused by `check()` up front, because the loop could never await it.

**`ProviderModel`.** An HTTP provider in two steps:

```python
class MyProvider(ProviderModel):
    def encode(self, agent) -> dict:        # pure: notebook + toolbox -> request body
        ...

    async def send(self, agent, body):      # the network call; retries live here
        ...
        yield Part("text", "…")             # yield the reply as Parts
```

`encode` is pure translation, which is what makes it easy to test. Everything
that can go wrong on a network lives in `send`. A provider that does not
stream just yields the Parts of its one response — the shipped adapters do
exactly that.

`send` speaks the Part protocol, and core folds the stream into one assistant
`Message` (firing `model_delta` per Part along the way):

- `Part("text", str)` is a delta — consecutive text parts concatenate into one.
- `Part("tool_call", ToolCall(…))` arrives whole — buffer partial JSON in the
  adapter, core never sees half an argument string.
- `Part("meta", dict)` merges into `Message.meta`, later keys winning.
- any other type lands in `content` untouched — thinking blocks, provider
  extras, whatever comes next. New provider features are new Part types,
  never new methods.

`encode` is where a `Part` gets opened. So adapters agree, one more spelling
is shared: `Part("image", {"url": …})` or
`Part("image", {"media_type": …, "data": <base64 str>})`.

Put what the reply cost in a meta Part, as
`{"usage": {"input_tokens": …, "output_tokens": …}}`. That is the convention
`Budget` and `Log` read; an adapter that leaves it out just counts as free.

### Test with a canned POST, then grade it

Replace the network call with a list of real response bodies — the reply
still travels the adapter's own parsing and the real fold path. No network,
no key:

```python
class Canned(Anthropic):
    def __init__(self, bodies) -> None:
        super().__init__(api_key="test")
        self.bodies, self.sent = list(bodies), []

    async def post(self, body) -> dict:
        self.sent.append(body)
        return self.bodies[len(self.sent) - 1]


check_model(Canned(BODIES))
```

`check_model` makes exactly three model calls, always in this order: a plain
reply, a reply asking for the `echo` tool, and a final reply after the tool
result — so a three-entry script lines up with it.

It grades the reply side: do the Parts `send` yields fold into a `Message`,
do tool calls arrive as whole `ToolCall`s with a dict of args, does the round
trip close. It cannot see whether `encode` built a body your provider would
accept — assert that yourself against `model.encode(agent)`.
`tests/test_anthropic.py` and `tests/test_openai.py` do both halves.
`check_model` calls `asyncio.run` inside, so call it from sync code only.

`FakeModel` is a `ProviderModel` whose `send` yields the Parts of a scripted
reply — tests ride the same fold and delta path with no network at all.

### The two shipped ones

`models/anthropic.py` — the Messages API, default `claude-sonnet-5`.
`models/openai.py` — Chat Completions (the format the compatible vendors
clone), default `gpt-5.6-luna`; point `base_url` at a clone and the same class
talks to it. Both take `api_key=` or read the environment, and pass any extra
keyword straight through to the request body. Neither streams off the socket
yet; both yield their one response as Parts, which is all the protocol asks.

## Tools

### A function

```python
@tool
def shout(word: str, times: int = 1) -> str:
    """Shout a word."""
    return (word.upper() + " ") * times
```

The name is the function's name, the description is its docstring, and the
schema is the signature: hints become JSON types, and a parameter with no
default is required. `@tool` takes no arguments at all — for a different name,
description or schema, write a `Tool` subclass; that is what the class is for.
It refuses `*args`/`**kwargs` and says so.

A sync function runs in a thread, an async one is awaited, and an async
generator streams — every `Part` it yields rings `tool_delta`.

A first parameter spelled exactly `agent` is filled in by the loop and kept out
of the schema, so the model never sees it:

```python
@tool
def notebook(agent) -> int:
    """How many lines the notebook holds."""
    return len(agent.messages)
```

### A class

```python
class Shout(Tool):
    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    async def execute(self, agent, word) -> str:
        return word.upper()
```

`execute` is async; wrap sync work in one `asyncio.to_thread` line. A class is
the answer whenever the tool holds something. Write `execute` as an async
generator and the tool streams: each yielded `Part` rings `tool_delta` and
folds into the tool message, same protocol as a streaming model.

### What comes back

Return a `str` and the model sees it. Return anything else and it arrives as
JSON — whatever JSON cannot hold becomes its `str`. Return a `Message` and it is
used as it is, with its role forced to `tool` and the call's id filled in. An
async generator's result is the fold of everything it yielded.

Raise, and the model sees `error: ValueError: nope` and gets another turn. Ask
for a tool that is not in the toolbox and it sees
`error: unknown tool: ghost`. Run-time mistakes are answers, not crashes. The
one exception is `Stop`, which always ends the run.

Tools live in a list, like hooks — `Agent(model, [shout])`, and
`agent.tools.append(shout)` mid-run. The name lives in one place, `tool.name`;
`check()` refuses two tools that answer to the same one, on every step.

## Hooks

A hook is a class with any of eight methods. The eight names are the eight
moments, and nothing else is a moment.

| method | fires | what it may touch |
| --- | --- | --- |
| `run_pre` | once, before the first turn | `agent.messages` — the place to put a system message |
| `model_pre` | before every model call | `agent.messages`; `agent.step` is already up |
| `model_delta` | per Part a model streams | `agent.delta` — the Part, already folded into `agent.response` |
| `model_post` | after every model answer | `agent.response` — replace it and your version is what gets written down |
| `tool_pre` | before every tool | `agent.call`; fill `agent.result` and the tool never runs |
| `tool_delta` | per Part a tool yields | `agent.delta` — the Part, already folded into `agent.result` |
| `tool_post` | after every tool | `agent.result`, always a tool `Message` by now |
| `run_post` | once, at the very end | anything; this is teardown |

Each may be sync or async. Hooks fire in list order, and a hook may append
another hook mid-run — the new one starts listening at the next moment. A
`Hook` subclass that overrides none of the eight is a typo, and `check()`
raises instead of letting it sit there doing nothing. For one moment and one
line, `on("run_post", fn)` builds the hook for you.

The deltas are how a streaming UI is written once and works with every
adapter: read `agent.delta` for the piece, `agent.response.text` (or
`agent.result.text`) for everything so far. A model that answers in one Part
still rings the bell — the granularity is coarser, the semantics identical.
Replace `agent.response` at `model_post`, not during deltas.

### Stopping

`raise Stop` from any hook or any tool ends the run cleanly: no traceback
reaches the caller, and the agent comes back holding everything written down so
far.

`run_post` always fires — after a clean finish, after a `Stop`, and after a
crash. It is a `finally`, so a `Stop` raised inside `run_post` is ignored (too
late to stop anything), and any other exception from `run_post` propagates the
ordinary way, with whatever was already in flight attached as `__context__`.

Everything else propagates: a model that raises, a hook that raises, a
`ContractError`. The loop catches exactly two things — a `Stop`, and a tool's
own exception, which becomes a result the model reads.

### Denying a tool

Fill `agent.result` in `tool_pre`. The loop finds a result already there, skips
the tool, and writes your answer down as the tool's result for the model to
read. That is the whole deny path — no third signal, no boolean to return.

### The four shipped hooks

| hook | what it does |
| --- | --- |
| `Steps(n)` | n model calls, then `Stop`. `Steps(3)` buys exactly three replies. |
| `Budget(max_tokens)` | adds `input_tokens + output_tokens` from every reply's `meta["usage"]` into `extra["budget.tokens"]`. Past the limit it keeps the reply you already paid for, then stops. A reply with no usage costs nothing. |
| `Permission(deny, allow, ask)` | the guard at the tool gate. `deny` wins; then an `allow` list settles it alone; only then does `ask=True` put the question to a human. Each of `deny`/`allow` is a set of names or a predicate `(agent, call) -> bool`. |
| `Log(emit)` | hands all eight moments to one callable, `emit(name, payload)`, with JSON-safe payloads. `emit` may be sync or async. |

`Permission(ask=True)` asks through `agent.extra["ui.ask"]`, an
`async (question, options=None) -> str` that you provide. Anything but
`"allow"` denies, and no channel at all denies too — a guard with no voice says
no.

## Patterns

### A sub-agent is a tool

Not a feature. A tool that builds a child agent on the parent's brain, runs it,
and hands back what it said:

```python
class Research(Tool):
    name = "research"
    description = "Ask a helper agent one question."
    parameters = {"type": "object", "properties": {"question": {"type": "string"}}}

    async def execute(self, agent, question) -> str:
        answer = await Agent(agent.model).run(question)
        return answer.text
```

The child gets its own notebook and its own spare pocket, so nothing comes back
except the sentence you return. Hand it its own tools and hooks when it should
be allowed less than its parent.

### Checkpoint and resume

A checkpoint is `messages`, `step` and `extra`. Nothing else — the wiring is
code, and you build it again:

```python
agent = Agent(**wiring, messages=saved, step=step, extra=extra)
```

Keep the notebook as objects and there is nothing to do. To put it on disk,
`dataclasses.asdict` each `Message`. Reading it back, rebuild the nouns inside
it yourself — `content` is always a list of parts, so there is exactly one
shape to revive:

```python
def revive(d: dict) -> Message:
    return Message(**{
        **d,
        "content": [Part(**p) for p in d["content"]],
        "tool_calls": [ToolCall(**c) for c in d["tool_calls"]],
    })
```

Never put a callable in `extra` that has to survive a checkpoint —
`extra["ui.ask"]` is a live channel, not saved state.

### Events over the wire

`Log` turns the eight moments into calls on one `emit`. `EventStream.emit` has
exactly that shape, so wiring them together is one line, and the transports turn
the stream into frames:

```python
stream = EventStream()
agent = Agent(model, tools, [*hooks, Log(stream.emit)])
asyncio.create_task(agent.run(ask))         # hold the task; that part is yours
return StreamingResponse(sse(stream.subscribe()), media_type="text/event-stream")
```

One stream is one run: the `run_post` event closes it, and every subscriber runs
out of frames. A subscriber that arrives late gets the buffered replay first (up
to 1000 events), then whatever happens next. `emit` never blocks — a subscriber
too slow to keep up loses its own oldest event, never the run. `sse()` ends with
a `done` frame; `ws_frames()` gives one JSON object per frame.

stellar ships no server. These are strings; where they go is yours.

## Layout

```
core/             the nine nouns and the loop — stdlib only, under 2000 lines
  types.py          Part, ToolCall, Message
  contracts.py      Model, ProviderModel, Tool, Hook, fire, fold, the skeletons
  agent.py          Agent — one method, run()
  loop.py           run, check, coerce
  tool.py           @tool, on()
  fake.py           FakeModel
  conformance.py    check_model
  __init__.py       the barrel: all of it, in one import
models/           anthropic.py, openai.py — the only place httpx is allowed
hooks/            steps.py, budget.py, permission.py, logging.py
drivers/          events.py (StepEvent, EventStream), transport.py (sse, ws_frames)
tests/            plain python files, assert-based, no pytest
```

Imports are flat and one-way: `from core import Agent`,
`from models.anthropic import Anthropic`, `from hooks.steps import Steps`.
`models/`, `hooks/` and `drivers/` speak `core` and the standard library and
never each other. `core/` imports none of them.

## Running the tests

```
for f in tests/test_*.py; do uv run python "$f" || exit 1; done
uv run ruff check .
```

Every test file is an ordinary Python script. Run one on its own and it prints
what passed.

## The rules

`tests/test_rules.py` fails the build on four of them:

- `core/*.py` is 2000 lines all together, at most.
- `core/` imports the standard library and itself, nothing else.
- `core/` never says a product or protocol name — no mcp, anthropic, openai,
  litellm, claude, gpt, a2a, acp.
- `models/`, `hooks/` and `drivers/` import only `core` and the standard
  library, plus httpx in `models/`.

Two more belong to whoever reviews the change, because no test can see them:

- Nine nouns. A tenth noun in `core/` is a conversation, not a merge.
- `Agent` has one method, `run()`. There is never a second one.
