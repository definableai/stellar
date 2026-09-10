# stellar core

A minified agent core: the infrastructure to run an agent, kept under 2000
readable lines (`wc -l core/*.py`) and built on the standard library alone.
Small enough to read top to bottom in one sitting, strong enough for hard tasks
when it is used well. Everything past the loop — providers, rule cards,
transports — is a small file outside `core/` that you can read, copy and change.

## The robot with one backpack

An agent is a robot with one backpack. Inside: a brain (`model`), a toolbox
(`tools`), a stack of rule cards (`hooks`), a radio (`events`) and one spare
pocket (`extra`).

One errand is one `Run`: its own notebook (`messages`), its own step count, its
own id, its own spare pocket, and its own rule cards if it wants any. The robot
reads the cards before and after it thinks, and before and after every tool it
picks up. Everything that happens goes out over the radio. When the model stops
asking for tools, the errand is done.

```
run.pre
  ┌ model.pre → the model answers, a model.delta per streamed Part → model.post
  │   the answer goes in the notebook; for each tool it asked for:
  │     tool.pre → the tool runs, a tool.delta per yielded Part → tool.post
  │     → the result goes in the notebook
  └ again, as long as the last answer asked for tools
run.post
```

**One `Agent`, as many `Run`s as you like.** The agent is the wiring, built
once and shared; the run is the conversation, and nothing of it leaks into
another. A hundred users is one line:

```python
runs = await asyncio.gather(*(agent.run(prompt) for prompt in prompts))
```

Any part can be swapped by assignment, even mid-run: `check()` reads the whole
backpack again at the top of every step, so a bad swap is caught before it runs.
`agent.hooks.attach(card)` mid-flight reaches every run already in the air, at
its next stage; `run.hooks` is the lane nobody else can see.

## The twelve nouns

| noun | kind | the contract |
| --- | --- | --- |
| `Part` | data | one piece of a message: `type` names it, `data` holds it. Three types are reserved words core folds itself — `text`, `tool_call`, `meta`; everything else passes through untouched. |
| `ToolCall` | data | the model asking for one tool: `id`, `name`, `args`. |
| `Message` | data | one line in the notebook: `role`, `content`, `tool_calls`, `tool_call_id`, `meta`. `content` is always `list[Part]` — a str coerces at construction, and `.text` joins the text parts back. |
| `Model` | contract | the brain. Write an async `invoke(run)`; hand back a `Message`. |
| `Tool` | contract | one thing the agent can do. Write an async `execute(run, **args)` — an async generator streams. |
| `Hooks` | registry | the rule cards, filed by stage: `attach`, `detach`, `fire`. The agent has one, every run has one. |
| `Event` | data | one thing that happened: `seq`, `name`, `data`, `source`, `run_id`, `ts`. |
| `Events` | the radio | the bus: `emit`, `listen`, `detach`, `stream`. |
| `Agent` | the bag | `model`, `tools`, `hooks`, `events`, `extra`. One method: `run()`. |
| `Run` | the errand | one conversation: `id`, `messages`, `step`, `hooks`, `extra`, and `emit()`. |
| `ContractError` | signal | you wired it wrong. The message carries the code to type instead. |
| `Stop` | signal | end the run cleanly. Raise it from any hook or tool. |

Everything is async, and everything core calls takes the run: `invoke(run)`,
`execute(run, …)`. A hook takes the payload of its stage, and the run behind it
if it asks for one; an event listener takes the event. There are no sync twins
and no bridges: sync code is one `asyncio.to_thread` line the implementer writes
inside the async method.

`from core import …` also gives you the things that are not new nouns:
`ProviderModel`, `FakeModel` and `Fallback` (all three are Models), the `@tool`
and `@hook` decorators, `STAGES`, and the loop's own `run` and `check`.
`check_model` lives in `core.conformance`.

## Quickstart

Build the agent once. Call `run()` per conversation.

```python
import asyncio

from core import Agent, tool
from hooks.steps import Steps
from models.anthropic import Anthropic


@tool
def shout(word: str) -> str:
    """Shout a word."""
    return word.upper()


agent = Agent(Anthropic("claude-sonnet-5"), [shout])
agent.hooks.attach(Steps(8))                       # a runaway cap, on every run


async def main() -> None:
    run = await agent.run("shout hello")
    print(run.messages[-1].text)                   # what the model said last

    again = await agent.run("again, louder",       # same conversation, carried on
                            messages=run.messages, run_id=run.id)
    print(again.messages[-1].text)


asyncio.run(main())
```

`Anthropic("claude-sonnet-5")` names the model — one id per instance, no
default — and reads `ANTHROPIC_API_KEY` from the environment unless you pass
`api_key=`. `run()` takes a str (wrapped into a user message), a `Message`
(appended as it is — images and all), or `messages=` (the notebook to start
from; the list you hand over is copied, never touched). With nothing at all it
raises `ContractError` — a run with nothing to say is a typo. `run_id=` names
the run, which is how a stream finds it and how a resume keeps its identity.

What comes back is the `Run`: `run.messages` is the whole conversation,
`run.messages[-1].text` is the answer, `run.step` is how many turns it took.

## Models

### Three ways in

**Wrap what you already have.** Subclass `Model` and write an async `invoke`.
A sync SDK is one `to_thread` line away:

```python
class MyModel(Model):
    async def invoke(self, run) -> Message:
        said = await asyncio.to_thread(my_sdk.complete, run.messages)
        return Message("assistant", said)
```

**No base class.** Any object whose own class defines an async `invoke(run)`
passes `check()` and runs — nothing checks your ancestry. A sync `invoke` is
refused by `check()` up front, because the loop could never await it.

**`ProviderModel`.** An HTTP provider in two steps: `encode(run)` builds the
request body — pure translation, which is what makes it easy to test — and
`async send(run, body)` owns the network and yields the reply as Parts.
Everything that can go wrong on a network lives in `send`. A provider that
streams yields a Part per chunk; one that does not yields the Parts of its one
response, and core cannot tell the difference. `core/llm.py` writes the
other half for you — the recipe is below.

`send` speaks the Part protocol, and core folds the stream into one assistant
`Message`, firing `model.delta` per Part along the way:

- `Part("text", str)` is a delta — consecutive text parts concatenate into one.
- `Part("tool_call", ToolCall(…))` arrives whole — buffer partial JSON in the
  adapter, core never sees half an argument string.
- `Part("meta", dict)` merges into `Message.meta`, later keys winning.
- arguments that are not valid JSON are `ToolCall(id, name, {})` plus
  `meta["invalid_args"][id] = <the raw string>` — the loop turns that into the
  tool's result, so the model reads its own mistake instead of a TypeError.
- any other type lands in `content` untouched — thinking blocks, provider
  extras, whatever comes next. New provider features are new Part types,
  never new methods.

`encode` is where a `Part` gets opened. So adapters agree, one more spelling
is shared: `Part("image", {"url": …})` or
`Part("image", {"media_type": …, "data": <base64 str>})`.

Put what the reply cost in a meta Part, as
`{"usage": {"input_tokens": …, "output_tokens": …}}`. That is the convention
`Budget` reads; an adapter that leaves it out just counts as free. Beside it,
`meta["model"]` names who answered — the wire's own word for itself, else the
id you asked for, never `None`.

### A provider is a folder

`core/llm.py` holds the half every HTTP provider shares: the key, the profile,
one client, the retries, an SSE parser. A provider writes the other half in
two files. `mapping.py` is the table: what crosses the wire, one row per thing
that crosses. `model.py` is the rest — `PROFILES`, `headers`, `encode` (the
body's own keys), `send` (the stream's grammar) — and inherits everything else.

| core noun | the wire's word | the row |
| --- | --- | --- |
| `Part` | a block, a content part | many per message, keyed by type: `OUT`, and `IN` where the reply has blocks. |
| `Message` | a turn | one per message: `turn()` — and on Claude a `notebook()` over the lot, because system lines ride outside the turns. |
| `Tool` | a tool schema | one per tool: `tool()`. |
| `Message.meta` | the reply's header | one per reply: `meta()` — what it cost, who answered, why it stopped. |

Rows are named for what they make: an out-row makes a wire thing — `text`,
`image`, `tool_use`, `function` — and an in-row makes a core one, `tool_call`
or `meta`. There is no dispatcher to write: a call site reads
`OUT.get(p.type, passthrough)(p)`, and the else-rule is that `.get` default,
under a name of its own. Claude's is `passthrough`, and a block it never
opened goes back as it came; Chat Completions' is `unsaid`, which raises
`ContractError`, because a profile word promised that part.

```python
# models/openai/model.py — mapping.py sits beside it, __init__.py re-exports
from core import Profile, Provider

from .mapping import meta, tool, tool_call, turn

CHAT = frozenset({"image", "tools", "stream", "tool_stream", "json", "system"})


class OpenAI(Provider):
    url, env = "https://api.openai.com/v1", "OPENAI_API_KEY"
    PROFILES = {"gpt-5.6-luna": Profile("gpt-5.6-luna", 1_050_000, 128_000, CHAT)}

    def headers(self) -> dict: ...        # the key rides in here
    def encode(self, run) -> dict: ...    # pure, and it reads self.profile
    async def send(self, run, body): ...  # Parts, off self.sse() or self.post()
```

The base hands you `self.key`, `self.profile`, `self.params` (the extra
keywords you were built with), `tool_schemas(run)` (the toolbox as plain
dicts), and two ways out: `post(path, body)` for one reply, `sse(path, body)`
for a stream of dicts. Both try three times, on a dead socket or a status in
`{408, 409, 429, 500, 502, 503, 504, 529}`, honouring `retry-after`; `sse`
stops retrying once a chunk is out, because there is no asking again without
replaying it. `aclose()` closes the client; a loop swap closes the one it
replaces, on the loop that built it.

The model id is positional and required: `OpenAI("gpt-5.6-luna")`, one id per
instance, no silent default. `api_key=` beats the environment, and `env = None`
is a model that wants no key at all. A missing key raises `ValueError` naming
the variable. A refusal raises `ProviderError` with `.status`; a socket that
never opened after three tries, a data line that is not JSON, and an error
event inside a stream raise the same thing with `status None`.

A `Profile` is one row off the provider's docs: `id`, `context`, `max_output`
(the max-tokens `encode` writes, unless a param of yours says otherwise) and a
set of words. `"image" in model.profile` is the whole API — a router's
question, and the one `accept()` asks of every Part you put in the notebook,
and of the toolbox, before anything is sent: a picture nobody can read never
costs a round trip. One rule: **a feature named after a Part type admits that
Part in what you send** — user, system and tool messages. The model's own
output is never graded: it is its past, not your request. The words in the box
are `image`, `document`, `thinking`, `tools`, `stream`, `tool_stream`, `json`
and `system`.

- `stream` picks the path: with it, `encode` asks for the event stream and
  `send` yields a text Part per chunk; without it, one POST and the same parse.
- `tool_stream` puts the argument fragments on the radio as `tool.args` events,
  `{"id", "name", "delta"}`, source `model:<id>` — an event and not a Part,
  because a Part would land in the message.
- an id nobody wrote a row for takes the longest one it starts with
  (`gpt-5.6-luna-2026-08-01` → `gpt-5.6-luna`), else the `ANY` profile and a
  logged warning, never an error. The numbers in the shipped rows come from the
  providers' own docs; a row nobody could verify is absent, not invented.

A vendor that copied the wire is three lines, and a local model wants no key:

```python
class Groq(OpenAI):                          # in your code, not in models/
    url, env = "https://api.groq.com/openai/v1", "GROQ_API_KEY"
    PROFILES = {…}                           # its own rows, off its own docs


class Ollama(OpenAI):
    url, env = "http://localhost:11434/v1", None
```

A clone that cannot hold a stream open gets a profile without `stream` —
`profile=` in the constructor, or its own `PROFILES` — and takes the one-POST
path.

### Test with canned traffic, then grade it

Replace the network with a list of real response bodies, or a script of real
stream chunks — the reply still travels the adapter's own parsing and the real
fold path. No network, no key:

```python
from core.conformance import check_model
from core import Profile

ONCE = Profile("claude-sonnet-5", 1_000_000, 128_000, frozenset({"tools"}))  # no SSE


class Canned(Anthropic):
    def __init__(self, bodies) -> None:
        super().__init__("claude-sonnet-5", api_key="test", profile=ONCE)
        self.bodies, self.sent = list(bodies), []

    async def post(self, path, body) -> dict:
        self.sent.append(body)
        return self.bodies[len(self.sent) - 1]


check_model(Canned(BODIES))
```

`post` and `sse` are the seam: override `post` for the one-POST path, `sse`
for the streaming one, never httpx. `check_model` makes exactly three model
calls, always in this order: a plain reply, a reply asking for the `echo` tool,
and a final reply after the tool result — so a three-entry script lines up
with it.

It grades the reply side: do the Parts `send` yields fold into a `Message`,
do the deltas add up to what landed, do tool calls arrive as whole `ToolCall`s
with a dict of args, does the round trip close. It cannot see whether `encode`
built a body your provider would accept — assert that yourself against
`model.encode(run)`. `tests/test_anthropic.py` and `tests/test_openai.py` do
both halves. `check_model` calls `asyncio.run` inside, so call it from sync
code only.

`FakeModel` is a `ProviderModel` whose `send` yields the Parts of a scripted
reply — tests ride the same fold and delta path with no network at all.

### The two shipped ones

`models/anthropic/` — the Messages API. `models/openai/` — Chat Completions,
the format the compatible vendors clone; point `base_url` at a clone and the
same class talks to it. Both name their model first —
`Anthropic("claude-sonnet-5")`, `OpenAI("gpt-5.6-luna")` — take `api_key=` or
read the environment, pass any extra keyword straight through to the request
body, and stream off the socket whenever the profile says `stream`.

To read a wire, open its `mapping.py`; the Chat Completions one says out loud
that its reply has no block types — text arrives as a string, tool calls as a
list — so it has an `OUT` and no `IN`.

A tool result that is more than text is where the two part. Claude puts the
tool's blocks inside the `tool_result` — its text when there is only text, else
every block it holds. Chat Completions keeps the tool line to text and sends
the pictures off a run of tool lines in one user line right after that run,
each captioned `[image returned by tool call <id>]`.

Chat Completions has no shape for a thinking block, so an assistant turn drops
what this wire cannot say — the model's own past, off another wire, replays
without it. A part *you* authored stays loud; nothing you wrote goes missing in
silence.

### Fallback, and routing

Routing composes Models. It is not a setting inside a provider: `Fallback` is a
`Model` made of `Model`s, asked in order, and the first that takes the run and
answers wins. That is why it crosses providers for free — the notebook is
core-shaped, and every adapter encodes from `Message`, never from another
adapter's wire.

```python
from core import Agent, Fallback
from models.anthropic import Anthropic
from models.openai import OpenAI

agent = Agent(
    model=Fallback(                                   # order = preference
        Anthropic("claude-opus-5"),
        OpenAI("gpt-5.6-terra", reasoning_effort="high"),
        OpenAI("gpt-5.6-luna"),                       # the last one's error is yours
    ),
    tools=[write],
)
run = await agent.run("what is in this pdf?")         # a document: opus takes it
run.messages[-1].meta["model"]                        # who actually answered
```

It catches exactly two things: `Unsupported`, the gate's no before any byte — a
picture on a text-only model costs no round trip — and `ProviderError`, the
wire's no with its retries already spent. Everything else stays loud,
`ContractError` first among them: a wiring mistake is not a reason to ask
someone else. The signature is `Fallback(first, *rest)`, one Model minimum, and
the last one is asked outside the net, so its refusal reaches you as itself.

Every skip rides the bus as a `model.skip` event carrying the refusal's own
words, source `model:<id>` — or the class name, for a Model that has no id. The
reply's `meta["model"]` is how you tell who ended up answering.

A smarter policy — cheap first, by toolbox or by notebook length — is a Model
of your own, and it nests inside a `Fallback`, because every layer here is a
Model:

```python
class Cheap(Model):
    def __init__(self, small, big) -> None:
        self.small, self.big = small, big

    async def invoke(self, run) -> Message:
        long = sum(len(m.text) for m in run.messages) > 20_000
        return await (self.big if long else self.small).invoke(run)


model = Fallback(Cheap(haiku, opus), OpenAI("gpt-5.6-luna"))
```

Two ceilings. A provider that is down is retried in full — three tries, with
backoff — at every step before the next one is asked; add a per-model cooldown
when that starts to hurt. And a stream that dies after its first delta falls
through like any other refusal: the notebook comes out right, but the radio
already carried the dead one's deltas.

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
default is required. `@tool` takes one keyword, `parallel=`, and nothing else —
for a different name, description or schema, write a `Tool` subclass; that is
what the class is for. It refuses `*args`/`**kwargs` and says so.

A sync function runs in a thread, an async one is awaited, and an async
generator streams — every `Part` it yields rings `tool.delta`.

A first parameter spelled exactly `run` is filled in by the loop and kept out
of the schema, so the model never sees it:

```python
@tool
def notebook(run) -> int:
    """How many lines the notebook holds."""
    return len(run.messages)
```

### Parallel tools

```python
@tool(parallel=True)                    # may run beside other parallel tools
def read(path: str) -> str: ...


@tool                                   # serial, the default: a barrier
def write(path: str, content: str) -> str: ...


class Search(Tool):
    parallel = True                     # the class form
```

The model's turn asks for N calls, in order, and the loop cuts them into
batches: consecutive calls on `parallel=True` tools ride together, any other
call rides alone. **A serial tool is a barrier** — everything before it
finishes first, it runs by itself, then the next batch starts. Serial is the
default, so nothing runs together until you opt a tool in.

Inside one batch every `tool.pre` rings first, in call order — a permission
card still asks one question at a time, and a denial skips that call only.
Then the calls run at once. Then every `tool.post` rings and its result lands
in the notebook, in call order: the notebook reads the same whichever tool
finished first. `tool.delta` from streaming siblings interleaves, and each
event names its own tool in `source`.

The batches are cut from the names the model asked for, before any `tool.pre`
rings, so a hook that reroutes a call to a differently-parallel tool does not
re-batch it.

A `Stop` or a `ContractError` from any tool in a batch is raised once the whole
batch is done, first in call order — nothing is cancelled mid-write, so a
half-written file is never left behind. The replies before it are already in
the notebook; the ones behind it are dropped, though their tools ran.

Two ceilings. A slow sibling delays the `Stop` by its own runtime. There is no
concurrency limit — a turn asking for fifty parallel calls runs fifty at once;
add a semaphore in `act()` when that hurts.

### A class

```python
class Shout(Tool):
    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    async def execute(self, run, word) -> str:
        return word.upper()
```

`execute` is async; wrap sync work in one `asyncio.to_thread` line. A class is
the answer whenever the tool holds something. Write `execute` as an async
generator and the tool streams: each yielded `Part` rings `tool.delta` and
folds into the tool message, same protocol as a streaming model.

### What comes back

Return a `str` and the model sees it. Return a `Part`, or a list of them, and
they are the result's content — a picture, or text and a picture together.
Return anything else and it arrives as JSON — whatever JSON cannot hold becomes
its `str`, and an empty list is still `[]`. Return a `Message` and it is used as
it is, with its role forced to `tool` and the call's id filled in. An async
generator yields those same Parts one at a time, and its result is the fold of
them.

The gate grades a tool's result like anything else you wrote: an image Part on
a profile without the word `image` raises `Unsupported`, before the network.

Raise, and the model sees `error: ValueError: nope` and gets another turn. Ask
for a tool that is not in the toolbox and it sees
`error: unknown tool: ghost`. Run-time mistakes are answers, not crashes. The
two exceptions are `Stop`, which always ends the run, and `ContractError`,
which stays loud.

`agent.tools` is a mapping of name to tool. Hand the constructor an iterable and
it keys one for you — `Agent(model, [shout])` — or hand it the mapping. Two
tools answering to one name is a `ContractError`, and a tool filed mid-run under
a key that is not its name is caught at the top of the next step, like every
other bad swap.

## Hooks — the control plane

A hook is an async function tagged with the stages it listens for. It fires
inline, and what it returns replaces the payload.

```python
@hook("tool.pre")
async def no_rm(call: ToolCall):
    if call.name == "rm":
        return "denied"                 # the tool is skipped; this is its result


agent.hooks.attach(no_rm)               # every run
run.hooks.attach(no_rm)                 # only this one
```

The eight stages are the eight moments, and nothing else is a moment.

| stage | the payload it hands over | what a return replaces |
| --- | --- | --- |
| `run.pre` | `Message` — the last message, once, before the first turn | that message |
| `model.pre` | `list[Message]` — the notebook, before every model call | rebinds `run.messages`, permanently |
| `model.delta` | `Part` — one streamed piece, before it is folded | the Part, and the new one is folded |
| `model.post` | `Message` — the reply, after every model call | the reply that gets written down |
| `tool.pre` | `ToolCall` — before every tool | a `ToolCall` runs instead; a `str` or `Message` skips the tool and *is* its result |
| `tool.delta` | `Part` — one yielded piece, before it is folded | the Part, and the new one is folded |
| `tool.post` | `ToolCall, Message` — the call and its result | the result |
| `run.post` | `Message` — the last message, once, at the very end | nothing; this is teardown |

Return `None` and the payload is left alone. Return the wrong type and it is a
`ContractError` naming your function, the stage and what to return instead. A
hook may take one argument more than its stage hands over, and that one is the
run:

```python
@hook("model.pre")
async def trim(messages, run) -> list[Message]:
    return messages[-20:] if run.step > 3 else messages
```

The arity is sniffed once, at attach — a hook that does not fit the stage is
refused there, not at the bell. `@hook` only tags; `attach` files, and a stage
named at attach beats the tag: `agent.hooks.attach(watch, "run.post")` works on
an untagged function. Hooks must be async; a sync one is refused with the
`to_thread` line to write instead.

Attach order is call order, the agent's cards ring before the run's, and a card
attached during a bell starts listening at the next one. At `tool.pre`, the
first hook that answers with a result ends the ring — the cards behind it never
hear that bell.

### Stopping

`raise Stop` from any hook or any tool ends the run cleanly: no traceback
reaches the caller, and you get the `Run` back holding everything written down
so far.

`run.post` always fires — after a clean finish, after a `Stop`, and after a
crash. It is a `finally`, so a `Stop` raised inside `run.post` is ignored (too
late to stop anything), and any other exception from `run.post` propagates the
ordinary way, with whatever was already in flight attached as `__context__`.

Everything else propagates: a model that raises, a hook that raises, a
`ContractError`. The loop catches exactly two things — a `Stop`, and a tool's
own exception, which becomes a result the model reads.

### Denying a tool

Return a `str` or a `Message` from a `tool.pre` hook. The loop skips the tool
and writes your answer down as its result for the model to read. That is the
whole deny path — no third signal, no boolean to return.

## Events — the data plane

Everything that happens is also an `Event` on the agent's bus: numbered per bus,
stamped with the run it belongs to, kept in `agent.events.log`.

```python
agent.events.listen(print)                          # everything
agent.events.listen(watch, "tool")                  # tool.pre, tool.delta, tool.post
agent.events.listen(watch, run_id=run.id)           # one conversation
```

A prefix stops at a dot: `"tool"` hears `tool.pre`, never `toolbox.x`.
Listeners are sync and `emit()` never awaits — a listener that raises is logged
to `core.events` and swallowed, and the next one still hears the event. That is
the difference between the planes: a hook can change the run or stop it, a
listener can only watch.

The loop emits every stage *after* its hooks have run, so a listener sees what
the cards agreed on — `source="loop"`, or the tool's own name on `tool.delta`.
Anything can emit anything: `run.emit("cache.hit", key)` stamps it with the
run's id and it travels with the rest.

For an async consumer, take the stream:

```python
async for event in agent.events.stream(run_id=rid, since=cursor):
    ...
```

`stream()` replays the log from `since` (inclusive), then goes live — no gap,
no doubles, in seq order. Given a `run_id` it ends on that run's `run.post`,
prefix or no prefix, so a reader never hangs on a finished run. The event log
and the stream queues are unbounded: one bus per agent, one process. Cap them
when that stops being true.

## The four shipped cards

| card | plane | what it does |
| --- | --- | --- |
| `Steps(n)` | hook, `model.pre` | n model calls per run, then `Stop`. `Steps(3)` buys exactly three replies. |
| `Budget(max_tokens)` | hook, `model.post` | adds `input_tokens + output_tokens` from every reply's `meta["usage"]` into `run.extra["budget.tokens"]`. Past the limit it keeps the reply you already paid for, then stops. A reply with no usage costs nothing. |
| `Permission(deny, allow, ask)` | hook, `tool.pre` | the guard at the tool gate. `deny` wins; then an `allow` list settles it alone; only then does `ask=True` put the question to a human. Each of `deny`/`allow` is a set of names or a predicate `(run, call) -> bool`. |
| `Log(write=print)` | listener | one readable line per event: its name, and what it carried. |

```python
agent.hooks.attach(Steps(8))
agent.hooks.attach(Permission(deny={"rm"}, ask=True))
agent.events.listen(Log())
```

Each card is a factory: it hands back the function, and the function is what you
attach. Nothing is kept in the card — `Steps` counts on `run.step`, `Budget`
totals into `run.extra` — so one card on the agent gives every run its own
ration, its own purse, its own answer.

`Permission(ask=True)` asks through `extra["ui.ask"]`, an
`async (question, options=None) -> str` that you provide. It looks in the run's
pocket first and the agent's after, so one channel can serve every conversation
or each one can bring its own. Anything but `"allow"` denies, and no channel at
all denies too — a guard with no voice says no.

## Patterns

### A sub-agent is a tool

Not a feature. A tool that builds a child agent on the parent's brain, runs it,
and hands back what it said:

```python
class Research(Tool):
    name = "research"
    description = "Ask a helper agent one question."
    parameters = {"type": "object", "properties": {"question": {"type": "string"}}}

    async def execute(self, run, question) -> str:
        helper = await Agent(run.agent.model).run(question)
        return helper.messages[-1].text
```

The child gets its own notebook, its own bus and its own spare pocket, so
nothing comes back except the sentence you return. Hand it its own tools and
hooks when it should be allowed less than its parent.

### Checkpoint and resume

A checkpoint is the `Run` — `id`, `messages`, `step`, `extra`, and nothing
else. `__getstate__` drops the agent and the hooks, because the wiring is code
you build again:

```python
frozen = pickle.dumps(run)                          # four fields, no wiring
...
saved = pickle.loads(frozen)
run = await agent.run("and now?", messages=saved.messages, run_id=saved.id)
```

The revived object is a checkpoint, not a live run: hand its parts back to
`agent.run` and you get a fresh `Run` under the same id. `extra` starts empty
there — if a card kept a total in it, seed it back yourself (a `run.pre` hook is
one line).

To put a checkpoint somewhere that is not a pickle, `dataclasses.asdict` each
`Message`. Reading it back, rebuild the nouns inside it yourself — `content` is
always a list of parts, so there is exactly one shape to revive:

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

The bus already keeps one run's events in order and ends the stream on
`run.post`. The transports only frame them:

```python
rid = uuid4().hex
asyncio.create_task(agent.run(ask, run_id=rid))   # hold the task; that part is yours
return StreamingResponse(sse(agent.events.stream(run_id=rid)),
                         media_type="text/event-stream")
```

`sse()` writes `id: <seq>` and one JSON `data:` line per event — the whole
record, nested `Message`s and `Part`s and all. The seq is the cursor: a browser
that drops sends its last id back as `Last-Event-ID`, and
`stream(run_id=rid, since=int(last) + 1)` hands over exactly what the drop cost.
`ws_frames()` gives the same records, one JSON object per frame.

The replay is why the race does not matter: a reader that opens the stream after
the run started — or after it finished — still gets the whole run, in order.

stellar ships no server. These are strings; where they go is yours.

## Layout

```
core/             the twelve nouns and the loop — stdlib (bar llm.py), under 2000 lines
  types.py          Part, ToolCall, Message
  contracts.py      Model, ProviderModel, Tool, Hooks, @hook, fold, the skeletons
  agent.py          Agent and Run
  loop.py           run, check, coerce
  events.py         Event, Events
  tool.py           @tool
  fake.py           FakeModel
  llm.py            Provider, Profile, Fallback — the HTTP half; the only httpx in core
  conformance.py    check_model
  __init__.py       the barrel: all of it, in one import
models/           one folder per provider: __init__.py, mapping.py, model.py
  anthropic/        the Messages API
  openai/           Chat Completions, the format the compatible vendors clone
hooks/            steps.py, budget.py, permission.py, logging.py
drivers/          transport.py — sse() and ws_frames() over events.stream()
tests/            plain python files, assert-based, no pytest
docs/             the site: docs.json is the table, content/ the pages, src/ the app
main.py           the front door: one agent, one tool, one card, one radio
```

Imports are flat and one-way: `from core import Agent`,
`from models.anthropic import Anthropic`, `from hooks.steps import Steps`.
`models/`, `hooks/` and `drivers/` speak `core` and the standard library and
never each other. A provider's own files speak each other with one dot —
`from .mapping import turn` — and never reach into another provider, so a
clone subclasses `OpenAI` in your code, not in `models/`. Nothing imports an
adapter, and `core/` imports none of them.

## The docs

`docs/` is a site of its own — the pages are MDX under `docs/content/`, the
table of contents is `docs/docs.json`, and a small React app renders them.
`cd docs && npm install && npm run dev` serves it; `npm run build` writes the
static files; `python3 docs/check.py` holds every page to the table, every
link to a page and every python fence to the parser. `docs/README.md` says
how to add a page.

## Running the tests

```
for f in tests/test_*.py; do uv run python "$f" || exit 1; done
uv run ruff check .
```

Every test file is an ordinary Python script. Run one on its own and it prints
what passed. `uv run python` is how it gets an interpreter — a bare `python`
need not be on your PATH.

## The rules

`tests/test_rules.py` fails the build on four of them:

- `core/*.py` is 2000 lines all together, at most.
- `core/` imports the standard library and itself, nothing else — plus httpx,
  in `llm.py` and nowhere else.
- `core/` never says a product or protocol name — no mcp, anthropic, openai,
  litellm, claude, gpt, a2a, acp.
- `models/`, `hooks/` and `drivers/` import only `core` and the standard
  library, plus httpx in `models/` — and a provider's own files, one dot deep.

Two more belong to whoever reviews the change, because no test can see them:

- Twelve nouns. A thirteenth noun in `core/` is a conversation, not a merge.
- `Agent` has one method, `run()`. There is never a second one.
