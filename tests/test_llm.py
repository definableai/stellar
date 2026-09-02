"""The shared half of an adapter: the profile gate, the key, the retries, the stream.

Run: uv run python tests/test_llm.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, Message, Part, Run, llm, tool  # noqa: E402
from core.llm import ANY, Profile, Provider, ProviderError, Unsupported  # noqa: E402

WAITS: list[float] = []
STREAM = (b": ready\n\n"
          b'data: {"n": 1}\n\n'
          b"\n"
          b'data: {"n": 2}\n\n'
          b"data: [DONE]\n\n")


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


async def nap(seconds: float) -> None:
    """asyncio.sleep, minus the sleeping: the waits are recorded instead."""
    WAITS.append(seconds)


llm.asyncio = SimpleNamespace(    # the module's own name for it, and only there
    sleep=nap, get_running_loop=asyncio.get_running_loop)


class Heard(logging.Handler):
    """Every line the module logged, so a test can read them back."""

    said: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.said.append(record.getMessage())


class Stub(Provider):
    """A provider of nothing: the base class with its four blanks filled in."""

    url, env = "https://stub.test/v1", "STUB_API_KEY"
    PROFILES = {
        "small": Profile("small", 8_000, 512, frozenset({"stream"})),
        "small-2": Profile("small-2", 8_000, 512, frozenset({"tools", "sound"})),
        "big": Profile("big", 200_000, 4_096, frozenset({"image", "tools"})),
    }

    def headers(self) -> dict:
        return {"authorization": self.key}

    def encode(self, run) -> dict:
        return {"model": self.model, "lines": len(run.messages)}

    async def send(self, run, body):
        yield Part("text", "ok")


class Wired(Stub):
    """The same stub, with the socket answering off a list of canned replies."""

    def __init__(self, *script) -> None:
        super().__init__("big", api_key="k")
        self.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.reply), base_url=self.base_url)
        self.script, self.sent = list(script), []

    def reply(self, request: httpx.Request) -> httpx.Response:
        """The next canned reply — or the next canned failure."""
        self.sent.append(str(request.url))
        answer = self.script.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def http(self) -> httpx.AsyncClient:
        return self.client


def test_lookup_takes_the_longest_id_it_starts_with() -> None:
    assert Stub.lookup("big") is Stub.PROFILES["big"]                 # exact
    assert Stub.lookup("small-2-2026-08-01") is Stub.PROFILES["small-2"]
    said = logging.getLogger("core.llm")
    said.addHandler(Heard())
    unknown = Stub.lookup("who-knows")
    said.handlers.pop()
    assert unknown.id == "who-knows" and unknown.features == ANY.features
    assert len(Heard.said) == 1 and "who-knows" in Heard.said[0]


def test_the_profile_gates_what_the_run_asks_for() -> None:
    small, big = Stub("small", api_key="k"), Stub("big", api_key="k")
    words = Message("user", "hello")
    picture = Message("user", [Part("image", {"url": "https://stub.test/cat.png"})])
    assert refused(big, [words]) == ""                    # text is always allowed
    assert refused(big, [picture]) == ""                  # "image" is in the profile
    assert "'image'" in refused(small, [picture])         # and not in this one
    assert "tools" in refused(small, [words], [echo])     # nor is a toolbox
    assert refused(big, [words], [echo]) == ""
    sound = Message("user", [Part("sound", b"...")])
    assert refused(Stub("small-2", api_key="k"), [sound]) == ""   # a word is a word
    assert big.tool_schemas(run_of(big, [words], [echo])) == [
        {"name": "echo", "description": "Repeat the text back.",
         "parameters": echo.parameters}]


def test_invoke_asks_the_gate_before_it_folds() -> None:
    small = Stub("small", api_key="k")
    said = asyncio.run(small.invoke(run_of(small, [Message("user", "hi")])))
    assert said.text == "ok"                              # encode, send, fold
    picture = Message("user", [Part("image", {"url": "u"})])
    try:
        asyncio.run(small.invoke(run_of(small, [picture])))
    except Unsupported:
        pass
    else:
        raise AssertionError("invoke must ask accept() first")


def test_the_key_comes_from_you_or_the_environment() -> None:
    os.environ["STUB_API_KEY"] = "from-env"
    assert Stub("big").key == "from-env"
    assert Stub("big", api_key="mine").key == "mine"      # explicit wins
    del os.environ["STUB_API_KEY"]
    try:
        Stub("big")
    except ValueError as ex:
        assert "STUB_API_KEY" in str(ex)                  # which variable to set
    else:
        raise AssertionError("a model with no key must say so at build time")

    class Local(Stub):
        """A model on your own machine: no variable, no key, no complaint."""

        env = None

    assert Local("big").key == ""


def test_post_tries_three_times_for_what_is_worth_retrying() -> None:
    WAITS.clear()
    model = Wired(busy(429, "0"), httpx.Response(200, json={"said": "hi"}))
    assert asyncio.run(model.post("say", {})) == {"said": "hi"}
    assert WAITS == [0.0]                                 # retry-after, to the second
    assert model.sent == ["https://stub.test/v1/say"] * 2  # base_url, then the path

    WAITS.clear()
    assert status(Wired(busy(400))) == 400                # ours to fix, not to repeat
    assert WAITS == []

    WAITS.clear()
    assert status(Wired(busy(503), busy(503), busy(503))) == 503
    assert WAITS == [1.0, 2.0]                            # three tries, two waits

    WAITS.clear()
    dead = Wired(*[httpx.ConnectError("no route") for _ in range(3)])
    try:
        asyncio.run(dead.post("say", {}))
    except httpx.TransportError:
        pass
    else:
        raise AssertionError("a socket that never opened is not a ProviderError")
    assert WAITS == [1.0, 2.0]

    WAITS.clear()
    asyncio.run(Wired(busy(429, "soon"), httpx.Response(200, json={})).post("s", {}))
    assert WAITS == [1.0]                                 # unparsable: the attempt


def test_sse_yields_the_data_lines_and_nothing_else() -> None:
    WAITS.clear()
    said = asyncio.run(drain(Wired(httpx.Response(200, content=STREAM))))
    assert said == [{"n": 1}, {"n": 2}]                   # no comment, no [DONE]

    model = Wired(busy(503), httpx.Response(200, content=STREAM))
    assert asyncio.run(drain(model)) == [{"n": 1}, {"n": 2}]
    assert WAITS == [1.0]                                 # the connect is retried

    try:
        asyncio.run(drain(Wired(busy(400))))
    except ProviderError as ex:
        assert ex.status == 400
    else:
        raise AssertionError("a 400 must not come back as an empty stream")


def test_the_client_is_one_per_instance_and_one_per_loop() -> None:
    model = Stub("big", api_key="k")

    async def twice() -> tuple:
        return model.http, model.http

    one, again = asyncio.run(twice())
    assert one is again                          # inside one loop, one client
    two, _ = asyncio.run(twice())
    assert two is not one                        # a new loop: the old pool is dead

    async def shut() -> httpx.AsyncClient:
        await model.aclose()
        return model.http

    assert asyncio.run(shut()) is not two and two.is_closed


def run_of(model, messages, tools=()) -> Run:
    """One Run built by hand: the gate needs a notebook and a toolbox, no loop."""
    return Run(Agent(model, tools), "rid", list(messages))


def refused(model, messages, tools=()) -> str:
    """What accept() said about this run — or "" when it let it by."""
    try:
        model.accept(run_of(model, messages, tools))
    except Unsupported as ex:
        return str(ex)
    return ""


def busy(status: int, retry_after: str = "") -> httpx.Response:
    """A refusal, carrying the header the retry reads when a test sets it."""
    return httpx.Response(status, text="too busy",
                          headers={"retry-after": retry_after} if retry_after else {})


def status(model) -> int | None:
    """The status the ProviderError carried; a reply that does not raise fails."""
    try:
        asyncio.run(model.post("say", {}))
    except ProviderError as ex:
        return ex.status
    raise AssertionError("that reply should have raised ProviderError")


async def drain(model) -> list[dict]:
    """Every chunk one stream yields."""
    return [piece async for piece in model.sse("say", {})]


if __name__ == "__main__":
    for test in (
        test_lookup_takes_the_longest_id_it_starts_with,
        test_the_profile_gates_what_the_run_asks_for,
        test_invoke_asks_the_gate_before_it_folds,
        test_the_key_comes_from_you_or_the_environment,
        test_post_tries_three_times_for_what_is_worth_retrying,
        test_sse_yields_the_data_lines_and_nothing_else,
        test_the_client_is_one_per_instance_and_one_per_loop,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_llm: all ok")
