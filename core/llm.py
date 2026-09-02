"""The HTTP half of a Model: the key, the profile, the socket.

A provider file writes four things — PROFILES, headers, encode, send — and
inherits the rest of a Model from here. A Profile says what one model id can
do; accept() reads it before anything is sent, so a picture nobody can see
never costs a round trip. Two ways out, retried the same way: post() for one
reply, sse() for a stream of them.

The one file in core that speaks httpx: a socket is what it is for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, AsyncIterator, ClassVar, cast

import httpx

from core.contracts import ProviderModel, Tool
from core.types import Message, Part

if TYPE_CHECKING:                   # the checker's eyes only: no runtime edge
    from core.agent import Run

__all__ = ["ALWAYS", "ANY", "Profile", "Provider", "ProviderError", "RETRY",
           "Unsupported", "args_of"]

RETRY = {408, 409, 429, 500, 502, 503, 504, 529}   # worth asking again
ALWAYS = frozenset({"text", "tool_call", "meta"})  # core folds these itself


class ProviderError(RuntimeError):
    """The provider said no.

    status is None when there was no HTTP status to give — a dead socket, a
    broken chunk, an error event inside a stream.
    """

    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(f"{status or 'no status'}: {detail}")
        self.status = status


class Unsupported(ValueError):
    """This model cannot do what the run asks — said before the network."""


@dataclass(frozen=True)
class Profile:
    """One row off the provider's docs: what this model id can do.

    `"image" in profile` is the whole API — a router's question, and the one
    accept() asks of every Part. A feature named after a Part type admits
    that Part; the rest are capabilities the adapter reads.
    """

    id: str
    context: int                        # input window, tokens
    max_output: int                     # default max output tokens
    features: frozenset[str] = frozenset()

    def __contains__(self, feature: str) -> bool:
        """True when this model has that word."""
        return feature in self.features


ANY = Profile("unknown", 128_000, 4096, frozenset(
    {"image", "document", "thinking", "tools", "stream", "tool_stream",
     "json", "system"}))                # what an unknown id gets, once warned


class Provider(ProviderModel):
    """The other half of every adapter: a key, a profile, one client.

    Set url, env and PROFILES, then write headers, encode and send. Nothing
    here knows a wire format — encode and send stay abstract, and core's
    skeletons already say what to type.
    """

    url: ClassVar[str]                  # the default base_url
    env: ClassVar[str | None]           # the key's variable; None = no key needed
    PROFILES: ClassVar[dict[str, Profile]] = {}

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        profile: Profile | None = None,
        **params,
    ) -> None:
        """One model id per instance — its profile comes along with it."""
        self.model = model
        self.params = params            # temperature, max_tokens, whatever else
        self.key = api_key or (os.environ.get(self.env, "") if self.env else "")
        if self.env and not self.key:
            raise ValueError(f"no API key: pass api_key= or set {self.env}")
        self.base_url = base_url or self.url
        self.profile = profile or self.lookup(model)
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def headers(self) -> dict:
        """What every request carries — the key goes in here."""
        raise NotImplementedError("headers")

    async def invoke(self, run: Run) -> Message:
        """The gate first, then core's own encode, send and fold."""
        self.accept(run)
        return await super().invoke(run)

    def accept(self, run: Run) -> None:
        """Refuse here what the wire would refuse later, for free.

        A content Part the profile does not name, or a toolbox on a model
        without tools, raises Unsupported — nothing has been sent yet. Only
        what you authored is graded: the model's own turns go back as they came.
        """
        if run.agent.tools and "tools" not in self.profile:
            raise Unsupported(f"{self.profile.id} takes no tools; empty the "
                              "toolbox, or name a model whose profile has them")
        for message in run.messages:    # every line you wrote: a string compare
            if message.role == "assistant":
                continue                # block() and line() replay it untouched
            for piece in cast(list[Part], message.content):
                if piece.type not in ALWAYS and piece.type not in self.profile:
                    raise Unsupported(
                        f"{self.profile.id} reads no {piece.type!r} part; it "
                        "reads " + ", ".join(sorted(self.profile.features)))

    @classmethod
    def lookup(cls, model: str) -> Profile:
        """The row for this id: the longest one it starts with, else everything."""
        known = [k for k in cls.PROFILES if model.startswith(k)]
        if known:
            return cls.PROFILES[max(known, key=len)]   # an id is its own longest
        logging.getLogger(__name__).warning(
            "%s: no profile for %r — assuming it can do everything",
            cls.__name__, model)
        return replace(ANY, id=model)

    def tool_schemas(self, run: Run) -> list[dict]:
        """The toolbox, one dict each; encode renames the keys its wire wants."""
        toolbox = cast(Mapping[str, Tool], run.agent.tools)   # check() proved it
        return [{"name": t.name, "description": t.description,
                 "parameters": t.parameters} for t in toolbox.values()]

    @property
    def http(self) -> httpx.AsyncClient:
        """One client per instance, opened on first use — and again per loop.

        A client outlives the loop that made it; its connection pool does
        not, so a second asyncio.run() gets a fresh one. The one it replaces is
        closed on the loop that built it, if that loop still runs; a closed
        loop took its sockets down with it, so that one is only dropped.
        """
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            old, was = self._client, self._loop
            if old is not None and was is not None and not was.is_closed():
                was.call_soon_threadsafe(was.create_task, old.aclose())
            self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=self.headers(),
                timeout=httpx.Timeout(10, read=600))
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        """Close the client, if one was ever opened. Nothing else holds a socket."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post(self, path: str, body: dict) -> dict:
        """One reply as a dict. Three tries, then the failure is yours to read.

        A status in RETRY earns another go — after retry-after, or the
        attempt number when nobody said. A dead socket earns the same.
        """
        for attempt in (1, 2, 3):
            try:
                answer = await self.http.post(path, json=body)
            except httpx.TransportError as ex:
                if attempt == 3:
                    raise ProviderError(None, f"{type(ex).__name__}: {ex}") from ex
                nap = float(attempt)
            else:
                if answer.status_code < 400:
                    return answer.json()
                if attempt == 3 or answer.status_code not in RETRY:
                    raise ProviderError(answer.status_code, answer.text[:200])
                nap = wait(answer, attempt)
            await asyncio.sleep(nap)
        raise AssertionError("unreachable: the last try returns or raises")

    async def sse(self, path: str, body: dict) -> AsyncIterator[dict]:
        """The event stream, one dict per data line — comments and [DONE] skipped.

        Retried like post, but only up to the first chunk: once a piece has
        been yielded there is no asking again without replaying it.
        """
        started = False
        for attempt in (1, 2, 3):
            nap = float(attempt)
            try:
                async with self.http.stream("POST", path, json=body) as answer:
                    if answer.status_code >= 400:
                        await answer.aread()            # .text wants the body first
                        if attempt == 3 or answer.status_code not in RETRY:
                            raise ProviderError(answer.status_code,
                                                answer.text[:200])
                        nap = wait(answer, attempt)
                    else:
                        async for text in answer.aiter_lines():
                            piece = chunk(text)
                            if piece is not None:
                                started = True
                                yield piece
                        return
            except httpx.TransportError as ex:
                if started or attempt == 3:
                    raise ProviderError(None, f"{type(ex).__name__}: {ex}") from ex
            await asyncio.sleep(nap)


def wait(answer: httpx.Response, attempt: int) -> float:
    """How long to hold off: what retry-after says, else the attempt number."""
    try:
        return float(answer.headers.get("retry-after", ""))
    except ValueError:                  # absent, or a date — our own guess wins
        return float(attempt)


def chunk(line: str) -> dict | None:
    """One line off the stream as a dict — None for anything that carries nothing."""
    if not line.startswith("data:"):
        return None                     # blank lines, `:` comments, `event:` names
    data = line[5:].strip()
    try:
        return None if not data or data == "[DONE]" else json.loads(data)
    except ValueError as ex:
        raise ProviderError(None, f"not JSON: {data[:200]}") from ex


def args_of(raw: str) -> tuple[dict, str | None]:
    """The arguments as one dict — or an empty one and the string back, unread."""
    try:
        args = json.loads(raw or "{}")
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
    except ValueError:                  # a broken parse is a ValueError too
        return {}, raw
    return args, None
