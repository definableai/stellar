"""kind(): a type hint as a JSON schema, all the way down.

Run: uv run python tests/test_schema.py
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tool import kind, tool  # noqa: E402


class Colour(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Point:
    x: int
    y: Annotated[int, "how far down"]
    label: str = "here"
    tags: list[str] = field(default_factory=list)


class Card(TypedDict, total=False):
    title: Annotated[str, "shown to the reader"]
    lines: list[int]


class Ticket(TypedDict):
    id: int
    seat: str


def test_leaves_come_from_the_table() -> None:
    assert [kind(t) for t in (str, int, float, bool)] == [
        {"type": "string"}, {"type": "integer"},
        {"type": "number"}, {"type": "boolean"},
    ]
    assert [kind(t) for t in (list, tuple, set, dict)] == [
        {"type": "array"}, {"type": "array"},
        {"type": "array"}, {"type": "object"},
    ]
    assert kind(Any) == kind(object) == {}          # unknown promises nothing


def test_a_sequence_carries_its_items() -> None:
    assert kind(list[str]) == {"type": "array", "items": {"type": "string"}}
    assert kind(set[int]) == {"type": "array", "items": {"type": "integer"}}
    assert kind(tuple[bool, ...]) == {"type": "array", "items": {"type": "boolean"}}
    assert kind(list[list[str]]) == {                # and nesting goes on down
        "type": "array", "items": {"type": "array", "items": {"type": "string"}}}
    assert kind(tuple[int, str]) == {"type": "array"}   # a fixed tuple: no one item


def test_a_mapping_is_a_bare_object() -> None:
    assert kind(dict[str, int]) == {"type": "object"}   # the value hint goes unsaid
    assert kind(dict[str, list[str]]) == {"type": "object"}


def test_none_is_admitted_beside_the_type() -> None:
    assert kind(str | None) == {"type": ["string", "null"]}
    assert kind(Optional[int]) == {"type": ["integer", "null"]}
    assert kind(Union[float, None]) == {"type": ["number", "null"]}
    assert kind(list[str] | None) == {
        "type": ["array", "null"], "items": {"type": "string"}}
    assert kind(Any | None) == {"anyOf": [{}, {"type": "null"}]}   # no single type
    assert kind(int | str) == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert kind(int | str | None) == {"anyOf": [
        {"type": "integer"}, {"type": "string"}, {"type": "null"}]}


def test_a_literal_is_an_enum_of_its_values() -> None:
    assert kind(Literal["x", "y"]) == {"enum": ["x", "y"], "type": "string"}
    assert kind(Literal[1, 2]) == {"enum": [1, 2], "type": "integer"}
    assert kind(Literal["x", 1]) == {"enum": ["x", 1]}   # mixed: no shared type
    assert kind(Literal["x"] | None) == {"enum": ["x"], "type": ["string", "null"]}


def test_an_enum_class_is_its_values() -> None:
    assert kind(Colour) == {"enum": ["red", "blue"]}
    assert kind(list[Colour]) == {
        "type": "array", "items": {"enum": ["red", "blue"]}}


def test_a_dataclass_is_an_object_with_properties() -> None:
    assert kind(Point) == {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer", "description": "how far down"},
            "label": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["x", "y"],                      # the two without a default
    }


def test_a_typeddict_reads_its_required_keys() -> None:
    assert kind(Ticket) == {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "seat": {"type": "string"}},
        "required": ["id", "seat"],
    }
    assert kind(Card) == {                           # total=False: none required
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "shown to the reader"},
            "lines": {"type": "array", "items": {"type": "integer"}},
        },
        "required": [],
    }


def test_a_note_at_any_depth_reaches_the_model() -> None:
    assert kind(Annotated[str, "a word"]) == {
        "type": "string", "description": "a word"}
    assert kind(list[Annotated[int, "a count"]]) == {
        "type": "array", "items": {"type": "integer", "description": "a count"}}
    assert kind(Annotated[str, "a word"] | None) == {
        "type": ["string", "null"], "description": "a word"}


def test_the_signature_still_skips_run_and_reads_defaults() -> None:
    @tool
    def plan(run, steps: list[str], colour: Colour | None = None) -> str:
        """Plan a walk."""
        return f"{steps}{colour}"

    assert plan.parameters == {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "string"}},
            "colour": {"anyOf": [{"enum": ["red", "blue"]}, {"type": "null"}]},
        },
        "required": ["steps"],                       # run is the loop's, not the model's
    }


if __name__ == "__main__":
    for test in (
        test_leaves_come_from_the_table,
        test_a_sequence_carries_its_items,
        test_a_mapping_is_a_bare_object,
        test_none_is_admitted_beside_the_type,
        test_a_literal_is_an_enum_of_its_values,
        test_an_enum_class_is_its_values,
        test_a_dataclass_is_an_object_with_properties,
        test_a_typeddict_reads_its_required_keys,
        test_a_note_at_any_depth_reaches_the_model,
        test_the_signature_still_skips_run_and_reads_defaults,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_schema: all ok")
