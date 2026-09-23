"""Figures declare independent views and their existing measured inputs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Table:
    name: str
    optional: bool = False
    module: str = ''
    scope: str = 'movie'


@dataclass(frozen=True)
class Stack:
    name: str
    optional: bool = False


@dataclass(frozen=True)
class Input:
    name: str
    optional: bool = False


@dataclass(frozen=True)
class Option:
    name: str
    default: Any = None
    description: str = ""
    choices: tuple = ()

    def __post_init__(self):
        from ._vocabulary import meaning
        definition = meaning(self.name)
        if not self.description:
            object.__setattr__(self, 'description', definition['description'])


@dataclass(frozen=True)
class View:
    key: str
    draw: Callable
    needs: tuple[str, ...] = ()
    title: str = ""
    polar: bool = False
    block: bool = False


@dataclass(frozen=True)
class Figure:
    key: str
    title: str
    family: str
    views: tuple[View, ...]
    reads: tuple[Table | Stack | Input, ...]
    prepare: str
    options: tuple[Option, ...] = ()
    layout: str = "row"
    refits: bool = False
    claim: str = ""
    saved: bool = False
    grammar: str = "small-multiples"
    purpose: str = 'result'

    @property
    def slug(self):
        return self.key.replace("_", "-")

    @property
    def source(self):
        import importlib.util
        from pathlib import Path
        return Path(importlib.util.find_spec(self.prepare.split(":")[0]).origin)

    def view(self, key):
        for view in self.views:
            if view.key == key:
                return view
        raise ValueError(f"{self.key}: unknown view {key!r}; choose {', '.join(v.key for v in self.views)}")


FIGURES: dict[str, Figure] = {}


def figure(spec):
    if spec.key in FIGURES:
        raise ValueError(f"Figure already registered: {spec.key}")
    if not spec.views or len({v.key for v in spec.views}) != len(spec.views):
        raise ValueError("A figure needs uniquely named views")
    FIGURES[spec.key] = spec
    return spec
