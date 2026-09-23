"""The exact plotted rows returned by one view."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Drawn:
    data: Any
    axes: Any = None
    extra: dict = field(default_factory=dict)
