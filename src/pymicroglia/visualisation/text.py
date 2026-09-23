"""Figure wording resolves explicit slots before recorded wording and defaults."""
from dataclasses import dataclass, field

SLOTS = ("title", "subtitle", "footnote", "note", "claim")


@dataclass
class Wording:
    title: str = ""
    subtitle: str = ""
    footnote: str = ""
    note: str = ""
    claim: str = ""
    sources: dict = field(default_factory=dict)


def resolve(defaults, recorded=None, explicit=None):
    values = {slot: str(defaults.get(slot, "")) for slot in SLOTS}
    sources = {slot: "default" for slot in SLOTS}
    for origin, layer in (("run", recorded or {}), ("call", explicit or {})):
        unknown = set(layer) - set(SLOTS)
        if unknown:
            raise ValueError(f"Unknown figure text slots: {sorted(unknown)}")
        for slot, value in layer.items():
            values[slot] = str(value)
            sources[slot] = origin
    return Wording(**values, sources=sources)


def figure_text(run, slug, *, argv=None, item=None, explicit=None, **defaults):
    from .._results import read_document, figure_plans
    try:
        manifest = read_document(run / "manifest.json")
    except FileNotFoundError:
        manifest = {}
    recorded = manifest.get("figures", {}).get(slug, {})
    if not isinstance(recorded, dict):
        recorded = {}
    recorded = {k:v for k,v in recorded.items() if k in SLOTS}
    recorded.update(figure_plans(run, manifest=manifest).get(item, {}).get("text", {}))
    return resolve(defaults, recorded, explicit)
