"""What an agent may ask PyMicroglia to do.

Actions bind to their implementation by **dotted name**, not by callable, so the
registry imports before the science modules exist — the same reason PyFLASH's
plot registry holds strings. An action whose target cannot be resolved is
reported as *pending*: thirteen of them are today, and that number falls to zero
as the stages land. A half-ported package is then a visible state rather than a
surprise.

The registry is the kit's when ``analysis_kit`` is installed and a small local
stand-in when it is not, so ``describe`` answers either way.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from . import catalogue
from ._optional import kit

__all__ = [
    "REGISTRY",
    "MODULES",
    "pending",
    "resolve",
    "build_registry",
    "CLAIM_TEMPLATES",
    "NEEDS_A_CLAIM",
    "ClaimRequired",
    "claim_for",
    "require_claim",
]

#: Dotted prefix -> the module that provides it. Filled in as each stage lands;
#: an absent entry is what makes its actions pending.
MODULE_NAMES: tuple[str, ...] = (
    "registration",
    "filtering",
    "cosmic",
    "display",
    "segmentation",
    "roi",
    "tracing",
    "controls",
    "rhythm",
    # The trace panel binds here, not under ``visualisation``: the drawing half
    # takes a finished table, and the half that builds one has to be allowed to
    # compute.
    "trace_tables",
    "visualisation.qc",
    "visualisation.overlays",
    "video",
    "pipelines.dluc_single_cell",
    "pipelines.cry1_dluc_photon",
)


def _load_modules() -> dict[str, ModuleType]:
    """Import whichever science modules exist. A missing one is not an error.

    ``ImportError`` here means "that stage has not landed yet", which is the
    normal state during the build-out. Anything else is a real fault in a module
    that does exist, and is allowed to propagate.
    """
    found: dict[str, ModuleType] = {}
    for name in MODULE_NAMES:
        try:
            found[name] = importlib.import_module(f"pymicroglia.{name}")
        except ImportError:
            continue
    return found


MODULES: dict[str, ModuleType] = _load_modules()

FIX_HINT = (
    "Actions bind by dotted name to modules that arrive stage by stage. "
    "Pending actions are expected until their stage lands; regenerate the "
    "catalogue with tools/regenerate_catalogue.py."
)


class _LocalRegistry:
    """Enough of the kit's Registry to answer describe/discover without it.

    Deliberately small: it exists so a missing audit layer costs a run record
    and nothing else. Anything richer belongs in the kit, not here.
    """

    def __init__(self, project: str, version: str, modules: dict[str, ModuleType]):
        self.project = project
        self.version = version
        self.modules = modules
        self.actions = {entry["name"]: entry for entry in catalogue.actions()}
        self.params = {doc.name: doc for doc in catalogue.param_docs()}
        self.reference_dir = None
        self.fix_hint = FIX_HINT

    def names(self) -> list[str]:
        return sorted(self.actions)

    def __contains__(self, name: object) -> bool:
        return name in self.actions

    def spec(self, name: str) -> dict[str, Any]:
        return self.actions[name]

    def binds_to(self, name: str) -> str:
        return self.actions[name]["method"]

    def action_params(self, name: str) -> list[dict[str, Any]]:
        return catalogue.action_params(name)

    def resolve(self, name: str):
        return resolve(self.actions[name]["method"], self.modules)

    def undeclared_params(self) -> list[str]:
        declared = set(self.params)
        used = {p for entry in self.actions.values() for p in entry["params"]}
        return sorted(used - declared)


def resolve(method: str, modules: dict[str, ModuleType] | None = None):
    """The callable behind a dotted target, or ``None`` while it is pending."""
    module_name, _, attribute = method.rpartition(".")
    module = (modules if modules is not None else MODULES).get(module_name)
    return getattr(module, attribute, None) if module is not None else None


def live_defaults(action: str) -> dict[str, Any]:
    """The default each parameter of one action actually gets.

    The catalogue records what the protocol script this was copied from
    declared; the function records what runs. They agree almost everywhere and
    disagree in sixty-five places — ``run_controls`` searches 16-32 h where the
    engine searched 15-40, and one engine default is the *string* ``"3.0 / 8.0"``.

    ``describe`` is read by an agent immediately before it passes an argument,
    so it has to report what will happen. The catalogue stays the source for
    names, types, units and prose; only the number comes from the code.
    """
    from .recording import _signature_defaults

    merged = dict((catalogue.action(action) or {}).get("defaults") or {})
    merged.update(_signature_defaults(action))
    return merged


def _project_registry(ak):
    """The kit's Registry, taught this project's per-action defaults.

    The kit's shared vocabulary deliberately carries none: one parameter name
    may honestly mean different values in different actions, and a shared entry
    holding one would misreport every other. PyMicroglia has both halves, so it
    merges them at the single accessor ``describe``, ``discover`` and the
    catalogue generator all read — rather than in three places that would drift.
    """

    class ProjectRegistry(ak.Registry):
        def action_params(self, action: str) -> list[dict[str, Any]]:
            rows = super().action_params(action)
            defaults = live_defaults(action)
            for row in rows:
                if row["name"] in defaults:
                    row["default"] = defaults[row["name"]]
            return rows

    return ProjectRegistry


def build_registry(reference_dir=None):
    """The kit's Registry when it is installed, otherwise the local stand-in.

    ``reference_dir`` is where the generated action catalogue lives. Only the
    skill's runner passes one — it is what lets ``discover`` report an action
    nobody documented, and the package itself has no docs folder to point at.
    """
    from . import __version__

    ak = kit()
    if ak is None:
        return _LocalRegistry("pymicroglia", __version__, MODULES)

    registry = _project_registry(ak)(
        "pymicroglia",
        version=__version__,
        params=list(catalogue.param_docs()),
        modules=MODULES,
        reference_dir=reference_dir,
        fix_hint=FIX_HINT,
    )
    covered = _covered_functions()
    for entry in catalogue.actions():
        registry.add(entry["name"], ak.ActionSpec(
            summary=entry["summary"],
            method=entry["method"],
            mutates=bool(entry.get("mutates")),
            destructive=bool(entry.get("destructive")),
            params=tuple(entry["params"]),
            covers=covered.get(entry["name"], ()),
        ))
    return registry


def _covered_functions() -> dict[str, tuple[str, ...]]:
    """Which public functions each action is the agent-facing route to.

    Computed from the bindings rather than typed out, so it cannot rot: a module
    is covered by the actions bound to it, and every public function in that
    module is a step inside one of them. ``segmentation.somata_by_prominence``
    is not a thing to call — it is part of what ``segment`` does.

    This is what stops ``discover`` reporting a hundred helpers as "a public
    function nobody exposed". That check earns its keep on a project whose
    action layer is behind its backend; here the twenty-six actions *are* the
    surface, and the catalogue is generated from the protocols they were copied
    from. What stays live is the check that matters — an action bound to a name
    that has since moved.
    """
    by_module: dict[str, list[str]] = {}
    for entry in catalogue.actions():
        module_name = str(entry["method"]).rpartition(".")[0]
        by_module.setdefault(module_name, []).append(entry["name"])

    covered: dict[str, tuple[str, ...]] = {}
    for module_name, actions in by_module.items():
        module = MODULES.get(module_name)
        if module is None:
            continue
        # Functions only. A dataclass in the same module is a return type, not
        # something an action could be the route to, and claiming to cover one
        # makes the kit look for a function by that name and not find it.
        public = sorted(
            name for name in dir(module)
            if not name.startswith("_")
            and inspect.isfunction(getattr(module, name, None))
            and getattr(getattr(module, name), "__module__", "")
            == module.__name__
        )
        # All of them to the first action alphabetically. Which action carries
        # the claim does not matter to any reader; that the module is covered
        # at all does.
        covered[sorted(actions)[0]] = tuple(public)
    return covered


REGISTRY = build_registry()


def pending() -> list[str]:
    """Actions declared but not yet backed by a resolvable target."""
    return [name for name in REGISTRY.names() if REGISTRY.resolve(name) is None]


# ------------------------------------------------------------------- the claim
class ClaimRequired(ValueError):
    """This action concludes something, and only a person can say what."""


#: One sentence saying what a run was meant to show, per action. A **template**,
#: not a default: it is filled in when the run happens and it names the file, so
#: a search for a recording's name finds every run that touched it.
#:
#: These are the mechanical actions — the ones whose point is the file that comes
#: out. "Cleaned the spikes out of MCG_04" is the whole of what that run meant,
#: and making somebody retype it would turn the field into boilerplate, which is
#: the failure this is guarding against rather than the one it looks like.
CLAIM_TEMPLATES: dict[str, str] = {
    "register": "registered {source} so its frames sit still enough to measure",
    "register_three_channel":
        "registered the three channels of {source} against its phase image",
    "export_registered_stack":
        "exported {source} registered and cropped, with no unmixing",
    "remove_cosmic_rays": "cleaned the cosmic-ray spikes out of {source}",
    "remove_static_background":
        "display copy of {source} with the static background taken off",
    "display_filter":
        "display copy of {source} that keeps the background visible",
    "background": "read the off-tissue background level of {source}",
    "export_roi": "wrote {source}'s objects out as a Fiji RoiSet",
    "extract_traces":
        "took one ring-subtracted trace per object out of {source}",
    "unmix": "unmixed {source}'s signal channel from its autofluorescence",
    "red_only_video": "red-only video of {source}",
    "composite_video": "timestamped red/green composite video of {source}",
    "phase_green_red_video": "timestamped green/red video of {source}",
    "stack_to_mp4": "video of {source} at its stated experimental rate",
    "trace_panel": "trace panel drawn from {source}",
    "registration_figure": "checked the registration applied to {source}",
    "cosmic_ray_preview": "checked what cosmic-ray removal replaced in {source}",
    "channel_figure": "checked which channel is which in {source}",
    "frames_figure": "checked the first, middle and last frames of {source}",
    "cell_overlay": "checked the segmented objects against {source}'s pixels",
    "roi_overlay": "checked the stored regions against the frame they were "
                   "drawn on in {source}",
}

#: Actions that conclude something, where no template can be honest. Which
#: objects are cells, whether a trace is rhythmic, whether a decoy beat the
#: signal — a run of one of these was made to find something out, and the only
#: person who knows what is the one who started it. These refuse an empty claim
#: at the front door, where a refusal costs nothing.
#:
#: ``bioluminescence`` and ``phase_green_red`` are pipelines rather than
#: registered actions; they record runs all the same, so they are listed here.
NEEDS_A_CLAIM: frozenset[str] = frozenset({
    "segment",
    "run_controls",
    "test_rhythm",
    "dluc_single_cell",
    "cry1_dluc_photon",
    "bioluminescence",
    "phase_green_red",
})


def _source_name(params: Mapping[str, Any] | None) -> str:
    """The recording's name, as a person would say it out loud.

    ``trace_panel`` is the one action with no ``source``: it draws from finished
    trace tables. Its claim should still name one of them, because the claim is
    what the index searches and a row reading "trace panel drawn from the stack"
    is a row nobody can find.
    """
    params = params or {}
    raw = str(params.get("source") or "").strip()
    if not raw:
        tables = params.get("input_csvs") or ()
        raw = str(next(iter(tables), "") if not isinstance(tables, str)
                  else tables).strip()
    if not raw:
        return "the stack"
    name = Path(raw.replace("\\", "/")).name or raw
    lowered = name.lower()
    for suffix in (".ome.tif", ".ome.tiff"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem or name


def claim_for(action: str, claim: str = "",
              params: Mapping[str, Any] | None = None) -> tuple[str, bool]:
    """The claim to record, and whether one is still owed.

    A claim the caller wrote always wins. Otherwise a mechanical action fills in
    its template and an interpretive one records nothing and says so, rather
    than restating its own summary — a thousand rows all reading "segment cells
    by soma prominence" is a thousand rows nobody can skim.
    """
    text = str(claim or "").strip()
    if text:
        return text, False
    template = CLAIM_TEMPLATES.get(action, "")
    if not template:
        return "", True
    return template.replace("{source}", _source_name(params)), False


def require_claim(action: str, claim: str = "") -> None:
    """Refuse, before anything runs, a run that owes a claim and has none."""
    if str(claim or "").strip() or action not in NEEDS_A_CLAIM:
        return
    raise ClaimRequired(
        f"{action} needs a claim: one sentence saying what this run is meant to "
        "show. It concludes something, so no wording generated from the action "
        "itself would be true. Pass claim=\"...\", or --claim \"...\" on the "
        "command line."
    )
