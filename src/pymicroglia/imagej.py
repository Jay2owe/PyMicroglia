r"""PyMicroglia's own way to reach Fiji, for the one step a person must do.

Some work genuinely needs Fiji: drawing an outline round the SCN by hand, or
looking at an overlay and saying whether it is right. That is the settled
position — Fiji for manual judgement, Python for everything measured. This
module is the narrow door between the two, and nothing else in the package
depends on it: every analysis action runs with no Fiji present at all.

**It attaches; it never launches.** ImageJAI's own guidance is never to start a
second Fiji beside a live one, and a package that silently started a desktop
application in the middle of a six-hour pipeline would be worse than one that
asks. When nothing is listening, :func:`available` says so and the error names
the launcher to run.

**It speaks ImageJAI's protocol by importing ImageJAI, not by reimplementing
it.** That transport does session negotiation, a shared installation token and
capability exchange; a second copy of it here would drift within a month. The
import is soft in exactly the way ``_optional.kit`` is — reached through
:func:`bridge`, cached, and returning ``None`` rather than raising.

Nothing here reads, runs or references anything in ``Protocols/Analysis``. The
Jython this module sends is written below, in this file, and no ``.ijm`` is
opened.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ENTRY",
    "ImageJUnavailable",
    "ImageJTimeout",
    "available",
    "bridge",
    "agent_dir",
    "status",
    "run_jython",
    "draw_roi",
    "read_outlines",
    "run_action",
]

#: What a run started through this bridge records as its ``entry_path``. The
#: field was reserved in stage 12 so the index never had to be migrated; this is
#: the front door that fills it.
ENTRY = "imagej"

#: Environment override naming the folder that holds ImageJAI's ``ij.py``.
AGENT_ENV = "PYMICROGLIA_IMAGEJAI"

#: How long to wait for somebody to draw an outline. Generous on purpose: a
#: person working on a large stack takes minutes, and timing out early would
#: throw away the one thing in this package a machine cannot redo.
DRAW_TIMEOUT_S = 600.0

#: How often to ask Fiji whether an outline has appeared yet.
POLL_S = 2.0

#: ImageJ roi types this package can read back. The parametric ones — rectangle,
#: oval, line — are refused rather than approximated, the same rule ``roi.py``
#: applies to a stored ``.roi`` file.
READABLE_TYPES = (0, 2, 3, 4, 7, 8)

_BRIDGE: Any = None
_TRIED = False


class ImageJUnavailable(RuntimeError):
    """Fiji is not reachable, and the thing being asked for needs it."""


class ImageJTimeout(TimeoutError):
    """Nobody drew an outline in time. Nothing was recorded."""


# --------------------------------------------------------------- finding Fiji
def agent_dir() -> Path | None:
    """The folder holding ImageJAI's ``ij.py``, or ``None``.

    ``PYMICROGLIA_IMAGEJAI`` first, so a different checkout or another machine
    is one variable away. The fallback walks up for the ``Experiments`` folder
    this package lives under and looks for ``ImageJAI/agent`` beside it — a
    guess about one lab's layout, which is why it is a fallback, is reported by
    ``doctor``, and is never required for anything to run.
    """
    override = os.environ.get(AGENT_ENV, "").strip()
    if override:
        candidate = Path(override)
        candidate = candidate if candidate.is_dir() else candidate.parent
        return candidate if (candidate / "ij.py").is_file() else None

    for parent in Path(__file__).resolve().parents:
        if parent.name.lower() != "experiments":
            continue
        candidate = parent / "ImageJAI" / "agent"
        if (candidate / "ij.py").is_file():
            return candidate
    return None


def bridge() -> Any:
    """ImageJAI's transport module, or ``None`` if it is not reachable.

    Loaded from its file rather than imported by name: ImageJAI is a checkout,
    not an installed package, and putting its folder on ``sys.path`` would drag
    three thousand lines of unrelated modules into every import of this one.

    Cached after the first attempt, including the failure, so a machine without
    Fiji pays one lookup per process rather than one per call.
    """
    global _BRIDGE, _TRIED
    if _TRIED:
        return _BRIDGE
    _TRIED = True

    folder = agent_dir()
    if folder is None:
        return None
    module_name = "pymicroglia._imagejai_bridge"
    if module_name in sys.modules:
        _BRIDGE = sys.modules[module_name]
        return _BRIDGE
    try:
        spec = importlib.util.spec_from_file_location(module_name, folder / "ij.py")
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        # Any failure here means no bridge, and no bridge means the manual step
        # is unavailable. It never means an analysis cannot run.
        sys.modules.pop(module_name, None)
        return None
    _BRIDGE = module
    return _BRIDGE


def reset_cache() -> None:
    """Forget whether the bridge was reachable. For tests only."""
    global _BRIDGE, _TRIED
    _BRIDGE, _TRIED = None, False


def available() -> bool:
    """Whether a Fiji this package can talk to is answering. Never launches."""
    return bool(status().get("ok"))


def status() -> dict[str, Any]:
    """What was looked for, what answered, and what to do if nothing did.

    Shaped for ``doctor``: an absent bridge is checkable rather than silent, the
    same discipline the audit layer gets.
    """
    folder = agent_dir()
    report: dict[str, Any] = {
        "ok": False,
        "agent": str(folder) if folder else None,
        "needed_for": "drawing an outline by hand. Nothing else in this "
                      "package uses Fiji.",
    }
    module = bridge()
    if module is None:
        report["reason"] = (
            f"ImageJAI was not found. Set {AGENT_ENV} to the folder holding "
            "its ij.py."
        )
        return report

    report["host"] = getattr(module, "HOST", "localhost")
    report["port"] = getattr(module, "PORT", None)
    try:
        answer = module.ping()
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return report
    if isinstance(answer, Mapping) and answer.get("ok"):
        report["ok"] = True
        return report
    report["reason"] = (
        "Fiji is not answering on that port. Start it, then open "
        "Plugins > AI Assistant and enable the TCP command server."
    )
    return report


def _reachable() -> Any:
    module = bridge()
    if module is None or not available():
        raise ImageJUnavailable(
            "Fiji is not reachable. This is only needed for drawing an ROI by "
            "hand; every analysis step runs without it. pymicroglia doctor "
            "reports what was looked for."
        )
    return module


# ------------------------------------------------------------- running Jython
def _preamble(args: Mapping[str, Any]) -> str:
    """Arguments as literal assignments at the top of the snippet.

    Injected as ``repr`` rather than interpolated into the body, so a Windows
    path with a backslash in it cannot become an escape sequence inside the
    script — the same trap the equivalent-script builder guards against.
    """
    lines = []
    for name, value in args.items():
        if not str(name).isidentifier():
            raise ValueError(f"{name!r} is not a usable Jython variable name")
        if isinstance(value, Path):
            value = str(value)
        lines.append(f"{name} = {value!r}")
    return "\n".join(lines) + ("\n\n" if lines else "")


def run_jython(script: str, *, timeout_s: float = 180.0, **args: Any) -> dict[str, Any]:
    """Run a snippet that ships inside this package, inside the open Fiji.

    No macro is read from disk and nothing in ``Protocols/Analysis`` is
    referenced: the only Jython this package sends is the text below in this
    module, plus whatever a caller passes here.
    """
    module = _reachable()
    return dict(module.run_jython(_preamble(args) + script,
                                  timeout=int(timeout_s)))


#: Reads whatever outline is on the image, or in the ROI Manager, and writes it
#: to a file as JSON. A file rather than the script's return value on purpose:
#: the transport's result shape is ImageJAI's to change, and a JSON file this
#: package wrote and reads is a contract with itself.
READ_OUTLINES = r'''
import json

from ij import WindowManager
from ij.plugin.frame import RoiManager


def _polygon(roi):
    shape = roi.getFloatPolygon()
    return {
        "name": roi.getName() or "roi",
        "type": int(roi.getType()),
        "x": [float(shape.xpoints[i]) for i in range(shape.npoints)],
        "y": [float(shape.ypoints[i]) for i in range(shape.npoints)],
    }


def _drawn():
    manager = RoiManager.getInstance()
    if manager is not None and manager.getCount() > 0:
        return list(manager.getRoisAsArray())
    image = WindowManager.getCurrentImage()
    live = image.getRoi() if image is not None else None
    return [live] if live is not None else []


handle = open(OUTPUT, "w")
try:
    json.dump({"rois": [_polygon(roi) for roi in _drawn()]}, handle)
finally:
    handle.close()
'''

#: Says what is wanted, in the status bar. Deliberately not a dialog: a modal
#: dialog blocks the command server thread, and then nothing can be asked of
#: Fiji again until somebody dismisses it by hand.
ANNOUNCE = r'''
from ij import IJ

IJ.showStatus(MESSAGE)
IJ.log(MESSAGE)
'''


def read_outlines(*, timeout_s: float = 60.0) -> list[dict[str, Any]]:
    """Every outline currently drawn in Fiji, as plain coordinate lists."""
    handle, path = tempfile.mkstemp(prefix="pymicroglia_roi_", suffix=".json")
    os.close(handle)
    target = Path(path)
    try:
        run_jython(READ_OUTLINES, timeout_s=timeout_s, OUTPUT=str(target))
        if not target.is_file() or not target.stat().st_size:
            return []
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    finally:
        try:
            target.unlink()
        except OSError:
            pass
    found = payload.get("rois") if isinstance(payload, Mapping) else None
    return [item for item in (found or []) if isinstance(item, dict)]


# ------------------------------------------------------------ the manual step
def _as_polygon(entry: Mapping[str, Any]):
    from .roi import Polygon

    kind = int(entry.get("type", 0))
    if kind not in READABLE_TYPES:
        raise ValueError(
            f"the outline is an ImageJ type {kind}, which has no vertices to "
            "read. Draw it with the polygon, freehand or wand tool."
        )
    xs, ys = list(entry.get("x") or ()), list(entry.get("y") or ())
    if len(xs) < 3 or len(xs) != len(ys):
        raise ValueError(f"an outline needs at least three vertices, got {len(xs)}")
    return Polygon(name=str(entry.get("name") or "roi"), x=xs, y=ys)


def _frame_to_file(series, frame: int, channel: int | None) -> Path:
    """One frame written where Fiji can open it.

    A single plane, not the stack: the person is drawing round the tissue, and
    handing Fiji ten gigabytes to open so somebody can trace one outline would
    cost minutes for nothing.
    """
    import numpy as np

    from . import io

    picked = 0 if channel is None else int(channel)
    plane = np.asarray(series.frame(int(frame), picked))
    folder = Path(tempfile.mkdtemp(prefix="pymicroglia_draw_"))
    name = Path(getattr(series, "path", "frame")).stem or "frame"
    target = folder / f"{name}_t{int(frame)}_c{picked}.tif"
    io.write_tiff(plane, target)
    return target


def draw_roi(source, *, series=None, frame: int = 0, channel: int | None = None,
             name: str = "hand", decision: str = "",
             timeout_s: float = DRAW_TIMEOUT_S, poll_s: float = POLL_S,
             force: bool = False):
    """Open one frame in Fiji, wait for an outline to be drawn, return it.

    The only interactive entry point in the package; everything else is
    headless. The polygon is stored as a **decision** keyed on the source alone,
    so it survives a parameter change and a ``METHOD_VERSION`` bump — a person
    answered this once, and a new engine version is not a reason to ask them
    again. Pass ``force=True`` to draw a replacement deliberately.

    ``decision`` names the store key, so a caller that already owns one -
    ``roi.draw_scn_roi`` owns ``scn_roi`` - records under it rather than
    beside it. Two keys for one question would mean an unattended run asking
    again for something already answered.

    Raises :class:`ImageJTimeout` rather than recording a half-answer. A timeout
    here must lose nothing: the decision is written only after a complete
    outline has been read back.
    """
    from . import store
    from .roi import Polygon

    decision = decision or f"hand_roi_{name}"
    if not force:
        stored = store.decision(decision, source)
        if stored is not None:
            return Polygon(name=str(stored.get("name") or name),
                           x=list(stored.get("x") or ()),
                           y=list(stored.get("y") or ()))

    module = _reachable()
    if series is None:
        from .series import open_series

        series = open_series(source)
    plane = _frame_to_file(series, frame, channel)

    module.open_image(str(plane))
    message = (f"PyMicroglia: draw the {name} outline on this frame with the "
               "polygon or freehand tool, then press T to add it.")
    run_jython(ANNOUNCE, MESSAGE=message)

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        drawn = read_outlines()
        if drawn:
            polygon = _as_polygon(drawn[-1])
            store.decision(
                decision, source,
                value={"name": polygon.name, "x": list(polygon.x),
                       "y": list(polygon.y), "frame": int(frame),
                       "channel": None if channel is None else int(channel)},
                note=f"drawn by hand in Fiji on frame {int(frame)}")
            return polygon
        time.sleep(float(poll_s))

    raise ImageJTimeout(
        f"no outline appeared within {int(timeout_s)} s, so nothing was "
        f"recorded. The frame is still open at {plane}; draw the outline and "
        "call draw_roi again, and it will be picked up."
    )


# ------------------------------------------------------------ starting a run
def run_action(action: str, **kwargs: Any) -> Any:
    """Run a registered action and record it as having come from Fiji.

    The one difference from :func:`pymicroglia.run_action` is the ``entry_path``
    in the record. Everything else about the run — the resolved parameters, the
    artefacts, the equivalent script — is identical, which is the point: a run
    started from the bench and a run started from a terminal must be comparable
    without anybody having to know which was which.
    """
    from .run import run_action as _run

    kwargs.setdefault("entry", ENTRY)
    return _run(action, **kwargs)
