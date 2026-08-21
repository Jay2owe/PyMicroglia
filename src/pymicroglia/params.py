"""Read a script's ``PROTOCOL PARAMETERS`` block into structured parameters.

Every protocol in this project puts its whole adjustable surface in one block at
the top of the file, with a comment on each line saying what the setting means.
That block is the parameter schema; this module turns it into ``ParamDoc``
entries so an agent can ask ``describe`` instead of reading source.

Three rules the implementation follows, each for a reason:

**Nothing is imported or executed.** Importing ``dluc_pipeline.py`` would pull
numpy, tifffile, scipy, scikit-image and matplotlib just to read a comment.
Blocks are located and enumerated by line scanning; for Python files the
abstract syntax tree is parsed *as data* to resolve literal values, which still
executes nothing.

**Enumeration matches ``protocol-that``'s ``register.py`` exactly.** That script
generates ``Protocols/INDEX.md``, so if this module counted parameters
differently the two would disagree about what a protocol has. The block markers,
the assignment pattern and the comment pattern below are its rules, deliberately.

**Three languages, one rule.** Seven of the thirteen protocols take parameters
from an ImageJ macro and two from PowerShell, so a Python-only reader would miss
a quarter of the schema.

This module reads files it is given. It hard-codes no path, requires no
particular folder to exist, and depends on nothing in ``Protocols/Analysis``.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ._optional import kit

__all__ = [
    "ParamDoc",
    "ParamBlock",
    "harvest",
    "harvest_many",
    "parameter_name",
    "infer_units",
    "UNIT_SUFFIXES",
]


# ── the block grammar, kept identical to protocol-that's register.py ──────────
START_RE = re.compile(r"PROTOCOL\s+PARAMETERS", re.I)
END_RE = re.compile(r"END\s+PROTOCOL\s+PARAMETERS", re.I)
ASSIGN_RE = re.compile(
    r"^\s*(?:var\s+|const\s+|let\s+|final\s+)?\$?([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|<-)\s*\S"
)
COMMENT_LINE_RE = re.compile(r"^\s*(#|//|\*|%|;|--|/\*|REM\b)")
INLINE_COMMENT_RE = re.compile(r"\s(#|//|%|--)\s")
SEPARATOR_RE = re.compile(r"^\s*(?:#|//|\*|%|;|--)+\s*[=\-_~]{3,}")
METHOD_VERSION_RE = re.compile(r"^\s*(?:var\s+)?METHOD_VERSION\s*=\s*(.+?)\s*;?\s*$")

LANGUAGES = {".py": "python", ".ijm": "ijm", ".ps1": "powershell"}
COMMENT_TOKENS = {"python": ("#",), "powershell": ("#",), "ijm": ("//",)}

#: Suffix on a constant's name -> the unit it is measured in. The project names
#: its parameters consistently enough that this is reliable; anything unmatched
#: stays "-", which is ParamDoc's own default for "no unit".
UNIT_SUFFIXES: dict[str, str] = {
    "_PX": "px",
    "_PCT": "%",
    "_PERCENT": "%",
    "_PERCENTILE": "%",
    "_PERCENTILES": "%",
    "_H": "h",
    "_HOURS": "h",
    "_MINUTES": "min",
    "_SECONDS": "s",
    "_MS": "ms",
    "_SIGMA": "sigma",
    "_HZ": "Hz",
    "_UM": "um",
    "_BYTES": "B",
    "_FRAMES": "frames",
    "_LEVEL": "level",
    "_COUNTS": "counts",
    "_FPS": "fps",
    "_DEG": "deg",
}


# ── ParamDoc: the kit's when it is installed, an identical local one when not ─
_kit_module = kit()
if _kit_module is not None and hasattr(_kit_module, "ParamDoc"):
    ParamDoc = _kit_module.ParamDoc  # type: ignore[assignment]
else:  # pragma: no cover - exercised by test_optional_import

    @dataclass(frozen=True)
    class ParamDoc:  # type: ignore[no-redef]
        """Local stand-in with the same six fields as ``analysis_kit.ParamDoc``.

        Defined so this module works with the kit absent. The field names and
        order match the kit's exactly, so a consumer cannot tell them apart.
        """

        name: str
        type: str
        description: str
        required: bool = False
        default: Any = None
        units: str = "-"

        def as_dict(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "type": self.type,
                "units": self.units,
                "required": self.required,
                "default": self.default,
                "description": self.description,
            }


@dataclass(frozen=True)
class ParamBlock:
    """One script's parameter block, as read."""

    source: Path
    language: str
    method_version: str
    params: tuple[ParamDoc, ...] = ()
    #: Constant names in file order, e.g. ``DEFAULT_THRESHOLD_SIGMA``. The
    #: ``ParamDoc`` name is the callable form; this is what INDEX.md lists.
    constants: tuple[str, ...] = ()
    lines: tuple[int, ...] = field(default=(), repr=False)

    def __len__(self) -> int:
        return len(self.params)

    def by_constant(self, constant: str) -> ParamDoc | None:
        for name, doc in zip(self.constants, self.params):
            if name == constant:
                return doc
        return None


# ── naming and units ─────────────────────────────────────────────────────────
def parameter_name(constant: str) -> str:
    """``DEFAULT_THRESHOLD_SIGMA`` -> ``threshold_sigma``.

    Exactly the mapping the engines already use between a module constant and
    the field of its ``Settings`` dataclass, so a harvested name matches the
    keyword a caller would pass. Constants without the prefix keep their whole
    name, lower-cased.
    """
    stripped = constant[len("DEFAULT_"):] if constant.startswith("DEFAULT_") else constant
    return stripped.lower()


def infer_units(constant: str) -> str:
    """The unit implied by a constant's name suffix, or ``"-"``."""
    upper = constant.upper()
    for suffix, unit in sorted(UNIT_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if upper.endswith(suffix):
            return unit
    return "-"


# ── low-level text handling ──────────────────────────────────────────────────
def _language(path: Path) -> str:
    return LANGUAGES.get(path.suffix.lower(), "python")


def _split_inline_comment(text: str, language: str) -> tuple[str, str]:
    """Split a line into (code, comment), respecting quotes.

    A naive ``split("#")`` breaks on ``DEFAULT_VIDEO_LUT = "#a340d1"``, which is
    exactly the kind of value this project uses.
    """
    tokens = COMMENT_TOKENS.get(language, ("#",))
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        for token in tokens:
            if text.startswith(token, i):
                return text[:i], text[i + len(token):]
        i += 1
    return text, ""


def _value_text(line: str, language: str) -> str:
    """The right-hand side of one assignment, without its comment.

    A trailing comma is dropped. Several protocols keep their block inside a
    ``dict(...)`` call — ``dluc_pipeline.py``'s thirty-nine settings are one
    call — where the comma separates entries and is not part of the value.
    Left in, ``GAP_H=1.0,`` reads as the one-element tuple ``(1.0,)``, and
    ``describe`` then tells an agent the type is ``tuple`` and the default is a
    sequence. It is separator syntax, so it goes.
    """
    code, _ = _split_inline_comment(line, language)
    _, _, rhs = code.partition("=")
    return rhs.strip().rstrip(";").strip().rstrip(",").strip()


def _is_container_opening(text: str) -> bool:
    """Whether this right-hand side only *opens* something.

    ``P = dict(`` matches the assignment grammar as well as any setting does,
    but it is the container the settings live in, not a setting. A value that
    leaves a bracket open is not complete on its line, and every real parameter
    here is.
    """
    if not text:
        return False
    depth = sum(text.count(o) - text.count(c)
                for o, c in (("(", ")"), ("[", "]"), ("{", "}")))
    return depth > 0


def _clean_comment(text: str) -> str:
    return re.sub(r"^\s*(?:#+|//+|\*+|%+|;+|--+|/\*|\*/|REM\b)\s?", "", text).strip()


def _is_comment(line: str) -> bool:
    return bool(COMMENT_LINE_RE.match(line))


# ── description assembly ─────────────────────────────────────────────────────
def _indented_comment(line: str) -> bool:
    """A comment that begins past column 0.

    The blocks in this project align run-on prose to a comment column on the
    right, while a description written *above* an assignment starts hard left::

        DEFAULT_VIDEO_LUT = "#a340d1"    # A hex value, to prove the split
                                         # is quote-aware.          <- indented
        # The description for this one sits above it.                <- column 0
        DEFAULT_UNDERLAY_GREY = "0.72"

    Indentation is therefore what separates "still talking about the parameter
    above" from "about to describe the one below", and it is the only signal
    available: both are plain comment lines otherwise.
    """
    return _is_comment(line) and not line.startswith(("#", "//", "/*", "*", ";", "%", "--"))


def _describe_block(lines, assignments, start, end, language):
    """Map each assignment's line index to its prose, in one pass.

    Three layouts are in use, and all three appear in the real blocks:

    1. a trailing comment, running on over indented lines beneath it;
    2. no trailing comment because the name is too long to leave room, so the
       prose starts on the next line at the comment column;
    3. the description written on the line above, at column 0.

    The forward pass records which comment lines it consumed so the upward
    fallback cannot claim prose that already belongs to the parameter above.

    Some prose is genuinely shared between two parameters — a sentence begun
    under one and finished under the next. Each gets the part written under it;
    no attempt is made to guess the split, because the source does not say.
    """
    consumed: set[int] = set()
    described: dict[int, list[str]] = {}

    for i in assignments:
        _, inline = _split_inline_comment(lines[i], language)
        parts = [inline.strip()] if inline.strip() else []
        j = i + 1
        while (j < end and lines[j].strip() and _indented_comment(lines[j])
               and not SEPARATOR_RE.match(lines[j])):
            parts.append(_clean_comment(lines[j]))
            consumed.add(j)
            j += 1
        described[i] = [part for part in parts if part]

    for i in assignments:
        if described[i]:
            continue
        above: list[str] = []
        for j in range(i - 1, start, -1):
            line = lines[j]
            if (j in consumed or not line.strip() or not _is_comment(line)
                    or SEPARATOR_RE.match(line)):
                break
            above.append(_clean_comment(line))
        described[i] = [part for part in reversed(above) if part]

    return {i: " ".join(parts).strip() for i, parts in described.items()}


# ── value parsing ────────────────────────────────────────────────────────────
def _python_literals(source: str, path: Path) -> dict[int, Any]:
    """Literal values of module-level assignments, keyed by line number.

    ``ast.literal_eval`` evaluates nothing but literals, so a malicious or
    merely expensive module cannot do anything here. Assignments whose value is
    not a literal are simply absent from the map and fall back to text.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return {}
    out: dict[int, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            out[node.lineno] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return out


_TYPE_NAMES = {
    bool: "bool", int: "int", float: "float", str: "str",
    list: "list", tuple: "tuple", dict: "dict", set: "set", type(None): "none",
}


def _typed(value: Any) -> str:
    return _TYPE_NAMES.get(type(value), type(value).__name__)


def _parse_text_value(text: str) -> tuple[str, Any]:
    """Best-effort type and value for a non-Python right-hand side."""
    if not text:
        return "unknown", None
    lowered = text.strip().lower()
    if lowered in {"true", "false", "$true", "$false"}:
        return "bool", lowered.endswith("true")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            value = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return "str", text.strip("\"'")
    return _typed(value), value


# ── the public reader ────────────────────────────────────────────────────────
def _find_block(lines: Sequence[str]) -> tuple[int, int] | None:
    end = next((i for i, line in enumerate(lines) if END_RE.search(line)), None)
    if end is None:
        return None
    start = next(
        (i for i in range(end) if START_RE.search(lines[i]) and not END_RE.search(lines[i])),
        None,
    )
    return None if start is None else (start, end)


def _method_version(lines: Sequence[str]) -> str:
    for line in lines:
        match = METHOD_VERSION_RE.match(line)
        if match:
            text = match.group(1).strip()
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text.strip("\"'")
            return str(parsed)
    return ""


def harvest(path: str | Path) -> ParamBlock | None:
    """Read one script's parameter block. ``None`` when it has no block.

    Never imports, executes or writes anything.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    language = _language(path)
    method_version = _method_version(lines)

    found = _find_block(lines)
    if found is None:
        return None
    start, end = found

    literals = _python_literals(source, path) if language == "python" else {}

    assignments = [
        i for i in range(start + 1, end)
        if lines[i].strip() and not _is_comment(lines[i])
        and ASSIGN_RE.match(lines[i])
        and not _is_container_opening(_value_text(lines[i], language))
    ]
    prose = _describe_block(lines, assignments, start, end, language)

    docs: list[ParamDoc] = []
    constants: list[str] = []
    numbers: list[int] = []
    for i in assignments:
        line = lines[i]
        constant = ASSIGN_RE.match(line).group(1)

        if i + 1 in literals:
            value = literals[i + 1]
            type_name = _typed(value)
        else:
            type_name, value = _parse_text_value(_value_text(line, language))

        docs.append(
            ParamDoc(
                name=parameter_name(constant),
                type=type_name,
                description=prose[i],
                required=False,
                default=value,
                units=infer_units(constant),
            )
        )
        constants.append(constant)
        numbers.append(i + 1)

    return ParamBlock(
        source=path,
        language=language,
        method_version=method_version,
        params=tuple(docs),
        constants=tuple(constants),
        lines=tuple(numbers),
    )


def harvest_many(paths: Iterable[str | Path]) -> ParamBlock:
    """Harvest several files that make up one protocol, de-duplicated.

    A protocol is often a Python engine plus an ImageJ wrapper, or a script plus
    its PowerShell runner, and a parameter may appear in more than one. First
    occurrence wins, matching how ``register.py`` builds INDEX.md, so the count
    here equals the count there.

    ``method_version`` is taken from the first file that declares one.
    """
    docs: list[ParamDoc] = []
    constants: list[str] = []
    numbers: list[int] = []
    method_version = ""
    sources: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        block = harvest(path)
        if block is None:
            continue
        sources.append(block.source)
        if not method_version and block.method_version:
            method_version = block.method_version
        for constant, doc, line in zip(block.constants, block.params, block.lines):
            if constant in seen:
                continue
            seen.add(constant)
            constants.append(constant)
            docs.append(doc)
            numbers.append(line)

    return ParamBlock(
        source=sources[0] if sources else Path(),
        language="mixed" if len({p.suffix for p in sources}) > 1 else (
            _language(sources[0]) if sources else "python"),
        method_version=method_version,
        params=tuple(docs),
        constants=tuple(constants),
        lines=tuple(numbers),
    )
