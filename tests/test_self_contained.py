"""PyMicroglia depends on nothing in ``Protocols/Analysis``.

The first house rule in the plan. Those scripts and macros are sources this
package was copied from — never imported, never called, never edited. A package
whose correctness depends on files outside it is not installable or movable, and
editing the originals would destroy the reference the copies are verified
against.

Checks run against the abstract syntax tree, so prose in a docstring or comment
may name that folder freely. Only *code* is constrained.

This test grows with the package. Every later stage must keep it passing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "pymicroglia"

#: Module names of the loose engines. Importing any of them would make this
#: package depend on a folder it must be able to live without.
ENGINE_MODULES = {
    "cry1_dluc_photon_pipeline",
    "dluc_pipeline",
    "microglia_bioluminescence_display",
    "microglia_cosmic_ray_removal",
    "microglia_phase_correlation_registration",
    "microglia_raw_registered_stack_export",
    "microglia_red_only_video_export",
    "microglia_static_background_removal",
    "microglia_timestamped_composite_video_export",
    "phase_green_red_timelapse_pipeline",
    "phase_green_red_video_export",
    "tiff_stack_to_mp4",
    "trace_panel_figure",
}

FORBIDDEN_IN_STRINGS = ("Protocols/Analysis", "Protocols\\Analysis")


def source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def parsed():
    for path in source_files():
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def docstring_nodes(tree) -> set[int]:
    """ids() of every docstring Constant in a tree.

    Prose may name ``Protocols/Analysis`` — the plan says to record where each
    method was copied from, and a docstring is the right place for it. Only
    executable code is constrained, so docstrings are excluded from the string
    scan below.
    """
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            found.add(id(body[0].value))
    return found


def test_there_is_something_to_check():
    assert source_files(), f"no modules found under {SRC}"


def test_no_module_imports_an_engine():
    offenders = []
    for path, tree in parsed():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name in ENGINE_MODULES:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, offenders


def test_no_code_references_the_protocols_analysis_path():
    """A string literal naming that folder is a runtime dependency in disguise."""
    offenders = []
    for path, tree in parsed():
        prose = docstring_nodes(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in prose:
                continue
            for needle in FORBIDDEN_IN_STRINGS:
                if needle in node.value:
                    offenders.append(f"{path.name}:{node.lineno} contains {needle!r}")
    assert not offenders, offenders


def test_only_optional_imports_the_kit():
    """Every route to analysis_kit goes through _optional.kit().

    A module-scope ``import analysis_kit`` anywhere else would make the audit
    layer a hard dependency, which is exactly what must not happen.
    """
    offenders = []
    for path, tree in parsed():
        if path.name == "_optional.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            if "analysis_kit" in names:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"analysis_kit imported outside _optional.py at {offenders}"
    )


def test_nothing_shells_out():
    """No subprocess, os.system or runpy anywhere in the package.

    Shelling out to an engine would satisfy every check above while still making
    this package depend on a script it does not own.
    """
    banned_modules = {"subprocess", "runpy"}
    offenders = []
    for path, tree in parsed():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name in banned_modules:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, offenders


@pytest.mark.parametrize("attribute", ["system", "popen", "execv", "spawnv"])
def test_no_os_process_calls(attribute):
    offenders = []
    for path, tree in parsed():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == attribute
                    and isinstance(node.value, ast.Name) and node.value.id == "os"):
                offenders.append(f"{path.name}:{node.lineno} calls os.{attribute}")
    assert not offenders, offenders
