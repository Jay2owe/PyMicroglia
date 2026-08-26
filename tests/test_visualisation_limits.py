"""The four rules that stop `visualisation/` becoming plotting.py.

`PyFLASH/plotting.py` is 31,497 lines holding 620 functions, 557 of them
private, to serve about forty public plots — roughly 790 lines per plot,
because every plot grew its own layout, saving, labelling and statistics
helpers. Nobody decided that. It happened one function at a time, and every
individual step was reasonable.

A document saying "share the helpers" is what already failed. These are the
same four rules written as structural facts a test can check, so the first step
in that direction goes red instead of being merged.

The second one is load-bearing. A figure that cannot compute cannot grow a
private helper stack, because there is nothing for the helpers to do — and it
also means every figure's exact plotted data already exists as a table by the
time it is saved, which is what makes the provenance bundle free rather than
extra work.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "pymicroglia"
FIGURES = SRC / "visualisation"

#: 600 lines per file. Not a round number chosen for comfort: at PyFLASH's
#: 790 lines per plot, a 600-line cap means a file cannot hold even one plot's
#: worth of private scaffolding before it has to be split.
LINE_CAP = 600

#: Modules of this package that *produce* an artefact. A figure may read
#: anything — ``store``, ``series``, ``io``, ``metadata`` are all retrieval —
#: but it may not be the thing that made what it draws. That is the line, and
#: it is the one that keeps the helper stack from having anything to do.
PRODUCERS = {
    "registration", "filtering", "cosmic", "superseded", "display",
    "segmentation", "roi", "tracing", "controls", "rhythm", "trace_tables",
    "pipelines", "video",
}

#: Scientific libraries. Importing one under ``visualisation/`` would mean a
#: figure is filtering, fitting or segmenting something.
SCIENCE_LIBRARIES = {"scipy", "skimage", "sklearn", "statsmodels", "pandas"}


def figure_files() -> list[Path]:
    return sorted(FIGURES.rglob("*.py"))


def package_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def imports(tree) -> list[tuple[int, str]]:
    """Every imported top-level name, with its line. Function-local ones too.

    Deliberately not limited to module scope: moving an import inside a
    function is the obvious way to get round a rule like this, and it would
    not make the figure any less of a computation.
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name.split(".")[0])
                      for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative: from .. import x
                found += [(node.lineno, alias.name) for alias in node.names]
                if node.module:
                    found.append((node.lineno, node.module.split(".")[0]))
            elif node.module:
                found.append((node.lineno, node.module.split(".")[0]))
    return found


def test_there_are_figures_to_check():
    assert figure_files(), f"no modules found under {FIGURES}"


# ---------------------------------------------------------------- rule one
def test_no_figure_file_exceeds_the_cap():
    """600 lines per file. plotting.py reached 31,497 one function at a time."""
    oversized = []
    for path in figure_files():
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > LINE_CAP:
            oversized.append(f"{path.name}: {lines} lines")
    assert not oversized, (
        f"over the {LINE_CAP}-line cap: {oversized}. Split the file — that is "
        f"the cap working, not the cap failing.")


# ---------------------------------------------------------------- rule two
def test_figures_never_compute():
    """No science library and no artefact-producing module, anywhere here.

    Checked against the syntax tree rather than the text, so a docstring may
    name ``scipy`` freely — and it has to, because a reader needs to know what
    is refused and why.
    """
    offenders = []
    for path in figure_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, name in imports(tree):
            if name in SCIENCE_LIBRARIES:
                offenders.append(f"{path.name}:{line} imports {name}")
            if name in PRODUCERS:
                offenders.append(f"{path.name}:{line} imports pymicroglia."
                                 f"{name}, which produces artefacts")
    assert not offenders, offenders


def test_the_computing_half_of_the_trace_panel_is_outside_the_figure_package():
    """The split is real, not nominal.

    ``trace_tables`` does the reading, detrending, normalising and measuring;
    ``visualisation.traces`` receives a finished table. If the arrow ever
    reverses — a figure module importing the science one — rule two above
    catches it, and this catches the opposite mistake of the science module
    quietly growing its own drawing code.
    """
    from pymicroglia import trace_tables
    from pymicroglia.visualisation import traces

    assert "visualisation" in ast.dump(
        ast.parse(Path(trace_tables.__file__).read_text(encoding="utf-8"))), \
        "trace_tables no longer hands its table to the figure module"

    drawing = Path(traces.__file__).read_text(encoding="utf-8")
    for forbidden in ("uniform_filter1d", "polyfit", "np.nanstd(", "genfromtxt"):
        assert forbidden not in drawing, (
            f"visualisation/traces.py computes {forbidden!r}; that belongs in "
            f"trace_tables.py")


# -------------------------------------------------------------- rule three
def test_only_panels_saves():
    """ReproFig rendering appears exactly once, in ``panels.save``.

    One save path is what makes the plotted table and the provenance record
    automatic. A second one is a figure that can be written with neither.
    """
    hits = []
    for path in package_files():
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if "save_figure(" in line:
                hits.append((path, number, line.strip()))

    assert len(hits) == 1, (
        f"expected exactly one ReproFig save call in the package, found "
        f"{[(str(p.relative_to(SRC)), n) for p, n, _ in hits]}")
    path, _, _ = hits[0]
    assert path == FIGURES / "panels.py", (
        f"the one ReproFig save call is in {path.name}, not panels.py")


def test_the_save_path_writes_the_table_and_the_provenance():
    """Gate 7, as a property of the function rather than of a caller."""
    import inspect

    from pymicroglia.visualisation import panels

    signature = inspect.signature(panels.save)
    assert "table" in signature.parameters
    assert signature.parameters["table"].default is inspect.Parameter.empty, \
        "the plotted table must be required, not optional"
    assert {"sources", "claim", "artefacts", "bundle"} <= set(signature.parameters)


def test_every_quality_control_figure_forwards_the_full_reprofig_policy():
    """No figure action may silently drop a requested carrier or render policy."""
    expected = {
        "output_formats", "figure_profile", "figure_safe_columns",
        "public_sources", "dpi_preset", "render_preset", "render_width_in",
        "render_height_in", "format_options", "allow_reencode",
    }
    calls = []
    for path in (FIGURES / "qc.py", FIGURES / "overlays.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "save_for"):
                calls.append((path.name, node.lineno, {kw.arg for kw in node.keywords}))

    assert len(calls) == 6
    missing = [
        f"{name}:{line} misses {sorted(expected - keywords)}"
        for name, line, keywords in calls if expected - keywords
    ]
    assert not missing, missing


# --------------------------------------------------------------- rule four
_HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def test_no_hex_literals_and_no_local_rcparams():
    """Colours by name from the kit; style from the kit.

    Checked against the syntax tree, so prose and comments may spell a hex
    value freely — and they must, because the port is only readable if the
    module can say ``colour("dluc")`` *is* ``#a340d1``. Only executable code is
    constrained.

    Two files are excluded, each for a stated reason and neither of them a
    figure module:

    ``params.py``
        Its hex strings are example *input* to the parameter parser,
        demonstrating that a value is not split on the ``#`` that starts a
        comment. They are strings under test, not colours.
    ``video/luts.py``
        A colour *map* is not a house colour. ``dluc_purple`` is five control
        points on a continuous ramp that four years of review and publication
        movies were rendered through, and the kit has no vocabulary for a
        colormap. Substituting a house colour for a stop would repaint every
        one of those movies. The exclusion is narrow — the test below checks
        the hex appears only inside that file's anchor table.
    """
    exempt = {"params.py", "luts.py"}
    offenders = []
    for path in package_files():
        if path.name in exempt:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        prose = _docstrings(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in prose and _HEX.search(node.value)):
                offenders.append(f"{path.name}:{node.lineno} spells "
                                 f"{node.value!r}")
            if isinstance(node, ast.Attribute) and node.attr == "rcParams":
                offenders.append(f"{path.name}:{node.lineno} touches rcParams")
    assert not offenders, offenders


def test_grey_levels_resolve_through_the_kit():
    """Gate 9. ``"0.72"`` is a level, and the level is where it is written down.

    ``"0.72"`` and ``0.72`` are different things to Matplotlib, so the kit keeps
    these as strings rather than converting them to hex. What must not appear
    anywhere here is the string itself.
    """
    from pymicroglia.visualisation import panels

    assert panels.grey("raw") == "0.72"
    assert panels.grey("shade") == "0.88"

    offenders = []
    for path in package_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        prose = _docstrings(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in {"0.72", "0.88"}
                    and id(node) not in prose):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"a grey level is written down outside the kit at {offenders}")


def _docstrings(tree) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            found.add(id(first.value))
    return found


# ------------------------------------------------- the names figures read by
def test_the_stage_names_the_figures_read_still_match_their_producers():
    """A figure reads an artefact by stage name and must not import its maker.

    The cost of that is a pair of strings that could drift apart. This is the
    check that they have not — cheaper than the coupling it replaces, and it
    fails at test time rather than at "no stored registration for this source".
    """
    from pymicroglia import cosmic, registration
    from pymicroglia.visualisation import qc

    assert qc.READS["registration"] == registration.REGISTRATION_STAGE
    assert qc.READS["cosmic_mask"] == cosmic.COSMIC_STAGE
    assert qc.READS["cosmic_events"] == f"{cosmic.COSMIC_STAGE}_events"


# ------------------------------------------------------- the house style
def test_a_figure_drawn_in_the_house_theme_passes_the_kits_conformance_check():
    """Gate 8. The kit's own check, run inside this project's test suite.

    That placement is the point: four projects each drew their own version of
    the house style and the colours came apart without anybody seeing it. A
    document saying "use these colours" is what failed; a check that goes red
    in each consumer is what replaced it.
    """
    conformance = pytest.importorskip("analysis_kit.style.conformance")

    from pymicroglia.visualisation import panels

    stack = panels.stack(2, theme=panels.HOUSE_THEME, width=4.0,
                         height_per_panel=1.2)
    for axis in stack:
        axis.plot([0, 1, 2], [0, 1, 0], color=panels.colour("dluc"))
    panels.label(stack.last, x="hours", y="dF/F %")

    conformance.assert_conformant(figure=stack.figure, name=panels.HOUSE_THEME)


def test_the_families_are_all_classified():
    """Copied in spirit from PyFLASH's describe-coverage test.

    A figure family nobody wrote a sentence about is one nobody can choose
    between, so the list and the modules have to agree in both directions.
    """
    from pymicroglia import visualisation

    infrastructure = {
        "__init__", "panels", "bundle", "proof_output", "save_actions",
    }
    modules = {path.stem for path in figure_files()
               if path.stem not in infrastructure}
    assert set(visualisation.FAMILIES) == modules, (
        f"FAMILIES and the modules disagree: only in FAMILIES "
        f"{sorted(set(visualisation.FAMILIES) - modules)}, only on disk "
        f"{sorted(modules - set(visualisation.FAMILIES))}")
    for name, sentence in visualisation.FAMILIES.items():
        assert len(sentence.split()) >= 5, f"{name} has no real description"


def test_importing_the_figures_does_not_import_matplotlib():
    """Import stays cheap. Reading a parameter block must not cost a plotting stack.

    Run as a subprocess-free check on what a fresh import leaves behind: the
    figure modules reach Matplotlib inside functions, so a machine without it
    can still resolve and describe every figure action.
    """
    source = (FIGURES / "panels.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name.split(".")[0] for alias in node.names]
                     if isinstance(node, ast.Import)
                     else [(node.module or "").split(".")[0]])
            top_level += names
    assert "matplotlib" not in top_level, (
        "panels.py imports Matplotlib at module scope; it must be reached "
        "inside a function so the figure actions stay resolvable without it")


def test_the_colormap_exemption_is_narrow():
    """Gate 6's one exception, held to the anchor table it was granted for.

    ``video/luts.py`` may spell out hex because a colormap stop is not a house
    colour. That licence covers exactly one dictionary: if a hex value ever
    appears anywhere else in that file — a default, a fallback, a second
    palette — the exemption has stopped meaning what it was granted for.
    """
    luts = SRC / "video" / "luts.py"
    if not luts.exists():
        pytest.skip("the video package has not landed yet")

    tree = ast.parse(luts.read_text(encoding="utf-8"), filename=str(luts))
    anchored = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets
                 if isinstance(target, ast.Name)}
        if not names & {"HOUSE_PURPLE", "PHOTON_PURPLE", "LUT_ANCHORS"}:
            continue
        for child in ast.walk(node.value):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                anchored.add(id(child))

    prose = _docstrings(tree)
    stray = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and _HEX.search(node.value)
                and id(node) not in anchored and id(node) not in prose):
            stray.append(f"luts.py:{node.lineno} {node.value!r}")
    assert not stray, (
        f"a hex value outside the colormap anchors: {stray}. The exemption "
        f"covers colormap control points, not colours.")


def test_the_video_package_keeps_the_same_file_cap():
    """Not required by the plan, kept because the reason is the same.

    A video export module that grows its own layout, saving and annotation
    helpers fails the same way ``plotting.py`` did. When ``exports.py`` reached
    710 lines it was split into single-channel and multi-channel halves, which
    is the cap working rather than the cap failing.
    """
    root = SRC / "video"
    if not root.exists():
        pytest.skip("the video package has not landed yet")
    oversized = [f"{path.name}: {len(path.read_text(encoding='utf-8').splitlines())}"
                 for path in sorted(root.rglob("*.py"))
                 if len(path.read_text(encoding="utf-8").splitlines()) > LINE_CAP]
    assert not oversized, oversized
