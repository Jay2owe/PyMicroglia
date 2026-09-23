"""The generated half of the skill's action catalogue.

A parameter table an agent reads is worse than none when it is stale: it will
confidently pass an argument that no longer exists, or the wrong default for one
that does. So the signatures are generated from the live registry and nothing
here is typed by hand.

Two things are checked. That the generator **produces the change** when the
registry moves — otherwise regenerating is theatre. And that the copy on disk is
currently in sync, which is skipped when the skill is not installed, because the
package has to be testable on a machine that has never seen it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymicroglia import catalogue
from pymicroglia.registry import build_registry, live_defaults

SKILL_DIR = Path.home() / ".claude" / "skills" / "pymicroglia"
CATALOG = SKILL_DIR / "references" / "action-catalog.md"
SLUG = "pymicroglia"


@pytest.fixture(scope="module")
def generator():
    kit = pytest.importorskip("analysis_kit.reference_gen")
    return kit


@pytest.fixture(scope="module")
def registry():
    built = build_registry()
    if not hasattr(built, "spec") or not hasattr(built, "add"):  # pragma: no cover
        pytest.skip("the kit's Registry is what the generator renders")
    return built


def _block(generator, registry) -> str:
    return "\n".join(generator.render_block_body(registry, "the test"))


# ------------------------------------------------------------ what it renders
def test_every_action_reaches_the_catalogue(generator, registry):
    block = _block(generator, registry)
    for name in registry.names():
        assert f"### `{name}`" in block, name
    assert len(catalogue.actions()) == len(registry.names())


def test_a_parameter_table_carries_the_default_the_code_applies(generator,
                                                                registry):
    """Not the one the protocol declared, where the two differ.

    ``run_controls`` searches 16-32 h; the engine it was copied from searched
    15-40. An agent reads this table immediately before passing an argument, so
    it has to say what will happen.
    """
    block = _block(generator, registry)
    section = block.split("### `run_controls`")[1].split("### `")[0]

    assert "| `ls_pmin` |" in section
    assert "`16.0`" in section
    assert "`15.0`" not in section
    assert live_defaults("run_controls")["ls_pmin"] == 16.0


def test_the_generator_produces_a_change_when_the_registry_moves(generator,
                                                                 registry):
    """Gate 6. Regenerating a catalogue that cannot change is theatre."""
    import analysis_kit as ak

    before = _block(generator, registry)
    assert "### `a_brand_new_action`" not in before

    moved = build_registry()
    moved.add("a_brand_new_action", ak.ActionSpec(
        summary="Something nobody has registered before.",
        method="segmentation.background", params=("source",)))
    after = _block(generator, moved)

    assert after != before
    assert "### `a_brand_new_action`" in after
    assert "Something nobody has registered before." in after


#: What the old block is filled with, so that finding it afterwards means the
#: splice missed it. A word that could plausibly appear in a real parameter
#: description is no sentinel at all: this was ``"stale"`` until a parameter
#: whose description warns about a stale copy of some weights made the test
#: fail for a reason that had nothing to do with splicing.
SENTINEL = "OLD_BLOCK_THAT_MUST_BE_REPLACED_5F3A"


def test_the_block_is_spliced_without_touching_the_prose(generator, registry):
    """People own the teaching prose; the generator owns the signatures."""
    existing = ("# Hand-written\n\nProse above.\n\n"
                f"{generator.start_marker(SLUG)}\n{SENTINEL}\n"
                f"{generator.end_marker(SLUG)}\n\nProse below.\n")
    spliced = generator.splice(existing, registry, slug=SLUG,
                               generator="the test")

    assert spliced.startswith("# Hand-written\n\nProse above.\n")
    assert spliced.endswith("\n\nProse below.\n")
    assert SENTINEL not in spliced
    assert "### `register`" in spliced


# ---------------------------------------------------------- the copy on disk
def test_the_installed_catalogue_is_in_sync(generator, registry):
    """Skipped where the skill is not installed: the package must be testable
    on a machine that has never seen it."""
    if not CATALOG.is_file():  # pragma: no cover - a fresh checkout
        pytest.skip(f"the /pymicroglia skill is not installed at {SKILL_DIR}")

    changes = generator.plan_updates(
        build_registry(reference_dir=CATALOG.parent), CATALOG, slug=SLUG,
        generator="tools/update_pymicroglia_references.py",
        reference_dir=CATALOG.parent)
    stale = [change for change in changes if change.stale]
    assert not stale, (
        "run: python tools/update_pymicroglia_references.py\n"
        + "\n".join(generator.diff_summary(change) for change in stale))


def test_the_skill_and_its_runner_are_where_the_runner_expects(generator):
    if not SKILL_DIR.is_dir():  # pragma: no cover - a fresh checkout
        pytest.skip(f"the /pymicroglia skill is not installed at {SKILL_DIR}")

    assert (SKILL_DIR / "SKILL.md").is_file()
    assert (SKILL_DIR / "scripts" / "pymicroglia_runner.py").is_file()
    for name in ("action-catalog.md", "translation-guide.md", "recipes.md"):
        assert (SKILL_DIR / "references" / name).is_file(), name

    front = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    assert "name: pymicroglia" in front
    assert "description:" in front
    assert "trigger:" not in front


def test_the_local_extension_skill_is_present():
    """Run from anywhere, extend where the code lives."""
    here = Path(__file__).resolve().parents[1]
    skill = here / ".claude" / "skills" / "pymicroglia-extend" / "SKILL.md"
    if not skill.is_file():
        pytest.skip("the checkout does not contain the local extension skill")
    assert skill.is_file(), f"no extension skill at {skill}"

    front = skill.read_text(encoding="utf-8").split("---")[1]
    assert "name: pymicroglia-extend" in front
    assert "trigger: /pymicroglia-extend" in front


def test_the_codex_mirror_is_the_same_bytes():
    """Gate 7. One copy, reached twice — never two copies to keep in step."""
    here = Path(__file__).resolve().parents[1]
    mirror = here / ".codex" / "skills" / "pymicroglia"
    if not mirror.exists() or not SKILL_DIR.is_dir():  # pragma: no cover
        pytest.skip("the skill or its Codex mirror is not installed here")

    for name in ("SKILL.md", "references/action-catalog.md",
                 "scripts/pymicroglia_runner.py"):
        assert (mirror / name).read_bytes() == (SKILL_DIR / name).read_bytes(), name


def test_the_runner_imports_and_answers_without_a_request():
    """It is a shim: importable, and every answer comes from the package."""
    if not SKILL_DIR.is_dir():  # pragma: no cover
        pytest.skip("the /pymicroglia skill is not installed")

    import importlib.util

    path = SKILL_DIR / "scripts" / "pymicroglia_runner.py"
    spec = importlib.util.spec_from_file_location("pymicroglia_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_ROOT_NAME == ".pymicroglia-agent"
    assert len(module.build_registry().names()) == len(catalogue.actions())

    refused = module.execute({"action": "no_such_thing", "params": {}})
    assert refused["error_type"] == "unknown_action"
    assert "pymicroglia-extend" in refused["error"]


def test_a_run_that_owes_a_claim_is_refused_by_the_runner(tmp_path, monkeypatch):
    """The skill branches on error_type, so the classification is the contract."""
    if not SKILL_DIR.is_dir():  # pragma: no cover
        pytest.skip("the /pymicroglia skill is not installed")

    import importlib.util

    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    path = SKILL_DIR / "scripts" / "pymicroglia_runner.py"
    spec = importlib.util.spec_from_file_location("pymicroglia_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    refused = module.execute({"action": "segment",
                              "params": {"source": str(tmp_path / "x.tif")},
                              "root": str(tmp_path)})
    assert refused["error_type"] == "bad_params"
    assert "needs a claim" in refused["error"]
    assert not list(tmp_path.glob("*.tif")), "nothing should have been written"


def test_the_runner_defaults_the_output_folder_to_the_working_directory():
    """Gate 1. The skill fires from anywhere; its outputs land where the person
    is, not somewhere they have to go looking."""
    if not SKILL_DIR.is_dir():  # pragma: no cover
        pytest.skip("the /pymicroglia skill is not installed")

    import importlib.util

    path = SKILL_DIR / "scripts" / "pymicroglia_runner.py"
    spec = importlib.util.spec_from_file_location("pymicroglia_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    here = Path.cwd().resolve()
    assert module._root({}) == here / module.DEFAULT_ROOT_NAME
    assert module._root({"root": "elsewhere"}) == here / "elsewhere"

    # Absolute, not relative. The record's path goes into the global index,
    # which exists so a run can be found from any directory; a relative pointer
    # is followable only from the one directory that wrote it, which is exactly
    # the failure the index was built to remove.
    assert module._root({}).is_absolute()
