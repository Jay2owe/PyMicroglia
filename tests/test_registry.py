"""The catalogue, the registry, and the knowledge commands built on them."""

from __future__ import annotations

import types

import pytest

from pymicroglia import catalogue, config, knowledge, registry


#: An action with nothing behind it, made by unbinding a real one.
#:
#: Until stage 11 there was always a genuinely pending action to borrow, and
#: these tests borrowed the first. Stage 11 bound the last two, so ``pending()``
#: is now empty — which is the whole point of it and not a reason to stop
#: testing the mechanism. Every future stage declares its actions before it
#: implements them, so "declared but not resolvable" has to keep working.
#:
#: The unbinding is a monkeypatch, so it lasts one test and the registry is
#: whole again afterwards.
UNBOUND = "remove_cosmic_rays"


@pytest.fixture
def pending_action(monkeypatch) -> str:
    """One action made pending for the duration of a test."""
    method = registry.REGISTRY.binds_to(UNBOUND)
    module_name, _, _ = method.rpartition(".")
    monkeypatch.delitem(registry.REGISTRY.modules, module_name, raising=False)
    assert UNBOUND in registry.pending()
    return UNBOUND


#: How many actions PyMicroglia declares. Thirteen came from the registered
#: protocols; stages add their own, so this is read from the shipped catalogue
#: rather than typed — the tests below check that every layer agrees on the
#: number, which is the property worth holding, not the number itself.
ACTION_COUNT = len(catalogue.actions())
assert ACTION_COUNT >= 13, "the thirteen registered protocols are the floor"


# ── the shipped catalogue ────────────────────────────────────────────────────
def test_catalogue_ships_with_the_package():
    """It is a data file, not a run-time read of the protocol scripts.

    The scripts may not be on the machine at all; the catalogue always is.
    """
    assert catalogue.DATA.exists()
    assert len(catalogue.actions()) == ACTION_COUNT


def test_every_action_has_a_summary_and_a_target():
    for entry in catalogue.actions():
        assert entry["summary"].strip(), entry["name"]
        assert entry["method"].strip(), entry["name"]
        assert entry["params"], entry["name"]


def test_defaults_are_per_action_not_shared():
    """One name, one meaning — but honestly different values per action.

    ``crf`` means the same thing everywhere, so the shared vocabulary holds one
    description for it. Its value differs between video exporters, so the
    default comes from the action.
    """
    shared = {doc.name: doc for doc in catalogue.param_docs()}
    assert shared["crf"].default is None

    users = [entry for entry in catalogue.actions() if "crf" in entry["params"]]
    assert len(users) > 1
    for entry in users:
        rows = {row["name"]: row for row in catalogue.action_params(entry["name"])}
        assert rows["crf"]["default"] == entry["defaults"]["crf"]
        assert rows["crf"]["description"] == shared["crf"].description


def test_action_params_of_an_unknown_action_is_empty():
    assert catalogue.action_params("no_such_action") == []


# ── the registry ─────────────────────────────────────────────────────────────
def test_all_actions_are_registered():
    assert len(registry.REGISTRY.names()) == ACTION_COUNT


def test_actions_bind_by_dotted_name():
    assert registry.REGISTRY.binds_to("remove_cosmic_rays") == "cosmic.remove_cosmic_rays"


def test_nothing_is_pending_any_more():
    """Gate 9 of stage 11: all thirteen registered protocols are backed.

    This is the number the whole plan was counting down. It started at thirteen
    and reaching zero is the claim that every protocol an agent can ask for is
    one this package can actually run.

    The Motion port's stage 02 added one action that is pending *by design*:
    ``track`` fronts a tracker reached through a dotted name outside this
    package, and its module says so through ``status()``. So what this asserts
    now is that the only pending actions are declared seams -- never one of
    the protocols, and never an action whose module simply failed to land.

    Stage 03 added a second seam: ``contrasts`` takes its tests from
    Circadian Workbench through the one importer module the rhythm stage
    adds, and ``measure.contrasts.status()`` reports it pending until then.
    """
    for name in registry.pending():
        answer = registry.seam_status(registry.REGISTRY.binds_to(name))
        assert answer is not None and answer[0] == "pending", (
            f"{name} is pending and is not a declared seam")
    assert set(registry.pending()) <= {"track", "contrasts"}


def test_unimplemented_actions_are_pending_not_broken():
    """A half-ported package is a visible state, not a surprise.

    Nothing is pending today, so this asserts the invariant rather than the
    count: whatever is pending, if anything, is a declared action whose target
    does not resolve — never an action that is missing from the registry.
    """
    assert set(registry.pending()) <= set(registry.REGISTRY.names())
    for name in registry.pending():
        assert registry.REGISTRY.resolve(name) is None


def test_binding_a_module_clears_its_pending_actions(monkeypatch,
                                                     pending_action):
    """Declaring an action before implementing it stays a workable state.

    It was the normal state until stage 11 and it is how every later stage
    starts: the action is in the catalogue, ``describe`` answers about it, and
    it reports as pending until its module arrives.
    """
    action = pending_action
    method = registry.REGISTRY.binds_to(action)
    module_name, _, attribute = method.rpartition(".")
    before = set(registry.pending())
    assert action in before

    fake = types.SimpleNamespace(**{attribute: lambda **kw: "cleaned"})
    monkeypatch.setitem(registry.REGISTRY.modules, module_name, fake)

    assert registry.REGISTRY.resolve(action) is not None
    assert action not in registry.pending()


def test_a_missing_science_module_is_not_an_error():
    """Import failures during build-out are expected and swallowed; other
    failures are not. Only ImportError is caught in _load_modules."""
    assert isinstance(registry.MODULES, dict)


# ── describe / discover / validate / doctor ──────────────────────────────────
def test_describe_lists_every_action():
    payload = knowledge.describe()
    assert payload["ok"] is True
    assert len(payload["actions"]) == ACTION_COUNT


def test_describe_one_action_carries_its_own_defaults():
    payload = knowledge.describe("remove_cosmic_rays")
    rows = {row["name"]: row for row in payload["params"]}
    assert rows["seed_z"]["default"] == 12.0
    assert "CAUTION" in rows["seed_z"]["description"]
    assert payload["method_version"]


def test_describe_an_unknown_action_says_what_is_available():
    payload = knowledge.describe("not_an_action")
    assert payload["ok"] is False
    assert payload["error"] == "unknown_action"
    assert "remove_cosmic_rays" in payload["available"]


def test_display_only_actions_are_flagged():
    """AGENTS.md branches display work off the measurement path; the catalogue
    carries that so a caller can see it before running anything."""
    flagged = {e["name"] for e in catalogue.actions() if e.get("display_only")}
    assert {"remove_static_background", "display_filter"} <= flagged
    assert "remove_cosmic_rays" not in flagged


def test_discover_reports_counts_and_pending():
    report = knowledge.discover()
    assert report["actions"] == ACTION_COUNT
    assert report["params"] > 0
    assert report["param_uses"] >= report["params"]
    assert set(report["pending"]) == set(registry.pending())
    assert not report["missing"]
    assert not report["unregistered"]
    assert not report["undocumented"]


def test_validate_rejects_an_unknown_parameter():
    result = knowledge.validate("remove_cosmic_rays", {"seed_z": 8, "nope": 1})
    assert result["ok"] is False
    assert result["unknown_params"] == ["nope"]
    assert "describe remove_cosmic_rays" in result["message"]


def test_validate_accepts_a_known_parameter_and_flags_pending(pending_action):
    result = knowledge.validate(pending_action, {})
    assert result["ok"] is True
    assert result["pending"] is True
    assert "not yet implemented" in result["note"]


def test_doctor_names_the_interpreter():
    """"The audit did nothing" and "you ran the wrong Python" look identical
    from outside. This is the command that tells them apart."""
    import sys

    report = knowledge.doctor()
    assert report["interpreter"] == sys.executable
    assert report["actions"] == ACTION_COUNT
    assert "store_root" in report and "store_free_gb" in report


def test_doctor_reports_where_the_store_is_without_complaining_about_it(
        monkeypatch, tmp_path):
    """A synced store is the design now, so its location is a fact, not a fault.

    What is still a fault is a dehydrated array, and that has its own test in
    ``test_store_tiers.py`` where the store fixture can plant one.
    """
    synced = tmp_path / "UK Dementia Research Institute Dropbox" / "store"
    monkeypatch.setenv("PYMICROGLIA_STORE", str(synced))
    report = knowledge.doctor()

    assert report["store_synced"] is True
    assert report["store_dehydrated"] == 0
    assert not any("synced folder" in complaint
                   for complaint in report["complaints"])


def test_store_root_is_the_project_store_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("PYMICROGLIA_STORE", raising=False)
    monkeypatch.delenv("AUTO_ORGANOTYPIC_STORE", raising=False)
    project = tmp_path / "Microglia Project"
    source = project / "PyMicroglia" / "src" / "pymicroglia" / "__init__.py"
    source.parent.mkdir(parents=True)
    assert config.store_root(start=source, projects=(project.name,)) == (
        project / config.STORE_DIRNAME)


def test_cache_cap_is_configurable(monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_CACHE_GB", "8")
    assert config.cache_cap_bytes() == 8 * 1024 ** 3
    monkeypatch.setenv("PYMICROGLIA_CACHE_GB", "nonsense")
    assert config.cache_cap_bytes() == config.DEFAULT_CACHE_GB * 1024 ** 3


# ── run_action ───────────────────────────────────────────────────────────────
def test_running_a_pending_action_says_which_stage_is_missing(pending_action):
    from pymicroglia import ActionPending, run_action

    action = pending_action
    with pytest.raises(ActionPending) as caught:
        run_action(action, source="x.tif")
    assert registry.REGISTRY.binds_to(action) in str(caught.value)


def test_running_an_unknown_action_lists_the_real_ones():
    from pymicroglia import ActionInvalid, run_action

    with pytest.raises(ActionInvalid) as caught:
        run_action("nope")
    assert "remove_cosmic_rays" in str(caught.value)


def test_a_bad_parameter_is_caught_before_anything_runs(monkeypatch):
    from pymicroglia import ActionInvalid, run_action

    called = []
    fake = types.SimpleNamespace(remove_cosmic_rays=lambda **kw: called.append(kw))
    monkeypatch.setitem(registry.REGISTRY.modules, "cosmic", fake)

    with pytest.raises(ActionInvalid):
        run_action("remove_cosmic_rays", not_a_parameter=1)
    assert not called, "validation must happen before the action is called"


def test_a_bound_action_runs(monkeypatch, tmp_path):
    from pymicroglia import run_action

    fake = types.SimpleNamespace(remove_cosmic_rays=lambda **kw: {"cleaned": True, **kw})
    monkeypatch.setitem(registry.REGISTRY.modules, "cosmic", fake)
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "store"))

    result = run_action("remove_cosmic_rays", claim="unit test",
                        output_roots=[tmp_path], seed_z=8)
    assert result["cleaned"] is True
    assert result["seed_z"] == 8


# ── the command line ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("argv", [["doctor"], ["discover"], ["describe"],
                                  ["describe", "remove_cosmic_rays"]])
def test_cli_commands_succeed(argv, capsys):
    from pymicroglia.cli import main

    assert main(argv) == 0
    assert capsys.readouterr().out.strip().startswith("{")


def test_cli_reports_an_unknown_action_as_failure(capsys):
    import json

    from pymicroglia.cli import main

    assert main(["describe", "nope"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "unknown_action"


def test_cli_parses_literal_values(capsys):
    import json

    from pymicroglia.cli import main

    assert main(["validate", "remove_cosmic_rays", "seed_z=8.5"]) == 0
    assert json.loads(capsys.readouterr().out)["unknown_params"] == []


def test_cli_run_of_a_pending_action_explains_itself(capsys, pending_action):
    import json

    from pymicroglia.cli import main

    assert main(["run", pending_action, "source=x.tif"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "pending"


def test_every_bound_action_accepts_exactly_the_parameters_it_declares():
    """``describe`` and ``validate`` must not promise a keyword ``run`` rejects.

    ``run`` calls the target as ``function(**params)``, so a name in the
    catalogue that the function does not take is a ``TypeError`` at the moment
    somebody actually starts a six-hour job — and a keyword the function takes
    but never declares is one no agent will ever find. Both directions are
    checked, for every action whose stage has landed.

    Parameters an engine declared and this package does not use are still
    accepted and documented as ignored, rather than dropped: the engine's
    parameter block is the shared vocabulary, and quietly losing a name from it
    is how two tools stop meaning the same thing by it.
    """
    import inspect

    for name in registry.REGISTRY.names():
        function = registry.REGISTRY.resolve(name)
        if function is None:
            continue                      # its stage has not landed yet
        accepted = {parameter.name for parameter in
                    inspect.signature(function).parameters.values()
                    if parameter.kind != inspect.Parameter.VAR_KEYWORD}
        declared = {row["name"] for row in catalogue.action_params(name)}
        assert declared - accepted == set(), \
            f"{name} declares parameters it cannot accept: " \
            f"{sorted(declared - accepted)}"
        assert accepted - declared == set(), \
            f"{name} accepts undeclared parameters: {sorted(accepted - declared)}"


def test_the_catalogue_shipped_with_the_package_is_current():
    """A stale catalogue is a lie about what the package does.

    The generator is the source of truth and the JSON is its output; this is
    ``tools/regenerate_catalogue.py --check`` run as a test, so a stage that
    adds an action cannot forget to regenerate.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "regenerate_catalogue.py"
    if not script.is_file():
        pytest.skip("generator not shipped")
    protocols = config.protocols_root() if hasattr(config, "protocols_root") else None
    if protocols is None:
        import os
        override = os.environ.get("PYMICROGLIA_PROTOCOLS")
        default = root.parent / "Protocols"
        protocols = Path(override) if override else default
    if not Path(protocols).is_dir():
        pytest.skip("Protocols folder not available; the generator reads it")

    finished = subprocess.run(
        [sys.executable, str(script), "--check", "--protocols", str(protocols)],
        capture_output=True, text=True)
    assert finished.returncode == 0, finished.stdout + finished.stderr
