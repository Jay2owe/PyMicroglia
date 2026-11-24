"""Run the existing recording producer with pinned optimized tracking adapters.

Completed native outputs are explicit inputs to a new correction invocation.
Their old signatures remain intact; this is not a migrated native checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time


STAGES = (
    "m0_inputs", "m1_motion", "m2_anchor", "m3_track", "m4_events", "m5_reconcile",
    "m7_persistence", "m8_merge", "m9_alias", "m12_identity_reservation",
    "m19_layered_lineage", "m20_oscillatory_lineage", "m22_accepted_history",
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def engine_digest(source):
    """The run marker's digest of a snapshot's engine: a folder now, a zip in older snapshots.

    Loaded by file path, because importing pymicroglia.tracking here would fix
    the implementation before the frozen one is chosen.
    """
    helper = Path(__file__).resolve().parents[1] / "src/pymicroglia/tracking/shipped_engines.py"
    spec = importlib.util.spec_from_file_location("_resume_shipped_engines", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.digest(source)


def contained(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Recorded input escapes its engine: " + relative)
    return result


def verify_snapshot(snapshot):
    manifest = json.loads((snapshot / "composition.json").read_text())
    for relative, expected in manifest["sources"].items():
        if sha(contained(snapshot, relative)) != expected:
            raise ValueError("Frozen candidate changed: " + relative)
    return manifest


def completed_native(root, run):
    return all((root / stage / run / "checkpoint.json").is_file()
               for stage in STAGES)


def verify_native(root, run, config, snapshot):
    """Verify original producers and every saved output without changing receipts."""
    state = json.loads((root / "tracking_resume.json").read_text())
    signature = state["signature"]
    if signature["config_sha256"] != sha(config):
        raise ValueError("Native parent configuration changed")
    if engine_digest(snapshot / "pymicroglia/tracking/data/motion_engine") != (root / "archive.sha256").read_text().strip():
        raise ValueError("Candidate changes the native engine archive")
    checks = {}
    for relative, expected in signature["code"].items():
        checks[str(contained(root, relative))] = expected
    for stage in STAGES:
        folder = root / stage / run
        receipt = json.loads((folder / "checkpoint.json").read_text())
        expected_signature = signature
        repair_record = root / "residency_frame_repair.json"
        if stage == "m22_accepted_history" and repair_record.is_file():
            expected_signature = {**signature, "stage_repair": json.loads(repair_record.read_text())}
        composition_record = root / 'parallel_repair_composition.json'
        if stage == 'm22_accepted_history' and composition_record.is_file():
            expected_signature = {**expected_signature, 'composition_repair': json.loads(composition_record.read_text())}
        lineage_record = root / 'local_lineage_overlap_repair.json'
        if stage == 'm22_accepted_history' and lineage_record.is_file():
            expected_signature = {**expected_signature, 'lineage_overlap_repair': json.loads(lineage_record.read_text())}
        owner_record = root / 'owner_consensus_attachment_repair.json'
        if stage == 'm22_accepted_history' and owner_record.is_file():
            expected_signature = {**expected_signature, 'owner_consensus_repair': json.loads(owner_record.read_text())}
        if receipt["signature"] != expected_signature:
            raise ValueError("Native stage has a different parent signature: " + stage)
        record = json.loads((folder / "run.json").read_text())
        if record.get("status") != "done":
            raise ValueError("Native stage is incomplete: " + stage)
        for relative, expected in receipt["files"].items():
            checks[str(contained(root, relative))] = expected
        for row in record.get("outputs", {}).values():
            path = Path(row["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Native stage output escapes its engine")
            checks[str(path)] = row["sha256"]
    for path, expected in checks.items():
        if sha(path) != expected:
            raise ValueError("Native source/output changed: " + path)
    return dict(status="verified native parent", native_stages=len(STAGES),
                verified_files=len(checks), parent_signature=signature,
                parent_checkpoint_sha256=sha(root / "m22_accepted_history" / run / "checkpoint.json"),
                parent=str(root), config_sha256=sha(config),
                verification="Original code and all checkpoint/manifest output hashes; original signatures retained")


def activate(snapshot):
    """Override only tracking in this process; other installed package APIs remain."""
    import pymicroglia
    import auto_organotypic
    if any(name.startswith("pymicroglia.tracking") for name in sys.modules):
        raise RuntimeError("Tracking imported before selecting the frozen implementation")
    pymicroglia.__path__.insert(0, str(snapshot / "pymicroglia"))
    for name in ("array_selection", "array_storage"):
        source = snapshot / (name + ".py")
        spec = importlib.util.spec_from_file_location("auto_organotypic." + name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(auto_organotypic, name, module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("well")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    authority = json.loads(args.authority.read_text())
    measurement_timeout = authority.get('measurement_timeout_seconds', 3600)
    if not isinstance(measurement_timeout, int) or not 1 <= measurement_timeout <= 86400:
        raise ValueError('Measurement timeout must be an explicit 1-86400 second batch setting')
    os.environ['PYMICROGLIA_MEASUREMENT_TIMEOUT_SECONDS'] = str(measurement_timeout)
    snapshot = Path(authority["snapshot"]).resolve()
    producer = Path(authority["producer"]).resolve()
    output = args.output.resolve()
    home = producer.parent
    if output != (home / "rigid_analysis" / args.well).resolve():
        raise ValueError("Output must be the existing declared recording folder")
    if sha(snapshot / "composition.json") != authority["composition_sha256"]:
        raise ValueError("Candidate composition changed after restart authorization")
    verify_snapshot(snapshot)
    if authority.get("residency_repair"):
        repair = authority["residency_repair"]
        if sha(Path(repair["source"])) != repair["sha256"]:
            raise ValueError("Authorized residency repair source changed")
    inputs = json.loads((home / "rigid_batch/.j" / args.well / "motion_inputs.json").read_text())
    stem = "95_full_" + args.well
    config = Path(inputs["prepared"][stem]["config"]).resolve()
    root, run = output / "tracking/engine", "pymicroglia_" + stem
    repair_record = root / "residency_frame_repair.json"
    if repair_record.is_file() and json.loads(repair_record.read_text()).get("repair_sha256") != authority.get("residency_repair", {}).get("sha256"):
        raise ValueError("Native parent repair differs from the authorized source")
    composition = authority.get('composition_repair')
    composition_record = root / 'parallel_repair_composition.json'
    if composition and sha(composition['source']) != composition['sha256']:
        raise ValueError('Authorized composition repair source changed')
    if composition_record.exists() and json.loads(composition_record.read_text()).get('repair_sha256') != (composition or {}).get('sha256'):
        raise ValueError('Native parent composition differs from the authorized source')
    lineage = authority.get('lineage_overlap_repair')
    lineage_record = root / 'local_lineage_overlap_repair.json'
    if lineage and sha(lineage['source']) != lineage['sha256']:
        raise ValueError('Authorized local-lineage repair source changed')
    if lineage_record.exists() and json.loads(lineage_record.read_text()).get('repair_sha256') != (lineage or {}).get('sha256'):
        raise ValueError('Native parent local-lineage repair differs from the authorized source')
    owner_repair = authority.get('owner_consensus_repair')
    owner_record = root / 'owner_consensus_attachment_repair.json'
    if owner_record.exists() and json.loads(owner_record.read_text()).get('repair_sha256') != (owner_repair or {}).get('sha256'):
        raise ValueError('Native parent owner-consensus repair differs from authorized source')
    for key in ('process_proposal_repair', 'fragment_observation_repair', 'owner_consensus_repair', 'core_replacement_evidence_repair'):
        if authority.get(key) and sha(authority[key]['source']) != authority[key]['sha256']:
            raise ValueError('Authorized correction repair source changed: ' + key)
    parent = verify_native(root, run, config, snapshot) if completed_native(root, run) else None
    if root.exists() and parent is None:
        # Ordinary resume may reuse only the same candidate, never older partial work.
        previous = output / "optimized_continuation.json"
        if not previous.is_file() or json.loads(previous.read_text()).get("composition_sha256") != authority["composition_sha256"]:
            raise ValueError("Older partial native run requires separate recovery; no implicit reuse")
    report = dict(status="ready", snapshot=str(snapshot),
                  composition_sha256=authority["composition_sha256"],
                  native_parent=parent, authority=str(args.authority.resolve()),
                  scientific_acceptance="review pending; user-authorized current-code execution")
    if args.verify_only:
        print(json.dumps(report, indent=2))
        return
    activate(snapshot)
    from pymicroglia.tracking import engine
    # Keep frozen archives intact; version the explicit overlap repair separately.
    correction_contexts = []
    for name in ('process_proposal_repair', 'fragment_observation_repair', 'core_replacement_evidence_repair'):
        if not authority.get(name): continue
        repair = authority[name]
        repair_source = Path(repair['source'])
        if sha(repair_source) != repair['sha256']:
            raise ValueError('Authorized process proposal repair source changed')
        spec = importlib.util.spec_from_file_location(name, repair_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from pymicroglia.tracking import completed_feedback
        correction_contexts.append(module.activate(completed_feedback, output / 'tracking'))
        report[name] = repair
    original = engine._execute_pipeline
    if parent:
        def reuse(pipeline, code, actual_root, run_name, actual_stem, actual_config, *pos, **kw):
            if (Path(actual_root).resolve() != root or run_name != run or actual_stem != stem
                    or Path(actual_config).resolve() != config):
                raise ValueError("Native-parent reuse called for an unverified recording")
            common = sys.modules["common"]
            pipeline._verify_pins(common.Config.load(config), stem)
            print("CHECKPOINT verified 13 native stages; running current corrections from their saved outputs", flush=True)
        engine._execute_pipeline = reuse
    elif authority.get("residency_repair") or composition or lineage or owner_repair:
        adapters = []
        for name in ('residency_repair', 'composition_repair', 'lineage_overlap_repair', 'owner_consensus_repair'):
            if not authority.get(name): continue
            repair = authority[name]
            repair_source = Path(repair['source'])
            if sha(repair_source) != repair['sha256']:
                raise ValueError('Authorized repair source changed: ' + name)
            spec = importlib.util.spec_from_file_location(name, repair_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            adapters.append(module)
            report[name] = repair
        def repaired(pipeline, code, actual_root, *pos, **kw):
            import importlib
            from contextlib import ExitStack
            from pymicroglia.tracking import checkpoints
            history = importlib.import_module("accepted_history")
            with ExitStack() as stack:
                for adapter in adapters:
                    stack.enter_context(adapter.install(history, checkpoints, actual_root))
                return original(pipeline, code, actual_root, *pos, **kw)
        engine._execute_pipeline = repaired
    output.mkdir(parents=True, exist_ok=True)
    # Repair contexts write provenance before the producer starts a fresh run.
    (output / 'tracking').mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    # Keep old invocation records. Do not rewrite any native checkpoint signature.
    for path in (root / "accepted_stage_timings.jsonl", output / "tracking_action.json"):
        if path.is_file():
            if path.name == "tracking_action.json":
                response = json.loads(path.read_text())
                result = response.get("result")
                if (response.get("ok") and isinstance(result, dict)
                        and result.get("labels") and result.get("raw")
                        and isinstance(result.get("source_frame_offset"), int)):
                    continue
            path.rename(path.with_name(path.stem + "_before_optimized_" + stamp + path.suffix))
    receipt = output / "optimized_continuation.json"
    if receipt.is_file():
        receipt.rename(receipt.with_name("optimized_continuation_before_" + stamp + ".json"))
    report["status"] = "running"
    receipt.write_text(json.dumps(report, indent=2))
    sys.path.insert(0, str(home))
    sys.argv = [str(producer), args.well, str(output)]
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            for context in correction_contexts: stack.enter_context(context)
            runpy.run_path(str(producer), run_name="__main__")
        report["status"] = "tracking and measurement complete"
    except BaseException as exc:
        report.update(status="failed; outputs preserved", error=repr(exc))
        raise
    finally:
        engine._execute_pipeline = original
        try:
            verify_snapshot(snapshot)
            report["source_unchanged"] = sha(snapshot / "composition.json") == authority["composition_sha256"]
        except (ValueError, OSError):
            report["source_unchanged"] = False
        if not report["source_unchanged"]:
            report["status"] = "failed; frozen execution source changed"
        receipt.write_text(json.dumps(report, indent=2))
    if not report["source_unchanged"]:
        raise RuntimeError("Tracking implementation changed during execution")


if __name__ == "__main__":
    main()
