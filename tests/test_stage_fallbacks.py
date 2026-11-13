"""A stage that cannot handle a recording is skipped whole; successful stages are unchanged."""
from types import ModuleType, SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest
from pymicroglia.tracking import shipped_engines
from pymicroglia.tracking import stage_fallbacks as fallbacks

ENGINE = shipped_engines.folder(shipped_engines.MOTION)


def wrap(function, count=5):
    rows, skipped = [], []
    stage = fallbacks.guarded('_well_x', function, count, fallbacks.Protection(), skipped, rows.append)
    return stage, rows, skipped


def movie():
    labels = np.zeros((3, 6, 6), np.uint16)
    labels[:, 1:3, 1:3] = 4
    return labels, np.zeros_like(labels)


def test_every_shipped_chain_stage_has_one_uniform_shape():
    source = (ENGINE / 'code/accepted_history.py').read_bytes().decode()
    shapes = fallbacks.stage_shapes(source)
    assert len(shapes) == 67
    assert shapes['_reconnected_companion_merge_partition'] == 5
    assert shapes['_generalised_repairs'] == 2
    assert '_confidence_layered_retracking' not in shapes and '_short_lived_reclaim' not in shapes


def test_failing_stage_passes_its_input_on_and_reports_zero():
    labels, unclaimed = movie()
    def stage(labels, unclaimed, config):
        table = pd.DataFrame([])
        return table.sort_values(['gap_start'])
    run, rows, skipped = wrap(stage)
    out = run(labels, unclaimed, {})
    assert out[0] is labels and out[1] is unclaimed
    assert [len(t) for t in out[2:4]] == [0, 0]
    assert out[4]['changed_pixels'] == 0 and out[4].get('applied_proposals', 0) == 0
    assert skipped == ['_well_x'] and rows[0]['action'] == 'skipped_input_kept'
    assert labels.flags.writeable and unclaimed.flags.writeable


def test_successful_stage_result_is_untouched_and_writable():
    labels, unclaimed = movie()
    def stage(labels, unclaimed):
        candidate = labels.copy(); candidate[0, 1, 1] = 9
        return candidate, unclaimed[:]
    run, rows, _ = wrap(stage, 2)
    out = run(labels, unclaimed)
    assert out[0][0, 1, 1] == 9 and labels[0, 1, 1] == 4
    assert out[1].flags.writeable and labels.flags.writeable and not rows
    out[1][0, 0, 0] = 1


def test_stage_writing_its_input_in_place_gets_its_normal_result():
    labels, unclaimed = movie()
    expected = labels.copy(); expected[1] = 7
    def stage(labels, unclaimed):
        labels[1] = 7
        return labels, unclaimed
    run, rows, _ = wrap(stage, 2)
    out = run(labels, unclaimed)
    assert np.array_equal(out[0], expected) and rows[0]['action'] == 'rerun_without_protection'


def test_failure_after_writing_input_in_place_still_stops_the_run():
    labels, unclaimed = movie()
    def stage(labels, unclaimed):
        labels[1] = 7
        raise AssertionError('duplicate claim')
    run, _, skipped = wrap(stage, 2)
    with pytest.raises(AssertionError):
        run(labels, unclaimed)
    assert not skipped


@pytest.mark.parametrize('error', [TypeError("'NoneType' object is not subscriptable"),
                                   RuntimeError('whole-movie lineage graph failed'),
                                   ValueError('duplicate lineage evidence for one identity pair')])
def test_unusual_data_errors_skip_the_stage(error):
    labels, unclaimed = movie()
    def stage(labels, unclaimed):
        raise error
    run, rows, skipped = wrap(stage, 2)
    assert run(labels, unclaimed)[0] is labels and skipped == ['_well_x']


def test_skipped_metrics_read_as_empty_in_the_history_summary():
    metrics = fallbacks.SkippedMetrics(stage_skipped=1)
    assert list(metrics['new_identities']) == [] and dict(metrics['target_counts']) == {}
    assert int(metrics['changed_pixels']) == 0 and not bool(metrics['old_identity_set_preserved'])
    assert json.loads(json.dumps({'n': metrics['n']})) == {'n': 0}


@pytest.mark.parametrize('error', [MemoryError(), ImportError('bad build'), OSError('disk'),
                                   NameError('undefined'),
                                   AttributeError("module 'x' has no attribute 'y'")])
def test_resource_and_code_faults_still_stop_the_run(error):
    labels, unclaimed = movie()
    def stage(labels, unclaimed):
        raise error
    run, _, _ = wrap(stage, 2)
    with pytest.raises(type(error)):
        run(labels, unclaimed)
    assert labels.flags.writeable


def test_missing_file_is_tolerated_only_after_an_earlier_skip():
    labels, unclaimed = movie()
    def stage(labels, unclaimed):
        raise FileNotFoundError('earlier stage output')
    run, _, skipped = wrap(stage, 2)
    with pytest.raises(FileNotFoundError):
        run(labels, unclaimed)
    skipped.append('_well_earlier')
    assert run(labels, unclaimed)[0] is labels


def test_install_wraps_chain_and_restores_everything(tmp_path, monkeypatch):
    source = '''
def run_accepted_history(labels, unclaimed):
    labels, unclaimed, audit, metrics = _well_a(labels, unclaimed)
    labels, unclaimed = _generalised_repairs(labels, unclaimed)
    return labels
def _well_a(labels, unclaimed):
    raise AssertionError('owners overlap')
def _generalised_repairs(labels, unclaimed):
    return labels, unclaimed
'''
    path = tmp_path/'accepted_history.py'; path.write_text(source)
    history = ModuleType('accepted_history'); history.__file__ = str(path)
    exec(source, history.__dict__)
    ledger = ModuleType('global_soma_ledger')
    ledger.resolve_anchor_identity_map = lambda *a: ({}, pd.DataFrame())
    monkeypatch.setitem(sys.modules, 'global_soma_ledger', ledger)
    class Stages:
        def call(self, method): return self.signature
    checkpoints = SimpleNamespace(Stages=Stages)
    originals = (history._well_a, history._generalised_repairs, ledger.resolve_anchor_identity_map, Stages.call)
    labels, unclaimed = movie()
    with fallbacks.install(history, checkpoints, tmp_path) as signature:
        assert history.run_accepted_history(labels, unclaimed) is labels
        stages = Stages(); stages.signature = {}
        assert stages.call('stage_m22_accepted_history')['stage_fallbacks'] == signature
    assert (history._well_a, history._generalised_repairs, ledger.resolve_anchor_identity_map, Stages.call) == originals
    audit = json.loads((tmp_path/fallbacks.AUDIT).read_text())
    assert [row['stage'] for row in audit] == ['_well_a']


@pytest.fixture
def soma_ledger(tmp_path, monkeypatch):
    shipped_engines.unpack(ENGINE, tmp_path, 'code/')
    before = set(sys.modules)
    monkeypatch.syspath_prepend(str(tmp_path/'code'))
    import global_soma_ledger
    yield global_soma_ledger
    for name in set(sys.modules) - before:
        if getattr(sys.modules[name], '__file__', '') and str(tmp_path) in sys.modules[name].__file__:
            sys.modules.pop(name)


def test_two_somas_claiming_one_anchor_name_get_separate_names(soma_ledger):
    objects = np.zeros((3, 40, 60), np.uint16)
    accepted = np.zeros_like(objects)
    objects[1, 5:12, 5:12] = 1; objects[1, 5:12, 45:52] = 2
    accepted[1, 5:12, 5:12] = 5; accepted[1, 5:12, 45:52] = 5
    params = soma_ledger.GlobalSomaLedgerParams(anchor_bookend_frames=1, minimum_substantial_core_px=4)
    with pytest.raises(AssertionError, match='duplicate anchor identity 5'):
        soma_ledger.resolve_anchor_identity_map(objects, accepted, 1, params)
    rows = []
    resolve = fallbacks.anchor_fallback(soma_ledger.resolve_anchor_identity_map, rows.append)
    mapping, audit = resolve(objects, accepted, 1, params)
    assert sorted(mapping.values()) == [5, 6] and rows[0]['action'] == 'new_identity_policy'
    accepted[1, 5:12, 45:52] = 8
    assert resolve(objects, accepted, 1, params)[0] == {1: 5, 2: 8} and len(rows) == 1


def test_anchor_policy_reaches_every_ledger_call_while_open(soma_ledger, tmp_path):
    objects = np.zeros((3, 40, 60), np.uint16)
    accepted = np.zeros_like(objects)
    objects[1, 5:12, 5:12] = 1; objects[1, 5:12, 45:52] = 2
    accepted[1, 5:12, 5:12] = 5; accepted[1, 5:12, 45:52] = 5
    params = soma_ledger.GlobalSomaLedgerParams(anchor_bookend_frames=1, minimum_substantial_core_px=4)
    audit = tmp_path/('final_' + fallbacks.AUDIT)
    with fallbacks.anchor_names(fallbacks.writer(audit)):
        mapping, _ = soma_ledger.resolve_anchor_identity_map(objects, accepted, 1, params)
    assert sorted(mapping.values()) == [5, 6]
    assert json.loads(audit.read_text())[0]['action'] == 'new_identity_policy'
    with pytest.raises(AssertionError, match='duplicate anchor identity 5'):
        soma_ledger.resolve_anchor_identity_map(objects, accepted, 1, params)


def test_final_stage_runs_under_the_anchor_policy():
    # MCG_10/MCG_11 (2026-10-28): the final stage replays tracking and met the
    # duplicate anchor after the accepted-history wrapper had been removed.
    import ast
    import inspect
    from pymicroglia.tracking import engine
    tree = ast.parse(inspect.getsource(engine))
    covered = [call for node in ast.walk(tree) if isinstance(node, ast.With)
               and any('anchor_names' in ast.unparse(item.context_expr) for item in node.items)
               for call in ast.walk(node) if isinstance(call, ast.Call)
               and ast.unparse(call.func) == '_execute_corrections']
    assert covered


def block_run(function, kind='ledger', steps=('first', 'second', 'third')):
    rows, skipped = [], []
    block = fallbacks.resilient_block('family', function, kind, list(steps), fallbacks.Protection(), skipped, rows.append)
    return block, rows, skipped


def test_fixing_block_failure_keeps_last_good_movie_for_remaining_steps():
    labels, ledger = movie()
    def apply(labels, ledger, raw, points, restored):
        first = labels.copy(); first[0, 1, 1] = 9
        yield 'first', first, ledger.copy(), {'audit': pd.DataFrame([{'x': 1}])}
        pd.DataFrame([]).owner
    block, rows, skipped = block_run(apply)
    items = list(block(labels, ledger, None, None, labels))
    assert [i[0] for i in items] == ['first', 'second', 'third']
    assert all(i[1] is items[0][1] for i in items[1:]) and items[0][1][0, 1, 1] == 9
    assert len(items[2][3]['recurrent_branch_applications']) == 0 and not dict(items[2][3])
    assert rows[0]['steps'] == ['second', 'third'] and skipped == ['family']
    assert labels.flags.writeable and items[0][1].flags.writeable


@pytest.mark.parametrize('kind', ['owner', 'arrays', 'process'])
def test_fixing_block_failing_before_any_step_passes_its_input_on(kind):
    labels, ledger = movie()
    def apply(*args):
        raise AssertionError('Original foreground/protection/application boundary failed')
        yield
    args = {'owner': (labels, None, None, ledger), 'process': ({'branch_labels': labels, 'ledger': ledger}, None)}
    block, _, _ = block_run(apply, kind)
    items = list(block(*args.get(kind, (labels, ledger))))
    assert [i[0] for i in items] == ['first', 'second', 'third']
    assert all(i[1] is labels for i in items)
    assert len(items[0]) == (5 if kind == 'arrays' else 4)
    assert items[0][3 if kind == 'owner' else 2] is not None


def test_fixing_block_writing_in_place_is_replayed_to_its_normal_result():
    labels, ledger = movie()
    def apply(labels, ledger):
        yield 'first', labels, ledger, {}
        labels[0] = 5
        yield 'second', labels, ledger, {}
    expected = [(i[0], i[1].copy()) for i in apply(*[a.copy() for a in movie()])]
    block, rows, _ = block_run(apply, steps=('first', 'second'))
    items = [(i[0], i[1].copy()) for i in block(labels, ledger)]
    assert [s for s, _ in items] == [s for s, _ in expected]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(items, expected))
    assert rows[0]['action'] == 'rerun_without_protection'


def test_fixing_block_that_succeeds_yields_exactly_the_original_steps():
    labels, ledger = movie()
    def apply(labels, ledger):
        for name in ('first', 'second'):
            labels = labels.copy(); labels[0, 0, 0] += 1
            yield name, labels, ledger.copy(), {'t': pd.DataFrame()}
    plain = list(apply(*movie()))
    block, rows, _ = block_run(apply, steps=('first', 'second'))
    wrapped = list(block(labels, ledger))
    assert [i[0] for i in wrapped] == [i[0] for i in plain] and not rows
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(wrapped, plain))


def test_feedback_blocks_wrap_every_block_and_restore(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(fallbacks, 'cyclic_repair', lambda module: nullcontext())
    monkeypatch.setitem(sys.modules, 'recurrent_cyclic_alias', None)
    engine = tmp_path/'completed_feedback_engine'; engine.mkdir()
    (engine/'source_hashes.json').write_text(json.dumps({'required_boundaries': [
        [family, 'only'] for family in fallbacks.FEEDBACK_BLOCKS]}))
    blocks = {family: SimpleNamespace(apply=lambda *a: iter(())) for family in fallbacks.FEEDBACK_BLOCKS}
    blocks['process'].inventory = inventory = lambda movies: iter(())
    blocks['process'].INVENTORIES = [('tracked', None)]
    sequence = SimpleNamespace(**blocks)
    originals = {family: block.apply for family, block in blocks.items()}
    from contextlib import contextmanager
    @contextmanager
    def methods(root, **kwargs):
        yield sequence
    feedback = SimpleNamespace(methods=methods)
    with fallbacks.feedback_blocks(feedback, tmp_path):
        with feedback.methods(tmp_path) as active:
            assert all(getattr(active, f).apply is not originals[f] for f in blocks)
    assert feedback.methods is methods
    assert all(block.apply is originals[f] for f, block in blocks.items())
    assert blocks['process'].inventory is inventory


def test_failed_process_scan_gives_no_eligible_pairs_for_the_remaining_scans():
    labels, _ = movie()
    def inventory(movies):
        yield 'tracked', pd.DataFrame({'well': ['A'], 'x': [1]})
        pd.DataFrame([]).valid_current_pair
    rows, skipped = [], []
    scan = fallbacks.resilient_inventory(inventory, ['tracked', 'temporal', 'owners'], fallbacks.Protection(), skipped, rows.append)
    items = list(scan({'A': {'labels': labels}}))
    assert [s for s, _ in items] == ['tracked', 'temporal', 'owners']
    assert list(items[-1][1].columns) == ['well'] and not len(items[-1][1])
    assert rows[0]['action'] == 'no_eligible_process_pairs' and labels.flags.writeable


@pytest.fixture
def cyclic(tmp_path, monkeypatch):
    shipped_engines.unpack(shipped_engines.folder(shipped_engines.COMPLETED_FEEDBACK), tmp_path, 'code/')
    before = set(sys.modules)
    monkeypatch.syspath_prepend(str(tmp_path/'code'))
    import recurrent_cyclic_alias
    yield recurrent_cyclic_alias
    for name in set(sys.modules) - before:
        if str(tmp_path) in (getattr(sys.modules[name], '__file__', '') or ''):
            sys.modules.pop(name)


def test_cyclic_alias_with_no_track_columns_counts_no_track_handoffs(cyclic, monkeypatch):
    labels = np.zeros((6, 20, 20), np.uint16)
    for t in range(6):
        labels[t, 4:9, 4:9] = 3 if t % 2 else 7
    points = pd.DataFrame(columns=['track_id', 'frame', 'x', 'y'])
    monkeypatch.setattr(cyclic, '_track_handoffs', lambda *a: pd.DataFrame([]))
    with pytest.raises(AttributeError, match='old_identity'):
        cyclic.discover(labels, points)
    with fallbacks.cyclic_repair(cyclic):
        proposals = cyclic.discover(labels, points)[0]
    assert len(proposals) and (proposals.track_handoffs == 0).all()
    columns = pd.DataFrame(columns=['old_identity', 'new_identity', 'frame'])
    monkeypatch.setattr(cyclic, '_track_handoffs', lambda *a: columns)
    expected = cyclic.discover(labels, points)[0]
    with fallbacks.cyclic_repair(cyclic):
        pd.testing.assert_frame_equal(cyclic.discover(labels, points)[0], expected)
