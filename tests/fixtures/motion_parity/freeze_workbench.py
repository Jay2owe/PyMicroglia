"""Re-record the rhythm-derived tables on the installed Circadian Workbench.

Motion was frozen with Circadian Workbench 0.8.2. Later Workbench releases
deliberately changed the rhythm arithmetic (the Lomb-Scargle peak is refined
between grid frequencies, and periodogram power, phase, amplitude and fit
quality moved with it), so the tables built from rhythm fits can no longer
reproduce Motion's numbers. Every column the Workbench does not compute is
still compared with Motion; the Workbench outputs are compared with the tables
this script writes, run with Motion's settings rather than today's defaults.

Re-run it only after confirming that a Workbench change is intended, naming
the groups to record (all four when none is named):

    python tests/fixtures/motion_parity/freeze_workbench.py [spatial] [tissue] [figure] [measured]
"""
import ast
import gzip
import importlib
import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASELINE = HERE / 'workbench'
RECORD = BASELINE / 'record.json'

#: Motion's settings, pinned so later package defaults (``asls`` detrending,
#: the ``integrated_density_mean3`` reporter) do not enter a parity check.
MOTION_SETTINGS = {'detrend': 'robust_linear', 'reference_metric': 'corrected_mean',
                   'reporter_metric': 'corrected_mean'}

#: Motion's module settings in the measured run where later defaults moved
#: (the reporter and ``asymmetric_least_squares`` detrending). The rhythms
#: block in ``config.json`` already pins its own.
MOTION_MODULE_SETTINGS = {
    'coupling': {'metric_pairs': [['turnover_index', 'corrected_mean'], ['area_px', 'turnover_index']],
                 'between_metrics': ['corrected_mean'], 'detrend': 'robust_linear'},
    'recurrence': {'detrend': 'robust_linear'},
    'trend': {'metrics': ['area_px', 'corrected_mean', 'solidity', 'ramification_index',
                          'turnover_index', 'step_px']},
}

#: A measurement added after Motion: its columns and rows are left out of the comparison.
ADDED_METRIC = 'integrated_density_mean3'

#: Measured-run table -> the columns a Workbench fit or detrend writes there.
#: These meet the baseline; every other column still equals Motion. Pooled
#: tables hold the one movie's rows, so both meet the same baseline.
MEASURED = {
    'rhythms.csv': ('best_period_hours', 'best_p_value', 'best_goodness_of_fit', 'best_phase_hours',
                    'best_phase_fraction', 'cycles_covered_at_best_period', 'lombscargle_period_hours',
                    'lombscargle_power', 'lombscargle_false_alarm'),
    'rhythm_methods.csv': ('cycles_covered', 'period_underdetermined', 'method_rhythmic', 'method_rhythm_status',
                           'workbench_run_record_json', 'period_hours', 'phase_hours', 'amplitude',
                           'goodness_of_fit', 'p_value', 'significant'),
    'rhythm_traces.csv': ('detrended_z',),
    'rhythms_population.csv': ('vector_length', 'rayleigh_p_value', 'mean_peak_fraction',
                               'phase_reference_period_hours', 'mean_peak_hour'),
    'coupling.csv': ('correlation', 'phase_difference', 'phase_difference_fraction', 'period_a_hours',
                     'period_b_hours', 'pair_phase_reference_period_hours'),
    'lag_profiles.csv': ('correlation', 'surrogate_mean', 'surrogate_lo', 'surrogate_hi'),
    'recurrence.csv': ('recurrence_rate', 'recurrence_surrogate_mean', 'recurrence_surrogate_hi'),
    'recurrence_quantification.csv': ('recurrence_threshold_distance', 'determinism', 'laminarity',
                                      'determinism_surrogate', 'laminarity_surrogate'),
}

#: Columns a Workbench fit writes. Everything else in a rhythm table must
#: still equal Motion.
WORKBENCH_COLUMNS = frozenset({
    'amplitude', 'cycles_observed', 'estimate_diagnostics', 'estimator_p_value',
    'goodness_of_fit', 'p_value', 'peak_power', 'period_hours', 'phase_hours', 'q_value',
    'significance_period_hours', 'significance_run_record_json', 'workbench_run_record_json'})

#: spatial figure key -> prepare kind, for the maps whose values pass through a Workbench fit or detrend.
SPATIAL = {'tissue_expansion_sequence': 'expansion', 'spatial_rhythm_progression': 'progression',
           'reporter_shape_timing': 'timing', 'spatial_rhythm_maps': 'period'}

#: Saved figures whose plotted values pass through a Workbench fit or detrend:
#: figure key -> (view, the options Motion's frozen figure recorded beyond
#: ``MOTION_SETTINGS``, columns besides ``WORKBENCH_COLUMNS`` a fit or detrend writes).
FIGURES = {
    'cd68_reporter_rhythm': ('raster', {}, ('detrended_z',)),
    'metric_rhythm_matrix': ('matrix', {}, ()),
    'all_cell_trace_grid': ('traces', {'multiple_testing': 'bh', 'fft_component_test': False,
                                       'fit_method': 'lomb', 'significance_method': 'lomb'},
                            ('processed_value', 'value')),
    'radial_occupancy_rhythms': ('matrix', {}, ()),
}

#: tissue_tectonics tables built from the rhythm fits.
TISSUE = ('significant_period_pixels.csv', 'significant_period_cells.csv',
          'rhythm_map_fits.csv', 'statistics.csv', 'figure_data.csv')


def motion_metric(value):
    """``value`` with the reporter added since Motion replaced by Motion's ``corrected_mean``."""
    if isinstance(value, str):
        return value.replace(ADDED_METRIC, 'corrected_mean')
    if isinstance(value, (list, tuple)):
        return type(value)(motion_metric(item) for item in value)
    return value


def options(key, **overrides):
    """A figure's defaults with Motion's settings and reporter in place of later defaults."""
    from pymicroglia.visualisation.figures import load
    spec = load()[key]
    defaults = {o.name: motion_metric(o.default) for o in spec.options}
    return {**defaults, **{k: v for k, v in MOTION_SETTINGS.items() if k in defaults}, **overrides}


def figure_table(key):
    """The plotted table of a ``FIGURES`` figure, drawn from Motion's run with Motion's settings."""
    from pymicroglia.figure_tables.saved import Tables
    from pymicroglia.visualisation.figures import load
    from tests.test_motion_parity import FROZEN_RUN
    view, overrides, _ = FIGURES[key]
    module, name = load()[key].prepare.split(':')
    prepare = getattr(importlib.import_module(module), name)
    return prepare(Tables(FROZEN_RUN, 'parity_A1'), options(key, **overrides))[0][view]['table']


def moved_columns(key):
    """The columns of a ``FIGURES`` table that meet the baseline instead of Motion."""
    return WORKBENCH_COLUMNS | set(FIGURES[key][2])


def spatial_table(key):
    from pymicroglia.figure_tables.saved import Tables
    from pymicroglia.figure_tables.spatial import prepare
    from tests.test_motion_parity import FROZEN_RUN
    return prepare(Tables(FROZEN_RUN, 'parity_A1'), options(key), SPATIAL[key])[0]


def tissue_tables():
    from pymicroglia.figure_tables.saved import Tables
    from tests.test_motion_parity import FROZEN_RUN
    from pymicroglia.visualisation.figures import load
    module, name = load()['tissue_tectonics'].prepare.split(':')
    prepare = getattr(importlib.import_module(module), name)
    return prepare(Tables(FROZEN_RUN, 'parity_A1'), options('tissue_tectonics'))[0].auxiliary


def measured_run(folder):
    """The parity fixture measured by the ported modules with Motion's settings, pooled and contrasted."""
    import os
    from measure_stubs import fixture_copy
    from pymicroglia.measure import load_config
    from pymicroglia.measure.contrasts import contrasts
    from pymicroglia.measure.pool import pool
    from pymicroglia.measure.run import run
    folder = Path(folder)
    path = fixture_copy(folder, keep_module_blocks=True)
    data = json.loads(path.read_text(encoding='utf-8'))
    for module, settings in MOTION_MODULE_SETTINGS.items():
        data.setdefault('modules', {})[module] = {**settings, **data['modules'].get(module, {})}
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    previous = {k: os.environ.get(k) for k in ('PYMICROGLIA_STORE', 'PYMICROGLIA_INDEX')}
    os.environ['PYMICROGLIA_STORE'] = str(folder / 'cache')
    os.environ['PYMICROGLIA_INDEX'] = str(folder / 'index')
    try:
        manifest = run(load_config(path), folder / 'outputs', run_label='parity',
                       claim='the stage-01 fixture measured by the ported modules')
        pool(manifest['run']['folder'])
        contrasts(manifest['run']['folder'])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return Path(manifest['run']['folder']), manifest


def without_additions(table):
    """``table`` without the measurement added since Motion."""
    table = table[[c for c in table if not c.startswith(ADDED_METRIC)]]
    if 'metric' in table:
        table = table[table['metric'] != ADDED_METRIC]
    return table.reset_index(drop=True)


def measured_tables(run):
    """The measured-run tables in ``MEASURED``, as written and without the additions."""
    from tests.test_motion_parity import ported_path, read_as_written
    return {name: without_additions(read_as_written(ported_path(run, f'parity_A1/tables/{name}')))
            for name in MEASURED}


def _kept(rows, moved):
    return [{k: v for k, v in row.items() if k not in moved} for row in rows]


def measured_differences(run, *, ignore=()):
    """Every departure of a measured run from Motion, or from the baseline in ``MEASURED`` columns."""
    from tests.test_motion_parity import (FROZEN_RUN, _frame_differences, _row_differences, expected,
                                          ported_path, read_as_written, sha256)
    problems = []
    for rel, record in expected()['measure']['tables'].items():
        if rel.startswith(ignore):
            continue
        path = ported_path(run, rel)
        if not path.exists():
            problems.append(f'missing {rel}')
            continue
        if sha256(path) == record['sha256']:
            continue
        name = rel.split('/')[-1]
        moved = list(MEASURED.get(name, ()))
        frame = without_additions(read_as_written(path))
        if moved:
            problems.extend(f'{rel}: {p}' for p in differences(f'measured_{name}', frame, columns=moved))
        frame = frame.drop(columns=moved, errors='ignore')
        if (FROZEN_RUN / rel).exists():
            frozen = read_as_written(FROZEN_RUN / rel).drop(columns=moved, errors='ignore')
            problems.extend(_frame_differences(frame, frozen, rel))
        else:
            record = {**record, 'columns': [c for c in record['columns'] if c not in moved],
                      'head': _kept(record.get('head', []), moved), 'tail': _kept(record.get('tail', []), moved)}
            problems.extend(_row_differences(frame, record, rel))
    return problems


def as_written(table):
    """A table as its CSV reads back, every cell a string."""
    return pd.read_csv(StringIO(table.to_csv(index=False)), dtype=str, keep_default_na=False)


def read(path):
    return pd.read_csv(StringIO(gzip.decompress(path.read_bytes()).decode()), dtype=str, keep_default_na=False)


#: Fields of an embedded run record that describe the machine or the code
#: build rather than the analysis: where it ran, on what processor, with which
#: versions, and a hash of results that are themselves compared value by value.
MACHINE_FIELDS = frozenset({('cwd',), ('result_sha256',), ('environment', 'versions'),
                            ('environment', 'code_sha256'), ('environment', 'execution')})


def _parsed(cell):
    """A dict cell as data: run records are JSON, diagnostics are Python literals."""
    if not cell.startswith('{'):
        return None
    try:
        return json.loads(cell)
    except ValueError:
        try:
            return ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return None


def _agrees(actual, recorded, path=()):
    """``actual`` keeps every recorded field and value (numbers to nine figures).

    Fields a later Workbench adds are allowed; a recorded setting that changed
    or disappeared is not.
    """
    if isinstance(recorded, dict):
        return isinstance(actual, dict) and all(
            k in actual and _agrees(actual[k], v, path + (k,))
            for k, v in recorded.items() if path + (k,) not in MACHINE_FIELDS and k != 'thread_environment')
    if isinstance(recorded, list):
        return (isinstance(actual, list) and len(actual) == len(recorded)
                and all(_agrees(a, b, path) for a, b in zip(actual, recorded)))
    if isinstance(recorded, str) and isinstance(actual, str):
        a, b = _parsed(actual), _parsed(recorded)
        return actual == recorded if a is None or b is None else _agrees(a, b)
    if isinstance(recorded, float) or isinstance(actual, float):
        from tests.test_motion_parity import _same_value
        return not isinstance(actual, (bool, str)) and _same_value(repr(actual), repr(recorded))
    return actual == recorded


def _same(column, actual, recorded):
    from tests.test_motion_parity import _same_cell
    if _same_cell(column, actual, recorded):
        return True
    a, b = _parsed(actual), _parsed(recorded)
    return a is not None and b is not None and _agrees(a, b)


def differences(name, table, *, columns=None):
    """Cells of ``table`` that depart from the recorded Workbench baseline."""
    import circadian_workbench
    actual, frozen = as_written(table), read(BASELINE / f'{name}.gz')
    if list(actual) != list(frozen) or len(actual) != len(frozen):
        found = [f'{name}: shape or columns differ from the Workbench baseline']
    else:
        found = [f'{name} row {i} {c}: {a!r} != recorded {b!r}' for c in (columns or frozen)
                 for i, (a, b) in enumerate(zip(actual[c], frozen[c])) if not _same(c, a, b)]
    recorded = json.loads(RECORD.read_text(encoding='utf-8'))['circadian_workbench']
    if found and circadian_workbench.__version__ != recorded:
        found.insert(0, f'Circadian Workbench {circadian_workbench.__version__} is installed; the baseline was '
                        f'recorded on {recorded}. If the change is intended, re-run {Path(__file__).name}.')
    return found[:50]


def motion_differences(actual, frozen, moved=WORKBENCH_COLUMNS):
    """Cells outside the Workbench outputs (``moved``) that depart from Motion."""
    from tests.test_motion_parity import _same_cell
    actual = as_written(actual)
    if list(actual) != list(frozen) or len(actual) != len(frozen):
        return ['shape or columns differ from Motion']
    return [f'row {i} {c}: {a!r} != Motion {b!r}' for c in frozen if c not in moved
            for i, (a, b) in enumerate(zip(actual[c], frozen[c])) if not _same_cell(c, a, b)][:50]


def _scrubbed(table):
    """Replace the machine's working folder in embedded run records; comparisons ignore ``cwd``."""
    table = table.copy()
    for column in (c for c in table if c.endswith('_run_record_json')):
        def scrub(cell):
            if not isinstance(cell, str) or not cell.startswith('{'):
                return cell
            record = json.loads(cell)
            if 'cwd' in record:
                record['cwd'] = '<pymicroglia>'
            return json.dumps(record)
        table[column] = table[column].map(scrub)
    return table


GROUPS = ('spatial', 'tissue', 'figure', 'measured')


def main(groups=GROUPS):
    import circadian_workbench
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise SystemExit(f'unknown groups {sorted(unknown)}; choose from {GROUPS}')
    kept = []
    if set(groups) != set(GROUPS) and RECORD.exists():
        previous = json.loads(RECORD.read_text(encoding='utf-8'))
        if previous['circadian_workbench'] != circadian_workbench.__version__:
            raise SystemExit(f"the baseline was recorded on Circadian Workbench {previous['circadian_workbench']}; "
                             'record every group on the installed one')
        kept = previous['tables']
    BASELINE.mkdir(exist_ok=True)
    tables = {}
    if 'spatial' in groups:
        tables.update({f'{key}.csv': spatial_table(key) for key in SPATIAL})
    if 'tissue' in groups:
        tissue = tissue_tables()
        tables.update({f'tissue_{name}': tissue[name] for name in TISSUE})
    if 'figure' in groups:
        tables.update({f'figure_{key}.csv': figure_table(key) for key in FIGURES})
    if 'measured' in groups:
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
            run, _ = measured_run(scratch)
            tables.update({f'measured_{name}': table for name, table in measured_tables(run).items()})
    for name, table in tables.items():
        (BASELINE / f'{name}.gz').write_bytes(gzip.compress(_scrubbed(table).to_csv(index=False).encode(), mtime=0))
    RECORD.write_text(json.dumps({
        'circadian_workbench': circadian_workbench.__version__,
        'settings': MOTION_SETTINGS,
        'tables': sorted(set(kept) | {f'{name}.gz' for name in tables}),
        'why': __doc__.split('\n\n')[1].replace('\n', ' '),
    }, indent=2) + '\n', encoding='utf-8')
    print(f'recorded {len(tables)} tables on Circadian Workbench {circadian_workbench.__version__}')


if __name__ == '__main__':
    sys.path.insert(0, str(HERE.parents[1]))  # measure_stubs
    sys.path.insert(0, str(HERE.parents[2] / 'src'))
    sys.path.insert(0, str(HERE.parents[2]))
    main(tuple(sys.argv[1:]) or GROUPS)
