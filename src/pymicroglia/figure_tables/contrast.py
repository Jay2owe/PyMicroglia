"""Prepare saved contrast results without testing or correcting them again."""
import numpy as np
import pandas as pd
from .prepared import PreparedViews
from ..visualisation.labels import describe,documented
COLUMNS=['contrast','family','metric','unit','aggregate','group_a','group_b','n_a','n_b','test','statistic','p_value','effect','effect_kind','effect_lo','effect_hi','correction','p_corrected','alpha','significant','refusal']

def nothing_declared(absent):
    table=pd.DataFrame(columns=COLUMNS)
    note='No comparisons were declared.' if absent else 'No declared comparison produced a row.'
    return PreparedViews({key:dict(table=table,message=note) for key in ('effects','correction','design')},wording=dict(title='Declared comparisons',subtitle=note)),None

ORDERS = {'family': 'declaration order, families kept together', 'effect': 'largest effect first', 'significance': 'surviving comparisons first'}

EFFECT_LABELS = {'median_difference': 'Difference in medians, second group minus first', 'median_paired_difference': 'Median of the per-unit differences', 'mean_difference': 'Difference in means, second group minus first', 'hedges_g': 'Standardised difference (Hedges g)', 'median_against_zero': 'Median across units, against zero', 'hedges_g_paired': 'Standardised per-unit difference (Hedges g)', 'epsilon_squared': 'Share of the variation explained (epsilon squared)', 'eta_squared': 'Share of the variation explained (eta squared)'}

def _effect_label(kind: str) -> str:
    return EFFECT_LABELS.get(kind, kind.replace('_', ' ').capitalize() or 'Effect')

def _scale(kind: str, metric: str, interval_minutes: float | None=None, group_by: str='', group: str='') -> str:
    """The axis this row may be drawn on, as a sentence.

    Three things decide it. The kind says what the number is - a difference of
    medians, a standardised difference. The unit of the tested column says what
    that number is counted in. And where the grouping column is itself a
    measurement name, that measurement's unit is the other half of the count.

    The third is not a nicety. A long table puts the measurement in a column:
    `trend` and `window_change` both carry a `metric` column, so a contrast on
    one of them tests a single column - a slope, a change - across groups that
    are themselves measurements. The tested column then reads "per h" for every
    row, and a slope of an area sits on the same axis as a slope of a
    brightness five orders of magnitude away, where the smaller of them is
    drawn as a dot on the no-effect line and reads as no effect.
    """
    if not kind:
        return 'no effect recorded'
    unit = describe(str(metric)).unit_text(interval_minutes)
    measured = describe(str(group)).unit_text(interval_minutes) if group_by == 'metric' and documented(str(group)) else ''
    counted = chr(32).join((part for part in (measured, unit) if part))
    return f'{_effect_label(kind)}{chr(32)}({counted})' if counted else _effect_label(kind)

def _alpha(rows: pd.DataFrame) -> str:
    values = sorted(set(rows['alpha'].dropna()))
    return ', '.join((f'{value:g}' for value in values)) or 'unset'

def _corrections(rows: pd.DataFrame) -> str:
    pairs = rows.dropna(subset=['correction']).groupby('correction')['family'].nunique()
    return ', '.join((f"{name} across {count} famil{('y' if count == 1 else 'ies')}" for name, count in pairs.items())) or 'none declared'

def _units(rows: pd.DataFrame) -> str:
    return ', '.join(sorted(set(rows['unit'].dropna()))) or 'unset'

def prepare(source,options):
    statistics = source.table('statistics.csv',optional=True)
    if statistics is None or statistics.empty:
        return nothing_declared(statistics is None)
    order = str(options.get('order'))
    if order not in ORDERS:
        raise ValueError(f'--order {order} is not one this page has. It takes: ' + '; '.join((f'{name} ({meaning})' for name, meaning in ORDERS.items())))
    rows = statistics.copy()
    wanted = [str(name) for name in options.get('metrics')]
    if wanted:
        available = sorted((str(value) for value in rows['metric'].dropna().unique()))
        unknown = [name for name in wanted if name not in available]
        if unknown:
            raise ValueError(f"--metrics {','.join(unknown)} was not tested in this run. statistics.csv holds: {', '.join(available)}")
        rows = rows[rows['metric'].isin(wanted)]
    refused = rows['p_value'].isna()
    rows['refusal'] = np.where(refused, rows['note'].fillna('Refused, with no reason recorded'), '')
    rows['is_refusal'] = refused
    rows['label'] = rows['contrast'].astype(str) + ' - ' + rows['metric'].astype(str)
    rows['significant'] = rows['significant'].fillna(False).astype(bool)
    interval = source.interval
    rows['scale'] = [_scale(kind, metric, interval, group_by=group_by, group=group) for kind, metric, group_by, group in zip(rows['effect_kind'].fillna(''), rows['metric'].fillna(''), rows['group_by'].fillna(''), rows['group_a'].fillna(''))]
    tested = rows[~rows['is_refusal']]
    by_contrast = dict(zip(tested['contrast'], tested['scale']))
    fallback = next(iter(tested['scale']), 'no effect recorded')
    rows.loc[rows['is_refusal'], 'scale'] = [by_contrast.get(name, fallback) for name in rows.loc[rows['is_refusal'], 'contrast']]
    sort_key = {'family': ['scale', 'family', 'contrast', 'metric'], 'effect': ['scale', 'effect'], 'significance': ['scale', 'significant', 'effect']}[order]
    ascending = order == 'family'
    drawable = rows.sort_values(sort_key, ascending=ascending, kind='stable')
    figure_data=rows[COLUMNS]
    units=tested.dropna(subset=['n_a'])
    counts=pd.concat([units.n_a,units.n_b]).groupby(list(units.label)*2).min() if not units.empty else pd.Series(dtype=float)
    counts=counts.reindex(list(dict.fromkeys(units.label))).dropna()
    designs=counts.rename('units').rename_axis('label').reset_index()
    return PreparedViews({'effects':dict(table=figure_data,rows=drawable),
                          'correction':dict(table=tested),
                          'design':dict(table=designs,unit=_units(tested))},wording=dict(
        title=f'Declared comparisons: {len(tested)} tested, {int(refused.sum())} refused',
        subtitle=f'One row per declared comparison per metric, ordered by {ORDERS[order]}.',
        footnote=f'Values, intervals and significance are read from saved statistics. Corrections apply within each family: {_corrections(rows)}. Different effect scales are drawn on separate axes; refusal reasons remain visible.')),figure_data
