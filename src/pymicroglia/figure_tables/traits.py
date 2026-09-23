"""Compare recorded within-cell spread with between-cell spread by condition."""
import numpy as np
import pandas as pd
from ..visualisation.labels import describe
from .._results import document,read_document
from .prepared import PreparedViews
TRAIT_MAX=.5
STATE_MIN=.7


def design_for(source):
    path=document(source.run/'conditions.json')
    if not path.exists():return source.manifest.get('design', {})
    source.sources.append(path)
    return read_document(path)

def prepare(source,options):
    metrics = options.get('metrics')
    requested_hues = options.get('hues')
    if not metrics:
        raise ValueError('--metrics needs at least one measurement to compare')
    summary = source.table('cell_summary.csv',scope='run')
    missing = [column for column in metrics if f'{column}_median' not in summary.columns or f'{column}_iqr' not in summary.columns]
    if missing:
        available = sorted((name[:-len('_median')] for name in summary.columns if name.endswith('_median') and f"{name[:-len('_median')]}_iqr" in summary.columns))
        raise ValueError(f"--metrics names {', '.join(missing)}, which pooled cell_summary.csv has no median and interquartile range for.\nMeasurements available: " + ', '.join(available))
    if 'condition' not in summary.columns:
        summary = summary.copy()
        summary['condition'] = 'all cells'
    summary['condition'] = summary['condition'].fillna('unassigned').astype(str)
    design = design_for(source)
    declarations = {str(item['name']): item for item in design.get('conditions', []) if item.get('name') is not None}
    observed = list(dict.fromkeys(summary['condition'].tolist()))
    condition_order = [name for name in declarations if name in observed]
    condition_order.extend(sorted((name for name in observed if name not in declarations)))
    if not condition_order:
        raise ValueError('pooled cell_summary.csv has no condition rows to draw')
    condition_labels = {name: str(declarations.get(name, {}).get('label') or name.replace('_', ' ').title()) for name in condition_order}
    if requested_hues and len(requested_hues) != len(condition_order):
        raise ValueError(f"--hues needs one colour for each displayed condition ({len(condition_order)}: {', '.join(condition_order)}); got {len(requested_hues)}")
    colours=requested_hues or ['teal','orange','blue','plum','circadian_green'][:len(condition_order)]
    if len(colours)<len(condition_order):colours=[['teal','orange','blue','plum','circadian_green'][i%5] for i in range(len(condition_order))]
    condition_colours = dict(zip(condition_order, colours))
    metric_descriptions = {column: describe(column) for column in metrics}
    rows = []
    for condition_index, condition in enumerate(condition_order):
        group = summary[summary['condition'] == condition]
        movie_count = int(group['stem'].nunique()) if 'stem' in group else 1
        for column in metrics:
            median_column = f'{column}_median'
            iqr_column = f'{column}_iqr'
            usable = group[[median_column, iqr_column]].dropna()
            medians = usable[median_column].to_numpy(float)
            within = usable[iqr_column].to_numpy(float)
            between = float(np.percentile(medians, 75) - np.percentile(medians, 25)) if len(medians) >= 2 else np.nan
            median_within = float(np.median(within)) if len(within) else np.nan
            ratio = float(median_within / between) if np.isfinite(between) and between > 0 else np.nan
            metric = metric_descriptions[column]
            rows.append({'condition': condition, 'condition_label': condition_labels[condition], 'condition_hue': condition_colours[condition], 'condition_order': condition_index, 'movies': movie_count, 'metric': column, 'label': metric.label, 'unit': metric.unit_text(source.interval), 'cells': int(len(usable)), 'between_cell_iqr_of_medians': between, 'median_within_cell_iqr': median_within, 'within_over_between': ratio})
    table = pd.DataFrame(rows)
    finite = table[np.isfinite(table['within_over_between'])]
    if finite.empty:
        raise ValueError('none of the requested measurements has a non-zero between-cell interquartile range in any condition')
    metric_order = finite.groupby('metric', sort=False)['within_over_between'].median().sort_values(kind='stable').index.tolist()
    metric_order.extend((column for column in metrics if column not in metric_order))
    metric_positions = {column: index for index, column in enumerate(metric_order)}
    table['metric_order'] = table['metric'].map(metric_positions).astype(int)
    table['classification'] = np.where(table['within_over_between'] <= TRAIT_MAX, 'trait-like', np.where(table['within_over_between'] >= STATE_MIN, 'state-like', 'intermediate'))
    table = table.sort_values(['condition_order', 'metric_order'], kind='stable').reset_index(drop=True)
    return PreparedViews({'ranking':dict(table=table,upper=max(1.5,np.ceil(float(finite.within_over_between.max())*1.22*2)/2))},wording=dict(title='Within-cell and between-cell variability by condition',footnote='Ratios compare median within-cell interquartile range with the interquartile range of cell medians. Trait-like and state-like cutoffs are descriptive guides, not biological tests.'+(' Synthetic software-test groups.' if design.get('synthetic') else ''))),None
