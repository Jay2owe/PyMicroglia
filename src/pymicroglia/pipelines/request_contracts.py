"""Readable request fields for the six packaged workflows; parsers remain authoritative."""
from ._requests import FAMILIES

FIELDS = {
 'rhythm_discovery': 'pipeline name test_measurements comparison_measurements pairs analysis_options biological_samples cells correction_scope settings_profile group_comparisons detection_agreement timing timing_summary',
 'method_audit': 'pipeline name measurements analysis_options candidates population biological_samples correction_scope benchmark_design score_policy stability recommendation_scope',
 'measurement_relationships': 'pipeline name measurements pairs within_cell lag between_cells support time_range_hours representation detrending inference sample_summary biological_samples cells table_grains',
 'behaviour_states': 'pipeline name features observation representation detrending learning candidates validation support assignment statistics biological_samples conditions table_grains time_range_hours cells',
 'spatial_coordination': 'pipeline name reference_measurements target_measurements pairs questions geometry support representation increment detrending shared_reference inference sample_summary biological_samples conditions table_grains time_range_hours cells',
 'intervention_response': 'pipeline name measurements summary anchors windows recording_windows support evidence relative_effects meaningful_effects inference controls timing rhythms coordinated biological_samples conditions matching table_grains cells',
}
MEANINGS = {
 'pipeline': ('str','Workflow family; the public action supplies this value if omitted.'),
 'name': ('str','Result-folder identifier; defaults to the workflow family.'),
 'test_measurements': ('list','Measured time-series columns to estimate and test independently.'),
 'comparison_measurements': ('list','Measured scalar summaries for comparisons with saved rhythm results.'),
 'measurements': ('list','Declared measured columns, optionally with explicit source table and summary.'),
 'reference_measurements': ('list','Measured columns at the reference endpoint of each cell pair.'),
 'target_measurements': ('list','Measured columns at the target endpoint of each cell pair.'),
 'features': ('list','Per-frame measured features used to learn cell states; design metadata are excluded.'),
 'pairs': ('mapping','Explicit pair list or the workflow-specific pair-generation mode.'),
 'analysis_options': ('mapping','Separate estimator, significance test, search, detrending and sufficiency settings. Omitted choices inherit the measured run.'),
 'biological_samples': ('mapping','Movie name to independently identified biological sample; repeated recordings are not independent samples.'),
 'conditions': ('mapping','Movie name to declared experimental condition.'),
 'cells': ('list','Optional explicit movie/identity objects limiting the input population.'),
 'correction_scope': ('str','Declared multiple-testing family, independent of later display selections.'),
 'settings_profile': ('mapping','Explicit previously exported method-audit profile; no automatic profile adoption.'),
 'group_comparisons': ('mapping','Declared comparisons of retained cell measurements and independently identified groups.'),
 'detection_agreement': ('mapping','Agreement of separate significant/not-significant calls; this does not compare phase.'),
 'timing': ('mapping','Explicit timing question and its support/model settings; no default common period or daily clock.'),
 'timing_summary': ('mapping','Aggregation of eligible saved timing comparisons under the declared sample design.'),
 'candidates': ('list','Explicit candidate method/model settings; synthetic example values are not biological defaults.'),
 'population': ('mapping','Audit input selection: all, explicit cells, or a declared representative sampling design.'),
 'benchmark_design': ('mapping','Known-truth simulation scenarios, independent development/confirmation partitions and seeds.'),
 'score_policy': ('mapping','Declared evidence and performance thresholds for method recommendation.'),
 'stability': ('mapping','Declared sensitivity/stability checks around each candidate method.'),
 'recommendation_scope': ('str','Whether an audit recommends separately by measurement or for the whole dataset.'),
 'within_cell': ('mapping','Enabled same-time association question, statistic and explicit evidence model.'),
 'lag': ('mapping','Enabled lag-search question, physical search range/resolution and separate delay uncertainty.'),
 'between_cells': ('mapping','Association between cell summaries with explicit independent experimental unit.'),
 'support': ('mapping','Observation count, coverage, span and maximum-gap requirements; insufficient data remain explicit.'),
 'time_range_hours': ('list','Optional [start, end] bounds on the original recording clock, in hours.'),
 'representation': ('str','Explicit raw or detrended measurements; no silent preprocessing choice.'),
 'detrending': ('mapping','Workbench baseline-removal settings, used only for a declared detrended representation.'),
 'inference': ('mapping','Significance threshold, multiple-testing correction and full hypothesis-family policy.'),
 'sample_summary': ('mapping','Aggregation and evidence settings for explicitly identified biological samples.'),
 'table_grains': ('mapping','Table name to its declared row identity columns for custom measurement tables.'),
 'observation': ('mapping','State-learning observation definition; currently explicit frame observations.'),
 'learning': ('mapping','Training balance, scaling, missing-data policy, observation limits and seed.'),
 'validation': ('mapping','Independent training/validation split, unit, support and justification.'),
 'assignment': ('mapping','State-assignment support and unknown-state policy.'),
 'statistics': ('mapping','Declared state occupancy/switching comparisons and their independent-unit policy.'),
 'questions': ('mapping','Separately enabled characteristic, simultaneous, lagged, proximity, rhythm or state questions.'),
 'geometry': ('mapping','Measured coordinates, units and pair-distance/observation rules.'),
 'increment': ('str','Explicit increment representation when the chosen coordination question requires it.'),
 'shared_reference': ('mapping','Declared shared-reference adjustment, with its required measured source.'),
 'summary': ('str','Explicit window summary: mean, median, min or max; may instead be declared per measurement.'),
 'anchors': ('mapping','Movie to original-clock anchor hours, intervention/control kind and label.'),
 'windows': ('list','Named half-open windows with explicit relative-hours, recording-hours or frame coordinates.'),
 'recording_windows': ('mapping','Movie-specific declared windows overriding the shared design where requested.'),
 'evidence': ('mapping','Explicit response model, estimand, error model and scientific justifications.'),
 'relative_effects': ('mapping','Explicit relative-effect calculation and safe baseline requirements.'),
 'meaningful_effects': ('mapping','Scientifically declared effect-size thresholds, separate from significance.'),
 'controls': ('mapping','Enabled reference/target condition comparisons and independent sample design.'),
 'rhythms': ('mapping','Opt-in window rhythm analysis with separate estimation and significance settings.'),
 'coordinated': ('mapping','Enabled response-pattern comparisons with declared measurement pairs and sample aggregation.'),
 'matching': ('list','Unique match identifiers linking distinct reference and target biological samples.'),
}

def circadian_fields():
 from .. import workbench
 from ..visualisation.figures._vocabulary import meaning
 rows = {}
 for name, default in workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS.items():
  row = dict(meaning(name)); row['default'] = default
  rows[name] = row
 methods = workbench.call('period_methods').data['methods']
 rows['fit_method']['choices'] = [row['key'] for row in methods]
 rows['significance_method']['choices'] = [row['key'] for row in methods if row['gives_significance']]
 return rows

def contract(action):
 fields = {name:{'type':MEANINGS[name][0],'description':MEANINGS[name][1],
                 'units':'hours' if name=='time_range_hours' else '-'}
           for name in FIELDS[action].split()}
 fields['pipeline'].update(default=FAMILIES[action][0],choices=[FAMILIES[action][0]])
 fields['name']['default'] = FAMILIES[action][0]
 if 'analysis_options' in fields: fields['analysis_options']['properties']=circadian_fields()
 if action=='intervention_response':
  fields['rhythms']['default']={'enabled':False}
  fields['rhythms']['properties']={'analysis_options':{'type':'mapping','properties':circadian_fields()}}
 return fields
