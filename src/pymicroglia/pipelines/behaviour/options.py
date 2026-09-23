"""Declare one shared behaviour-state vocabulary and its independent validation.

Parsing and resolution perform no fitting. Model-specific capabilities belong to
their producers; unavailable algorithms cannot silently become legacy regimes.
"""
from __future__ import annotations
from pymicroglia._sources import source_file

from dataclasses import dataclass
from pathlib import Path
import re

from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, InputIdentity, Measurement, PipelineRecipe, Record, Settings, StepResult, StepSpec, cell_number, content_id, text_key
from pymicroglia.pipelines.relationships.options import _number, _range
from pymicroglia.pipelines.rhythm.discovery import MeasurementChoice, _catalogue, _choices, _known_keys, _object, _population, _resolve_measurement, _shape_problem


def _exact(value, fields, where):
    value = _object(value, where)
    _known_keys(value, set(fields), where)
    if set(fields) - value.keys(): raise ValueError(where + ' requires ' + ', '.join(sorted(set(fields) - value.keys())))
    return value


def _fraction(value, where, *, inclusive=True):
    value = _number(value, where, minimum=0)
    if value > 1 or (not inclusive and value == 1): raise ValueError(where + ' must be between zero and one')
    return value


def _seed(value, where):
    value = _number(value, where, minimum=0, integer=True)
    if value >= 2**32: raise ValueError(where + ' must be less than 2**32')
    return value


@dataclass(frozen=True)
class ObservationKey(Record):
    cell: CellKey
    frame_index: int
    hours: float

    def __post_init__(self):
        object.__setattr__(self, 'frame_index', _number(self.frame_index, 'frame_index', minimum=0, integer=True))
        object.__setattr__(self, 'hours', _number(self.hours, 'hours'))


@dataclass(frozen=True)
class StateKey(Record):
    model_id: str
    component: int

    def __post_init__(self):
        text_key(self.model_id, 'model_id')
        object.__setattr__(self, 'component', _number(self.component, 'component', minimum=0, integer=True))


@dataclass(frozen=True)
class BoutKey(Record):
    state: StateKey
    first: ObservationKey
    last: ObservationKey

    def __post_init__(self):
        if self.first.cell != self.last.cell: raise ValueError('A bout cannot cross cells or recordings')
        if self.last.hours < self.first.hours or self.last.frame_index < self.first.frame_index:
            raise ValueError('Bout observations must follow original time order')


@dataclass(frozen=True)
class BehaviourRequest(Record):
    name: str
    declaration: Settings
    features: tuple[MeasurementChoice, ...]
    observation: Settings
    representation: str
    detrending: Settings
    learning: Settings
    candidates: tuple[Settings, ...]
    validation: Settings
    support: Settings
    assignment: Settings
    statistics: Settings
    biological_samples: Settings
    conditions: Settings
    table_grains: Settings
    time_range_hours: tuple[float, float] | None = None
    cells: tuple[tuple[str, int], ...] | None = None
    pipeline: str = 'cell-behaviour-states'

    @classmethod
    def from_dict(cls, block, groups, *, where='pipeline'):
        allowed = {'pipeline', 'name', 'features', 'observation', 'representation', 'detrending', 'learning',
            'candidates', 'validation', 'support', 'assignment', 'statistics', 'biological_samples', 'conditions',
            'table_grains', 'time_range_hours', 'cells'}
        _known_keys(block, allowed, where)
        if block.get('pipeline') != 'cell-behaviour-states': raise ValueError(where + ': expected cell-behaviour-states')
        name = text_key(block.get('name', 'cell-behaviour-states'), where + '.name')
        if re.fullmatch(r'[a-z][a-z0-9_-]*', name) is None: raise ValueError(where + '.name must be a lower-case identifier')
        features = _choices(block.get('features'), groups, where + '.features')
        if not features: raise ValueError('features requires at least one measured column')
        metadata = {'subject', 'condition', 'sample', 'biological_sample', 'cell_id', 'movie', 'source_run'}
        for feature in features:
            if feature.summary is not None: raise ValueError('State features cannot repeat whole-recording summaries over time')
            if feature.column in metadata: raise ValueError(feature.column + ' is design metadata, not a model feature')
        observation = _exact(block.get('observation'), {'kind'}, 'observation')
        if observation['kind'] != 'frame': raise ValueError('observation.kind currently requires frame; no implicit temporal windows')
        representation = block.get('representation')
        if representation not in {'raw', 'detrended'}: raise ValueError('representation must explicitly be raw or detrended')
        detrending = _object(block.get('detrending', {}), 'detrending')
        if representation == 'raw' and detrending: raise ValueError('raw representation cannot request detrending')
        if representation == 'detrended': text_key(detrending.get('detrend'), 'detrending.detrend')
        learning = _exact(block.get('learning'), {'balance', 'max_observations_per_cell', 'min_observations',
            'seed', 'scaling', 'missing'}, 'learning')
        if learning['balance'] not in {'sample_cell', 'cell'}: raise ValueError('learning.balance must be sample_cell or cell')
        _number(learning['max_observations_per_cell'], 'learning.max_observations_per_cell', minimum=2, integer=True)
        _number(learning['min_observations'], 'learning.min_observations', minimum=2, integer=True)
        _seed(learning['seed'], 'learning.seed')
        scaling = _exact(learning['scaling'], {'method'}, 'learning.scaling')
        if scaling['method'] not in {'standard', 'robust', 'none'}: raise ValueError('learning.scaling.method must be standard, robust or none')
        missing = _exact(learning['missing'], {'method', 'max_fraction'}, 'learning.missing')
        if missing['method'] not in {'complete_case', 'median'}: raise ValueError('learning.missing.method must be complete_case or median')
        fraction = _fraction(missing['max_fraction'], 'learning.missing.max_fraction', inclusive=False)
        if missing['method'] == 'complete_case' and fraction != 0: raise ValueError('complete_case requires max_fraction=0')
        candidates = block.get('candidates')
        if not isinstance(candidates, list) or not candidates: raise ValueError('candidates requires explicit model settings')
        models = []
        for candidate in candidates:
            candidate = _object(candidate, 'candidate'); text_key(candidate.get('method'), 'candidate.method')
            settings = Settings(candidate)
            if settings not in models: models.append(settings)
        validation = _exact(block.get('validation'), {'unit', 'split', 'min_groups', 'independence_justification'}, 'validation')
        if validation['unit'] not in {'biological_sample', 'recording'}: raise ValueError('validation.unit must be biological_sample or recording')
        text_key(validation['independence_justification'], 'validation.independence_justification')
        split = _object(validation['split'], 'validation.split')
        if split.get('method') == 'explicit':
            _exact(split, {'method', 'roles'}, 'validation.split')
            roles = _object(split['roles'], 'validation.split.roles')
            if not roles: raise ValueError('Explicit validation roles cannot be empty')
            for group, role in roles.items():
                text_key(group, 'validation group')
                if role not in {'learning', 'development', 'confirmation', 'assignment_only'}:
                    raise ValueError('Unknown validation role: ' + str(role))
        elif split.get('method') == 'group_shuffle':
            _exact(split, {'method', 'development_fraction', 'confirmation_fraction', 'seed'}, 'validation.split')
            fractions = [_fraction(split[k], 'validation.split.' + k, inclusive=False) for k in ('development_fraction', 'confirmation_fraction')]
            if any(value <= 0 for value in fractions) or sum(fractions) >= 1:
                raise ValueError('Development and confirmation fractions must be positive and leave learning groups')
            _seed(split['seed'], 'validation.split.seed')
        else: raise ValueError('validation.split.method must be explicit or group_shuffle')
        counts = _exact(validation['min_groups'], {'learning', 'development', 'confirmation'}, 'validation.min_groups')
        for role, count in counts.items(): _number(count, 'validation.min_groups.' + role, minimum=1, integer=True)
        support = _object(block.get('support'), 'support'); text_key(support.get('method'), 'support.method')
        assignment = _exact(block.get('assignment'), {'min_probability', 'outlier_quantile', 'min_observed_fraction'}, 'assignment')
        _fraction(assignment['min_probability'], 'assignment.min_probability')
        _fraction(assignment['outlier_quantile'], 'assignment.outlier_quantile', inclusive=False)
        if _fraction(assignment['min_observed_fraction'], 'assignment.min_observed_fraction') <= 0:
            raise ValueError('assignment.min_observed_fraction must be positive')
        statistics = _exact(block.get('statistics'), {'interval_rule', 'max_gap_hours', 'transition_interval_hours',
            'sample_aggregation', 'comparison'}, 'statistics')
        if statistics['interval_rule'] != 'adjacent_midpoint': raise ValueError('statistics.interval_rule requires adjacent_midpoint')
        _number(statistics['max_gap_hours'], 'statistics.max_gap_hours', positive=True)
        bounds = _range(statistics['transition_interval_hours'], 'statistics.transition_interval_hours')
        if bounds[0] < 0 or bounds[1] > statistics['max_gap_hours']:
            raise ValueError('Transition interval bounds must be nonnegative and within max_gap_hours')
        if statistics['sample_aggregation'] not in {'mean', 'pooled_exposure'}:
            raise ValueError('statistics.sample_aggregation must be mean or pooled_exposure')
        comparison = _object(statistics['comparison'], 'statistics.comparison')
        text_key(comparison.get('method'), 'statistics.comparison.method')
        samples, conditions = [_object(block.get(key, {}), key) for key in ('biological_samples', 'conditions')]
        for mapping, where in ((samples, 'biological_samples'), (conditions, 'conditions')):
            for key, value in mapping.items(): text_key(key, where); text_key(value, where + '.' + key)
        bounds = block.get('time_range_hours')
        bounds = None if bounds is None else tuple(_range(bounds, 'time_range_hours'))
        cells = None
        if block.get('cells') is not None:
            if not isinstance(block['cells'], list): raise ValueError('cells must be a list of movie/identity objects')
            cells = []
            for cell in block['cells']:
                _exact(cell, {'movie', 'identity'}, 'cells')
                key = (text_key(cell['movie'], 'cells.movie'), cell_number(cell['identity']))
                if key not in cells: cells.append(key)
        return cls(name, Settings(block), features, Settings(observation), representation, Settings(detrending),
            Settings(learning), tuple(models), Settings(validation), Settings(support), Settings(assignment),
            Settings(statistics), Settings(samples), Settings(conditions), Settings(_object(block.get('table_grains', {}), 'table_grains')),
            bounds, None if cells is None else tuple(cells))


@dataclass(frozen=True)
class ResolvedBehaviourRequest(Record):
    request: BehaviourRequest
    inputs: InputIdentity
    features: tuple[Measurement, ...]
    detrending: Settings
    processing_version: str | None

    @property
    def scientific_id(self):
        request = self.request.as_dict(); request.pop('name'); request.pop('declaration')
        return content_id({'request': request, 'inputs': self.inputs, 'features': self.features,
            'detrending': self.detrending, 'processing_version': self.processing_version})


def resolve_request(request, *, source_run, tables, input_hashes, rhythm_params=None, table_grains=None):
    import pandas as pd
    text_key(source_run, 'source_run')
    if any(not isinstance(name, str) or not isinstance(frame, pd.DataFrame) for name, frame in tables.items()):
        raise ValueError('tables must map names to measured DataFrames')
    declared_grains = request.table_grains.as_dict()
    for name, grain in (table_grains or {}).items():
        if name in declared_grains and tuple(grain) != tuple(declared_grains[name]): raise ValueError('Conflicting table grain: ' + name)
        declared_grains[name] = grain
    declarations, grains = _catalogue(declared_grains)
    features = tuple(_resolve_measurement(choice, tables, declarations, grains, testing=True) for choice in request.features)
    names = {feature.table for feature in features}
    for feature in features:
        if set(feature.grain) - {'stem', 'identity', 'frame_index', 'hours'}:
            raise ValueError(feature.column + ': select an unambiguous per-cell observation table')
        if 'frame_index' not in tables[feature.table]: raise ValueError(feature.column + ': frame observations require frame_index')
    if 'cell_summary' in tables:
        problem = _shape_problem(tables['cell_summary'], grains['cell_summary'], summary=None, testing=False)
        if problem: raise ValueError('cell_summary population: ' + problem)
        names.add('cell_summary')
    cells, samples = _population(source_run, tables, sorted(names), request)
    known_movies = {cell.movie for cell in cells}
    if set(request.conditions) - known_movies: raise ValueError('conditions names unselected or unavailable recordings')
    by_sample = {}
    for sample in samples:
        condition = request.conditions.get(sample.movie)
        if sample.confirmed and condition is not None: by_sample.setdefault(sample.sample, set()).add(condition)
    if any(len(values) > 1 for values in by_sample.values()):
        raise ValueError('A biological sample has conflicting conditions; repeated-measure comparisons need a separate explicit design')
    fingerprints = {}
    for name in sorted(names):
        fingerprint = input_hashes.get(name)
        if not isinstance(fingerprint, str) or re.fullmatch(r'[a-fA-F0-9]{64}', fingerprint) is None:
            raise ValueError('input_hashes.' + name + ': verified SHA-256 is required')
        fingerprints[name] = fingerprint.lower()
    detrending, version = Settings(), None
    if request.representation == 'detrended':
        import pymicroglia.workbench as circadian
        _known_keys(request.detrending, set(circadian.DETREND_DEFAULTS), 'detrending')
        detrending = Settings(circadian.detrend_settings(request.detrending.as_dict())); version = circadian.WORKBENCH_VERSION
    return ResolvedBehaviourRequest(request, InputIdentity(source_run, Settings(fingerprints), cells, samples), features, detrending, version)


ACCEPTED = 'state-support:accepted-model'
RECIPE = PipelineRecipe('cell-behaviour-states', 1, (
    StepSpec('behaviour-design', 'behaviour-design', inputs=('measured-tables',)),
    StepSpec('feature-inputs', 'feature-inputs', ('behaviour-design',)),
    StepSpec('candidate-models', 'candidate-models', ('feature-inputs',)),
    StepSpec('state-support', 'state-support', ('feature-inputs', 'candidate-models')),
    StepSpec('state-assignments', 'state-assignments', ('feature-inputs', 'candidate-models', 'state-support'), selection=ACCEPTED, requires_selected_rows=True),
    StepSpec('durations-and-switches', 'durations-and-switches', ('state-support', 'state-assignments'), selection=ACCEPTED, requires_selected_rows=True),
    StepSpec('state-sample-comparisons', 'state-sample-comparisons', ('state-support', 'durations-and-switches'), selection=ACCEPTED, requires_selected_rows=True),
    # Diagnostics must still run when the complete scientific decision is no supported states.
    StepSpec('state-support-figures', 'state-support-figures', ('feature-inputs', 'candidate-models', 'state-support'), kind='render'),
    StepSpec('state-profile-figures', 'state-profile-figures', ('state-support', 'state-assignments'), selection=ACCEPTED, requires_selected_rows=True, kind='render'),
    StepSpec('state-timelines', 'state-timelines', ('state-support', 'state-assignments', 'durations-and-switches'), selection=ACCEPTED, requires_selected_rows=True, kind='render'),
    StepSpec('state-switching-figures', 'state-switching-figures', ('state-support', 'state-assignments', 'durations-and-switches', 'state-sample-comparisons'), selection=ACCEPTED, requires_selected_rows=True, kind='render'),
    StepSpec('state-report-cards', 'state-report-cards', ('feature-inputs', 'state-support', 'state-assignments', 'durations-and-switches'), selection=ACCEPTED, requires_selected_rows=True, kind='render'),
    StepSpec('linked-results-index', 'linked-results-index', ('behaviour-design', 'feature-inputs', 'candidate-models', 'state-support', 'state-assignments',
        'durations-and-switches', 'state-sample-comparisons', 'state-support-figures', 'state-profile-figures', 'state-timelines', 'state-switching-figures', 'state-report-cards'), kind='render')))


def identity(context):
    from pymicroglia.pipelines._screening import file_hash, read_verified_tables
    read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    return content_id({'request': context.request.scientific_id, 'code': file_hash(Path(__file__))})


def freeze_design(context):
    from pymicroglia.pipelines._screening import file_hash, _write_json
    context.output.mkdir(parents=True)
    path = context.output / 'behaviour_design.json'
    _write_json(path, {'schema_version': 1, 'scientific_id': context.scientific_id, 'request': context.request.as_dict(),
        'scientific_evaluation': False, 'state_meaning': 'Model-scoped categorical components; separate from legacy regimes',
        'support_required_before_assignments': True, 'validation_roles_precede_learned_transformations': True})
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Frozen shared state definition, observations and validation design',
        (ArtifactRef('behaviour_design', path.name, file_hash(path), context.scientific_id),))


def _pending(context):
    from pymicroglia.pipelines._runner import Unavailable
    raise Unavailable('Required ' + context.step.name + ' producer is awaiting its implementation stage')


def feature_settings(resolved):
    """Only choices that change prepared observations, groups or eligibility."""
    request=resolved.request
    return {'inputs':resolved.inputs.as_dict(),'features':[feature.as_dict() for feature in resolved.features],
        'observation':request.observation.as_dict(),'representation':request.representation,'detrending':resolved.detrending.as_dict(),
        'processing_version':resolved.processing_version,'learning':request.learning.as_dict(),'validation':request.validation.as_dict(),
        'conditions':request.conditions.as_dict(),'time_range_hours':request.time_range_hours,
        'max_gap_hours':request.statistics['max_gap_hours'],'assignment_min_observed_fraction':request.assignment['min_observed_fraction']}


def time_settings(statistics):
    """Observed-time choices, separate from downstream sample inference."""
    return {name:statistics[name] for name in ['interval_rule','max_gap_hours','transition_interval_hours']}


def scoped_identity(implementation):
    """Keep unaffected science reusable when a downstream question changes.

    Prepared provenance retains its original full request. A later request is
    recorded by the design and final index; it does not rewrite that snapshot.
    """
    def identity_for_step(context):
        from pymicroglia.pipelines._screening import file_hash
        request=context.request.request;step=context.step.name
        dependencies={name:{'scientific_id':saved.outcome.scientific_id,'artifacts':[ref.as_dict() for ref in saved.outcome.artifacts],
            'selections':[selection.record_id for selection in saved.outcome.selections]} for name,saved in context.dependencies.items()}
        settings={}
        if step=='feature-inputs':
            settings=feature_settings(context.request);dependencies={}
        elif step=='candidate-models':
            settings={'candidates':[candidate.as_dict() for candidate in request.candidates],'learning':request.learning.as_dict(),
                'assignment':request.assignment.as_dict(),'representation':request.representation,'detrending':context.request.detrending.as_dict(),
                'features':[feature.as_dict() for feature in context.request.features]}
        elif step=='state-support':settings={'support':request.support.as_dict(),'validation':request.validation.as_dict()}
        elif step=='state-assignments':settings={'assignment':request.assignment.as_dict()}
        elif step=='durations-and-switches':settings={'time':time_settings(request.statistics),'time_range_hours':request.time_range_hours}
        elif step=='state-sample-comparisons':settings={'aggregation':request.statistics['sample_aggregation'],'comparison':request.statistics['comparison']}
        else:raise ValueError('No scoped scientific identity for '+step)
        return content_id({'recipe':RECIPE.name,'recipe_version':RECIPE.version,'step':step,'settings':settings,'dependencies':dependencies,
            'selection':context.selection.record_id if context.selection else None,'implementation':implementation,
            'identity_code':file_hash(Path(__file__)),'runner':file_hash(source_file('runner.py'))})
    return identity_for_step


def producers():
    from pymicroglia.pipelines._runner import Producer
    import pymicroglia.pipelines.behaviour.inputs as behaviour_inputs
    import pymicroglia.pipelines.behaviour.candidates as behaviour_candidates
    import pymicroglia.pipelines.behaviour.validation as behaviour_validation
    import pymicroglia.pipelines.behaviour.assignments as behaviour_assignments
    import pymicroglia.pipelines.behaviour.durations as behaviour_durations
    import pymicroglia.pipelines.behaviour.samples as behaviour_samples
    import pymicroglia.pipelines.behaviour.figures as behaviour_figures
    import pymicroglia.pipelines.behaviour.timeline_figures as behaviour_timeline_figures
    import pymicroglia.pipelines.behaviour.summary_figures as behaviour_summary_figures
    import pymicroglia.pipelines.behaviour.card_figures as behaviour_card_figures
    import pymicroglia.pipelines.behaviour.index as behaviour_index
    modules={'feature-inputs':behaviour_inputs,'candidate-models':behaviour_candidates,'state-support':behaviour_validation,
        'state-assignments':behaviour_assignments,'durations-and-switches':behaviour_durations,'state-sample-comparisons':behaviour_samples}
    scientific={}
    for name,module in modules.items():
        implementation=module.implementation_version()
        scientific[name]=Producer(module.produce,identity=scoped_identity(implementation),version=implementation)
    return {**scientific, 'behaviour-design': Producer(freeze_design, identity),
        'state-support-figures': Producer(behaviour_figures.produce_support, version=behaviour_figures.version(), accepts_unavailable_dependencies=True),
        'state-profile-figures': Producer(behaviour_figures.produce_profiles, version=behaviour_figures.version()),
        'state-timelines': Producer(behaviour_timeline_figures.produce, version=behaviour_timeline_figures.version()),
        'state-switching-figures': Producer(behaviour_summary_figures.produce, version=behaviour_summary_figures.version()),
        'state-report-cards': Producer(behaviour_card_figures.produce, version=behaviour_card_figures.version()),
        'linked-results-index': Producer(behaviour_index.produce, version=behaviour_index.version(), accepts_unavailable_dependencies=True)}


def run_request(resolved, table_paths, output, *, presentation=None, only=None):
    from pymicroglia.pipelines._runner import run_pipeline
    return run_pipeline(RECIPE, producers(), request=resolved, scientific_settings={'behaviour': resolved.scientific_id},
        table_paths=table_paths, output=output, presentation=presentation, only=only)
