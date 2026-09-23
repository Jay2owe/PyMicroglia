"""Declare spatial questions without pairing, testing or loading optional results.

Geometry determines the eligible scientific population. Report limits consume
saved evidence later and are deliberately absent from these request fields.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import re

from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, InputIdentity, Measurement, PipelineRecipe, Record, Settings, StepResult, StepSpec, cell_number, content_id, text_key
from pymicroglia.pipelines.relationships.options import _number, _range, _support, LAG_CONVENTION
from pymicroglia.pipelines.rhythm.discovery import MeasurementChoice, _catalogue, _choices, _known_keys, _object, _population, _resolve_measurement, _shape_problem


QUESTIONS = ('characteristics', 'simultaneous', 'delay', 'proximity', 'rhythm', 'states')


@dataclass(frozen=True)
class EndpointKey(Record):
    cell: CellKey
    measurement: Measurement


@dataclass(frozen=True)
class CoordinationPairKey(Record):
    reference: EndpointKey
    target: EndpointKey
    oriented: bool

    def __post_init__(self):
        a, b = self.reference.cell, self.target.cell
        if a == b: raise ValueError('Coordination requires distinct cells')
        if (a.source_run, a.movie) != (b.source_run, b.movie):
            raise ValueError('A coordination pair cannot cross recordings or source runs')
        if not isinstance(self.oriented, bool): raise ValueError('oriented must be boolean')
        if not self.oriented and (b, self.target.measurement.record_id) < (a, self.reference.measurement.record_id):
            first, second = self.reference, self.target
            object.__setattr__(self, 'reference', second)
            object.__setattr__(self, 'target', first)


def _enabled(value, where):
    block = _object(value, where)
    enabled = block.get('enabled', False)
    if not isinstance(enabled, bool): raise ValueError(where + '.enabled must be true or false')
    if not enabled: _known_keys(block, {'enabled'}, where)
    return block, enabled


def _evidence(value, where):
    block = _object(value, where)
    text_key(block.get('method'), where + '.method')
    if block['method'] == 'none' and set(block) != {'method'}:
        raise ValueError(where + ': none accepts no additional settings')
    return Settings(block).as_dict()


def _source(value, where):
    block = _object(value, where)
    _known_keys(block, {'execution_record', 'scientific_id', 'model_id'}, where)
    text_key(block.get('execution_record'), where + '.execution_record')
    for key in ['scientific_id'] + (['model_id'] if where.startswith('states') else []):
        if re.fullmatch('[a-f0-9]{64}', str(block.get(key, ''))) is None:
            raise ValueError(where + '.' + key + ' requires the actual saved SHA-256 identity')
    # Paths are declarations here; only an enabled scientific adapter opens them.
    return Settings(block).as_dict()


def _question(value, name):
    block, enabled = _enabled(value, name)
    if not enabled: return Settings({'enabled': False})
    common = {'enabled', 'statistic', 'evidence'}
    extras = {'characteristics': {'summary', 'distance_effect'}, 'simultaneous': set(),
        'delay': {'range_hours', 'resolution_hours', 'peak_resolution'},
        'proximity': {'window_hours', 'step_hours', 'coordination', 'coordination_measure', 'min_windows', 'min_distance_range'},
        'rhythm': {'source', 'settings'}, 'states': {'source', 'settings'}}[name]
    _known_keys(block, common | extras, name)
    result = {**block, 'enabled': True, 'statistic': text_key(block.get('statistic'), name + '.statistic'),
        'evidence': _evidence(block.get('evidence'), name + '.evidence')}
    if name == 'characteristics':
        if block.get('summary') not in {'mean', 'median', 'min', 'max'}:
            raise ValueError('characteristics.summary must explicitly be mean, median, min or max')
        result['distance_effect'] = text_key(block.get('distance_effect', 'spearman'), 'characteristics.distance_effect')
    if name == 'delay':
        low, high = _range(block.get('range_hours'), 'delay.range_hours')
        step = _number(block.get('resolution_hours'), 'delay.resolution_hours', positive=True)
        count = (high-low)/step
        if not math.isclose(count, round(count), abs_tol=1e-8, rel_tol=0) or count > 10000:
            raise ValueError('delay resolution must divide its full range into at most 10001 points')
        result.update(range_hours=[low, high], resolution_hours=step,
            peak_resolution=Settings(_object(block.get('peak_resolution', {}), 'delay.peak_resolution')).as_dict())
    if name == 'proximity':
        for key in ['window_hours', 'step_hours']:
            result[key] = _number(block.get(key), 'proximity.' + key, positive=True)
        if block.get('coordination') not in {'simultaneous', 'delay'}:
            raise ValueError('proximity.coordination must explicitly be simultaneous or delay')
        measures = {'signed_coefficient', 'absolute_coefficient'} if block['coordination'] == 'simultaneous' else {'maximum_absolute_lag_coefficient'}
        result['coordination_measure'] = block.get('coordination_measure', 'signed_coefficient' if block['coordination'] == 'simultaneous' else 'maximum_absolute_lag_coefficient')
        if result['coordination_measure'] not in measures: raise ValueError('proximity.coordination_measure must match the chosen simultaneous or complete-lag question')
        result['min_windows'] = _number(block.get('min_windows', 3), 'proximity.min_windows', minimum=3, integer=True)
        result['min_distance_range'] = _number(block.get('min_distance_range', 0.), 'proximity.min_distance_range', minimum=0)
    if name in {'rhythm', 'states'}:
        result['source'] = _source(block.get('source'), name + '.source')
        result['settings'] = Settings(_object(block.get('settings', {}), name + '.settings')).as_dict()
    return Settings(result)


def _geometry(value, groups):
    block = _object(value, 'geometry')
    _known_keys(block, {'x', 'y', 'definition', 'input_unit', 'unit', 'scale', 'recording_scales', 'neighbourhood', 'field'}, 'geometry')
    coordinates = []
    for axis in ['x', 'y']:
        chosen = _choices([block.get(axis)], groups, 'geometry.' + axis)
        if len(chosen) != 1 or chosen[0].summary is not None:
            raise ValueError('geometry.' + axis + ' requires one original coordinate column')
        coordinates.append(chosen[0])
    if coordinates[0] == coordinates[1]: raise ValueError('geometry x and y must differ')
    if block.get('definition') not in {'centroid', 'soma', 'recorded_point'}:
        raise ValueError('geometry.definition must explicitly be centroid, soma or recorded_point')
    result = {**block, 'x': coordinates[0].as_dict(), 'y': coordinates[1].as_dict()}
    for key in ['input_unit', 'unit']: text_key(block.get(key), 'geometry.' + key)
    def scale(v, where):
        if not isinstance(v, list) or len(v) != 2: raise ValueError(where + ' requires [x_scale, y_scale]')
        return [_number(x, where, positive=True) for x in v]
    result['scale'] = scale(block.get('scale'), 'geometry.scale')
    result['recording_scales'] = {text_key(movie, 'recording'): scale(v, 'geometry.recording_scales.' + movie)
        for movie, v in _object(block.get('recording_scales', {}), 'geometry.recording_scales').items()}
    neighbourhood = _object(block.get('neighbourhood'), 'geometry.neighbourhood')
    _known_keys(neighbourhood, {'method', 'distance', 'neighbours', 'comparison'}, 'geometry.neighbourhood')
    if neighbourhood.get('method') not in {'all', 'radius', 'nearest'}:
        raise ValueError('geometry.neighbourhood.method must explicitly be all, radius or nearest')
    allowed = {'method'}
    if neighbourhood['method'] == 'radius':
        allowed |= {'distance', 'comparison'}
        neighbourhood['distance'] = _number(neighbourhood.get('distance'), 'neighbourhood.distance', positive=True)
    if neighbourhood['method'] == 'nearest':
        allowed |= {'neighbours', 'comparison'}
        neighbourhood['neighbours'] = _number(neighbourhood.get('neighbours'), 'neighbourhood.neighbours', minimum=1, integer=True)
    _known_keys(neighbourhood, allowed, 'geometry.neighbourhood')
    if neighbourhood['method'] != 'all' and neighbourhood.get('comparison') not in {'retain_distant', 'restrict'}:
        raise ValueError('neighbourhood.comparison must explicitly retain_distant or restrict')
    result['neighbourhood'] = neighbourhood
    field = _object(block.get('field', {}), 'geometry.field')
    _known_keys(field, {'bounds', 'edge_margin'}, 'geometry.field')
    if field:
        bounds = field.get('bounds')
        if not isinstance(bounds, list) or len(bounds) != 4: raise ValueError('geometry.field.bounds requires [xmin, xmax, ymin, ymax] in output units')
        field['bounds'] = _range(bounds[:2], 'field x bounds') + _range(bounds[2:], 'field y bounds')
        field['edge_margin'] = _number(field.get('edge_margin'), 'field.edge_margin', minimum=0)
    result['field'] = field
    return Settings(result), tuple(coordinates)


def _reference(value):
    block = _object(value, 'shared_reference')
    if block.get('method') == 'none':
        _known_keys(block, {'method'}, 'shared_reference'); return Settings(block)
    _known_keys(block, {'method', 'min_cells', 'measurements', 'adjustment', 'units'}, 'shared_reference')
    if block.get('method') not in {'leave_pair_out_mean', 'external'}:
        raise ValueError('shared_reference.method must explicitly be none, leave_pair_out_mean or external')
    if block.get('adjustment') != 'subtract': raise ValueError('shared_reference.adjustment must explicitly be subtract in original units')
    if block['method'] == 'leave_pair_out_mean':
        _known_keys(block, {'method', 'min_cells', 'adjustment'}, 'shared_reference')
        block['min_cells'] = _number(block.get('min_cells'), 'shared_reference.min_cells', minimum=1, integer=True)
    else:
        _known_keys(block, {'method', 'measurements', 'adjustment', 'units'}, 'shared_reference')
        mapping = _object(block.get('measurements'), 'shared_reference.measurements')
        if not mapping: raise ValueError('external reference requires an endpoint-column to recorded-reference-column mapping')
        for key, value in mapping.items(): text_key(key, 'reference measurement'); text_key(value, 'external measurement')
        units = _object(block.get('units', {}), 'shared_reference.units')
        if set(units) - (set(mapping) | set(mapping.values())): raise ValueError('Reference units must name a requested endpoint or recorded reference')
        for key, value in units.items(): text_key(key, 'reference unit column'); text_key(value, 'reference unit')
        block['units'] = units
    return Settings(block)


@dataclass(frozen=True)
class CoordinationRequest(Record):
    name: str
    declaration: Settings
    reference_measurements: tuple[MeasurementChoice, ...]
    target_measurements: tuple[MeasurementChoice, ...]
    measurement_pairs: tuple[Settings, ...]
    questions: Settings
    geometry: Settings
    coordinates: tuple[MeasurementChoice, ...]
    support: Settings
    representation: str
    detrending: Settings
    shared_reference: Settings
    inference: Settings
    sample_summary: Settings
    biological_samples: Settings
    conditions: Settings
    table_grains: Settings
    time_range_hours: tuple[float, float] | None
    cells: tuple[tuple[str, int], ...] | None
    increment: str | None = None
    pipeline: str = 'spatial-coordination'

    @classmethod
    def from_dict(cls, value, groups, *, where='pipeline'):
        block = _object(value, where)
        _known_keys(block, {'pipeline', 'name', 'reference_measurements', 'target_measurements', 'pairs',
            'questions', 'geometry', 'support', 'representation', 'increment', 'detrending', 'shared_reference', 'inference',
            'sample_summary', 'biological_samples', 'conditions', 'table_grains', 'time_range_hours', 'cells'}, where)
        if block.get('pipeline') != 'spatial-coordination': raise ValueError('Expected spatial-coordination pipeline')
        name = text_key(block.get('name', 'spatial-coordination'), 'name')
        if re.fullmatch('[a-z][a-z0-9_-]*', name) is None: raise ValueError('name must be a lower-case identifier')
        reference = _choices(block.get('reference_measurements'), groups, 'reference_measurements')
        target = _choices(block.get('target_measurements'), groups, 'target_measurements')
        if not reference or not target: raise ValueError('Both independent endpoint measurement lists must be nonempty')
        if any(choice.summary is not None for choice in (*reference, *target)):
            raise ValueError('Declare characteristic summaries under questions.characteristics; endpoints retain original values')
        pairs = _object(block.get('pairs'), 'pairs'); _known_keys(pairs, {'mode', 'pairs'}, 'pairs')
        if pairs.get('mode') not in {'same_measurement', 'cartesian', 'explicit'}:
            raise ValueError('pairs.mode must explicitly be same_measurement, cartesian or explicit')
        a, b = {m.column for m in reference}, {m.column for m in target}
        if pairs['mode'] == 'explicit': raw = pairs.get('pairs')
        else:
            _known_keys(pairs, {'mode'}, 'pairs')
            raw = [[x.column, y.column] for x in reference for y in target if pairs['mode'] == 'cartesian' or x.column == y.column]
        if not isinstance(raw, list) or not raw: raise ValueError('pairs must select at least one endpoint measurement pair')
        selected = []
        for pair in raw:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                    any(not isinstance(item, str) for item in pair) or pair[0] not in a or pair[1] not in b):
                raise ValueError('Every explicit pair must follow the reference and target measurement lists')
            item = Settings({'reference': pair[0], 'target': pair[1]})
            if item not in selected: selected.append(item)
        question = _object(block.get('questions'), 'questions'); _known_keys(question, set(QUESTIONS), 'questions')
        questions = {key: _question(question.get(key, {}), key).as_dict() for key in QUESTIONS}
        if not any(q['enabled'] for q in questions.values()): raise ValueError('Enable at least one coordination question')
        if questions['proximity']['enabled'] and not questions[questions['proximity']['coordination']]['enabled']:
            raise ValueError('Enable the coordination question used by proximity')
        geometry, coordinates = _geometry(block.get('geometry'), groups)
        representation = block.get('representation')
        if representation not in {'raw', 'detrended', 'changes'}:
            raise ValueError('representation must explicitly be raw, detrended or changes')
        increment = block.get('increment', 'difference') if representation == 'changes' else None
        if (representation != 'changes' and 'increment' in block) or (representation == 'changes' and increment not in {'difference', 'rate'}):
            raise ValueError('increment must be difference or rate and applies only to the changes representation')
        detrending = _object(block.get('detrending', {}), 'detrending')
        if representation != 'detrended' and detrending: raise ValueError('Only detrended representation accepts detrending settings')
        if representation == 'detrended': text_key(detrending.get('detrend'), 'detrending.detrend')
        sample, enabled = _enabled(block.get('sample_summary', {}), 'sample_summary')
        if enabled:
            _known_keys(sample, {'enabled', 'aggregation', 'evidence', 'contrasts', 'timing'}, 'sample_summary')
            if sample.get('aggregation') not in {'mean', 'median'}: raise ValueError('sample_summary.aggregation must explicitly be mean or median')
            sample['evidence'] = _evidence(sample.get('evidence'), 'sample_summary.evidence')
            sample['contrasts'] = list(sample.get('contrasts', []))
            sample['timing'] = _object(sample.get('timing', {}), 'sample_summary.timing')
            for contrast in sample['contrasts']:
                if not isinstance(contrast, (list, tuple)) or len(contrast) != 2 or contrast[0] == contrast[1]:
                    raise ValueError('sample_summary.contrasts requires distinct reference/target condition pairs')
                for entry in contrast: text_key(entry, 'contrast condition')
        sample['enabled'] = enabled
        inference = _object(block.get('inference', {}), 'inference')
        inferential = any(q['enabled'] and q['evidence']['method'] != 'none' for q in questions.values()) or (enabled and sample['evidence']['method'] != 'none')
        _known_keys(inference, {'alpha', 'multiple_testing', 'correction_scope'}, 'inference')
        if inferential:
            alpha = _number(inference.get('alpha'), 'inference.alpha', positive=True)
            if alpha >= 1: raise ValueError('inference.alpha must be less than one')
            if inference.get('multiple_testing') not in {'none', 'bonferroni', 'sidak', 'bh'}: raise ValueError('Unknown multiple_testing method')
            if inference.get('correction_scope') not in {'all', 'question'}: raise ValueError('correction_scope must explicitly be all or question; evidence levels remain separate')
        elif inference: raise ValueError('inference requires an enabled evidence method')
        mappings = []
        for key in ['biological_samples', 'conditions']:
            mapping = _object(block.get(key, {}), key)
            for movie, value in mapping.items(): text_key(movie, key); text_key(value, key + '.' + movie)
            mappings.append(Settings(mapping))
        bounds = block.get('time_range_hours'); bounds = None if bounds is None else tuple(_range(bounds, 'time_range_hours'))
        cells = None
        if block.get('cells') is not None:
            if not isinstance(block['cells'], list): raise ValueError('cells must be movie/identity objects')
            cells = []
            for entry in block['cells']:
                entry = _object(entry, 'cells'); _known_keys(entry, {'movie', 'identity'}, 'cells')
                key = (text_key(entry.get('movie'), 'cells.movie'), cell_number(entry.get('identity')))
                if key not in cells: cells.append(key)
            cells = tuple(cells)
        return cls(name, Settings(block), reference, target, tuple(selected), Settings(questions), geometry, coordinates,
            _support(block.get('support')), representation, Settings(detrending), _reference(block.get('shared_reference')),
            Settings(inference), Settings(sample), *mappings, Settings(_object(block.get('table_grains', {}), 'table_grains')), bounds, cells, increment=increment)


@dataclass(frozen=True)
class ResolvedCoordinationRequest(Record):
    request: CoordinationRequest
    inputs: InputIdentity
    reference_measurements: tuple[Measurement, ...]
    target_measurements: tuple[Measurement, ...]
    coordinates: tuple[Measurement, ...]
    reference_columns: tuple[Measurement, ...]
    detrending: Settings
    processing_version: str | None

    @property
    def scientific_id(self):
        request = self.request.as_dict(); request.pop('name'); request.pop('declaration')
        return content_id({**self.as_dict(), 'request': request})


def resolve_request(request, *, source_run, tables, input_hashes, rhythm_params=None, table_grains=None):
    """Resolve measurements, units and populations, without opening optional sources."""
    import pandas as pd
    text_key(source_run, 'source_run')
    if any(not isinstance(name, str) or not isinstance(frame, pd.DataFrame) for name, frame in tables.items()):
        raise ValueError('tables must map names to measured DataFrames')
    declared = request.table_grains.as_dict()
    for name, grain in (table_grains or {}).items():
        if name in declared and tuple(declared[name]) != tuple(grain): raise ValueError('Conflicting grain: ' + name)
        declared[name] = grain
    declarations, grains = _catalogue(declared)
    temporal = any(request.questions[q]['enabled'] for q in QUESTIONS if q != 'characteristics')
    def measurement(choice, testing=temporal):
        if testing:
            m = _resolve_measurement(choice, tables, declarations, grains, testing=True)
        else:
            candidates, problems = [], []
            for name in ([choice.table] if choice.table else tables):
                if name not in tables or choice.column not in tables[name]: continue
                trace = bool({'hours', 'frame_index'} & set(grains.get(name, ())))
                selected = replace(choice, table=name, summary=request.questions['characteristics']['summary'] if trace else None)
                try: candidates.append(_resolve_measurement(selected, tables, declarations, grains, testing=False))
                except ValueError as error: problems.append(str(error))
            if len(candidates) != 1:
                raise ValueError(choice.column + (': ambiguous measured source' if candidates else ': unsuitable or unavailable; ' + '; '.join(problems)))
            m = candidates[0]
        trace = 'hours' in m.grain or 'frame_index' in m.grain
        if not trace and (request.time_range_hours is not None or request.representation != 'raw'):
            raise ValueError(m.column + ': a saved scalar cannot represent a new time window or temporal transformation')
        return m
    reference = tuple(measurement(c) for c in request.reference_measurements)
    target = tuple(measurement(c) for c in request.target_measurements)
    if request.shared_reference['method'] == 'external':
        from pymicroglia.pipelines.coordination.references import declared_unit
        units = request.shared_reference['units']
        reference = tuple(declared_unit(m, units) for m in reference)
        target = tuple(declared_unit(m, units) for m in target)
    coordinates = tuple(measurement(c, True) for c in request.coordinates)
    for m in coordinates:
        if m.unit and m.unit != request.geometry['input_unit']:
            raise ValueError(m.column + ': recorded coordinate unit differs from geometry.input_unit')
    if request.geometry['input_unit'] == request.geometry['unit'] and (request.geometry['scale'] != [1., 1.] or any(v != [1., 1.] for v in request.geometry['recording_scales'].values())):
        raise ValueError('An unchanged coordinate unit requires unit scale')
    by_reference = {m.column: m for m in reference}; by_target = {m.column: m for m in target}
    for pair in request.measurement_pairs:
        a, b = by_reference[pair['reference']], by_target[pair['target']]
        if request.questions['characteristics']['enabled'] and request.questions['characteristics']['statistic'] in {'absolute_difference', 'squared_difference'}:
            if a != b and (not a.unit or a.unit != b.unit):
                raise ValueError('Characteristic differences require the same measurement or explicitly matching recorded units')
    external = []
    if request.shared_reference['method'] == 'external':
        from pymicroglia.pipelines.coordination.references import resolve_external
        mapping = request.shared_reference['measurements']; endpoints = {**by_reference, **by_target}
        if set(mapping) != set(endpoints): raise ValueError('External reference mapping must cover every requested endpoint measurement')
        for key, column in mapping.items():
            m = resolve_external(column, tables, declarations, grains, request.shared_reference['units']); endpoint = endpoints[key]
            if m == endpoint or not m.unit or m.unit != endpoint.unit:
                raise ValueError('External subtraction requires a distinct recorded reference with matching known units')
            if m not in external: external.append(m)
    names = {m.table for m in (*reference, *target, *coordinates, *external)}
    if 'cell_summary' in tables:
        problem = _shape_problem(tables['cell_summary'], grains['cell_summary'], summary=None, testing=False)
        if problem: raise ValueError('cell_summary population: ' + problem)
        names.add('cell_summary')
    cells, samples = _population(source_run, tables, sorted(name for name in names if 'identity' in grains[name]), request)
    known_movies = {movie for name in names for movie in tables[name].stem.unique()}
    if set(request.geometry['recording_scales']) - known_movies: raise ValueError('Calibration declares an unknown recording')
    if set(request.conditions) - known_movies: raise ValueError('conditions declares an unknown recording')
    conditions = {}
    for sample in samples:
        if sample.confirmed:
            condition = request.conditions.get(sample.movie)
            if sample.sample in conditions and conditions[sample.sample] != condition:
                raise ValueError('Recordings of one biological sample must have consistent declared conditions')
            conditions[sample.sample] = condition
    fingerprints = {}
    for name in sorted(names):
        value = input_hashes.get(name)
        if re.fullmatch('[a-fA-F0-9]{64}', str(value)) is None: raise ValueError('input_hashes.' + name + ': verified SHA-256 required')
        fingerprints[name] = value.lower()
    detrending, version = Settings(), None
    if request.representation == 'detrended':
        import pymicroglia.workbench as circadian
        _known_keys(request.detrending, set(circadian.DETREND_DEFAULTS), 'detrending')
        detrending = Settings(circadian.detrend_settings(request.detrending.as_dict())); version = circadian.WORKBENCH_VERSION
    return ResolvedCoordinationRequest(request, InputIdentity(source_run, Settings(fingerprints), cells, samples),
        reference, target, coordinates, tuple(external), detrending, version)


BRANCHES = {'characteristic-similarity': 'characteristics', 'simultaneous-coordination': 'simultaneous',
    'lagged-coordination': 'delay', 'changing-proximity': 'proximity', 'rhythm-coordination': 'rhythm', 'state-coordination': 'states'}
SCIENCE = (
    StepSpec('coordination-design', 'coordination-design', inputs=('measured-tables',)),
    StepSpec('pair-inputs', 'pair-inputs', ('coordination-design',)),
    StepSpec('characteristic-similarity', 'characteristic-similarity', ('pair-inputs',)),
    StepSpec('simultaneous-coordination', 'simultaneous-coordination', ('pair-inputs',)),
    StepSpec('lagged-coordination', 'lagged-coordination', ('pair-inputs', 'simultaneous-coordination')),
    StepSpec('changing-proximity', 'changing-proximity', ('pair-inputs', 'simultaneous-coordination')),
    StepSpec('rhythm-coordination', 'rhythm-coordination', ('pair-inputs',)),
    StepSpec('state-coordination', 'state-coordination', ('pair-inputs',)),
    StepSpec('coordination-evidence', 'coordination-evidence', ('pair-inputs', *BRANCHES)),
    StepSpec('coordination-samples', 'coordination-samples', ('pair-inputs', 'coordination-evidence')))
RENDERS = (
    StepSpec('coordination-overview', 'coordination-overview', tuple(s.name for s in SCIENCE), kind='render'),
    StepSpec('connection-maps', 'connection-maps', ('pair-inputs', 'coordination-evidence'),
        selection='coordination-evidence:supported-pairs', requires_selected_rows=True, kind='render'),
    StepSpec('pair-report-cards', 'pair-report-cards', ('pair-inputs', *BRANCHES, 'coordination-evidence'),
        selection='coordination-evidence:supported-pairs', requires_selected_rows=True, kind='render'))
RECIPE = PipelineRecipe('spatial-coordination', 1, (*SCIENCE, *RENDERS,
    StepSpec('linked-results-index', 'linked-results-index', tuple(s.name for s in (*SCIENCE, *RENDERS)), kind='render')))


def identity(context):
    from pymicroglia.pipelines._screening import file_hash, read_verified_tables
    read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    return content_id({'request': context.request.scientific_id, 'code': file_hash(Path(__file__))})


def freeze_design(context):
    from pymicroglia.pipelines._screening import file_hash, _write_json
    context.output.mkdir(parents=True); path = context.output / 'coordination_design.json'
    _write_json(path, {'schema_version': 1, 'scientific_id': context.scientific_id, 'request': context.request.as_dict(),
        'scientific_evaluation': False, 'lag_convention': LAG_CONVENTION,
        'evidence_levels': 'Pair, recording and biological sample evidence remain separate; shared-cell pairs are not independent replicates',
        'geometry_population': 'Scientific neighbourhood is defined before examining relationship values; display limits do not change it',
        'optional_sources_opened': False})
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Frozen spatial questions, independent endpoints, geometry and sample definitions',
        (ArtifactRef('coordination_design', path.name, file_hash(path), context.scientific_id),))


def _pending(context):
    from pymicroglia.pipelines._runner import Unavailable
    if context.step.name in BRANCHES and not context.request.request.questions[BRANCHES[context.step.name]]['enabled']:
        return StepResult(context.step.name, context.scientific_id, 'skipped-empty', 'Question explicitly disabled; no optional source was opened')
    raise Unavailable('Required ' + context.step.name + ' producer is awaiting its implementation stage')


def temporal_branch_identity(context):
    """Disabled delay/proximity do not need prepared temporal observations."""
    from pymicroglia.pipelines._screening import file_hash
    question=BRANCHES[context.step.name]
    if not context.request.request.questions[question]['enabled']:
        return content_id({'question':question,'enabled':False,'code':file_hash(__file__)})
    import pymicroglia.pipelines.coordination.delay as coordination_delay
    import pymicroglia.pipelines.coordination.proximity as coordination_proximity
    return (coordination_delay if question=='delay' else coordination_proximity).identity(context)


def producers():
    from pymicroglia.pipelines._runner import Producer
    import pymicroglia.pipelines.coordination.inputs as coordination_inputs
    import pymicroglia.pipelines.coordination.characteristics as coordination_characteristics
    import pymicroglia.pipelines.coordination.simultaneous as coordination_simultaneous
    import pymicroglia.pipelines.coordination.delay as coordination_delay
    import pymicroglia.pipelines.coordination.proximity as coordination_proximity
    import pymicroglia.pipelines.coordination.rhythm as coordination_rhythm
    import pymicroglia.pipelines.coordination.states as coordination_states
    import pymicroglia.pipelines.coordination.evidence as coordination_evidence
    import pymicroglia.pipelines.coordination.samples as coordination_samples
    import pymicroglia.pipelines.coordination.map_figures as coordination_map_figures
    import pymicroglia.pipelines.coordination.overview_figures as coordination_overview_figures
    import pymicroglia.pipelines.coordination.card_figures as coordination_card_figures
    import pymicroglia.pipelines.coordination.index as coordination_index
    return {**{step.producer: Producer(_pending) for step in RECIPE.steps},
        'coordination-design': Producer(freeze_design, identity),
        'pair-inputs': Producer(coordination_inputs.produce, coordination_inputs.identity),
        'characteristic-similarity': Producer(coordination_characteristics.produce, coordination_characteristics.identity),
        'simultaneous-coordination': Producer(coordination_simultaneous.produce, coordination_simultaneous.identity),
        'lagged-coordination': Producer(coordination_delay.produce, temporal_branch_identity),
        'changing-proximity': Producer(coordination_proximity.produce, temporal_branch_identity),
        'rhythm-coordination': Producer(coordination_rhythm.produce, coordination_rhythm.identity),
        'state-coordination': Producer(coordination_states.produce, coordination_states.identity),
        'coordination-evidence': Producer(coordination_evidence.produce, coordination_evidence.identity),
        'coordination-samples': Producer(coordination_samples.produce, coordination_samples.identity),
        'connection-maps': Producer(coordination_map_figures.produce, version=coordination_map_figures.version()),
        'coordination-overview': Producer(coordination_overview_figures.produce, version=coordination_overview_figures.version(), accepts_unavailable_dependencies=True),
        'pair-report-cards': Producer(coordination_card_figures.produce, version=coordination_card_figures.version(), accepts_unavailable_dependencies=True),
        'linked-results-index': Producer(coordination_index.produce, version=coordination_index.version(), accepts_unavailable_dependencies=True)}


def execution_recipe(request):
    """Only requested scientific evidence is a prerequisite of its collector.

    Disabled producers still appear in a full execution and in the collector's
    explicit status inventory. Their legitimate empty outcomes cannot block
    unrelated enabled questions. Enabled failures retain ordinary runner gates.
    """
    from dataclasses import replace
    enabled={step for step,question in BRANCHES.items() if request.questions[question]['enabled']}
    if any(request.questions[q]['enabled'] for q in ['simultaneous','delay','proximity']):enabled.add('simultaneous-coordination')
    return replace(RECIPE,steps=tuple(replace(step,prerequisites=tuple(name for name in step.prerequisites if name not in BRANCHES or name in enabled))
        if step.name=='coordination-evidence' else replace(step,prerequisites=('pair-inputs',))
        if step.name in {'lagged-coordination','changing-proximity'} and not request.questions[BRANCHES[step.name]]['enabled'] else step for step in RECIPE.steps))


def run_request(resolved, table_paths, output, *, presentation=None, only=None):
    from pymicroglia.pipelines._runner import run_pipeline
    return run_pipeline(execution_recipe(resolved.request), producers(), request=resolved, scientific_settings={'coordination': resolved.scientific_id},
        table_paths=table_paths, output=output, presentation=presentation, only=only)
