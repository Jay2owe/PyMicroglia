"""Persistent comparison designs and thin native figure/job adapters."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
import re
from pathlib import Path
import shutil
from threading import RLock
from uuid import uuid4

from .exploration_data import manifest, recording, table_path
from ..measure.comparison_conditions import parse_cell_ids, validate_cell_conditions, frame_conditions

DRAW_LOCK = RLock()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '-' + uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _folder(run):
    root, _ = manifest(run)
    return root / 'AI_Exports' / 'comparisons'


def _id(run, stem):
    return sha256((str(Path(run).resolve()) + '\n' + stem).encode()).hexdigest()[:20]


def _recordings(run):
    root, data = manifest(run)
    from ..measure.conditions import ConditionSet
    from ..visualisation.panels import resolve_colour
    declared = ConditionSet.from_config(data.get('settings', {}).get('conditions'))
    rows, conditions = [], []
    for movie in data['movies']:
        spec = movie.get('spec', {})
        condition = movie.get('condition', spec.get('condition', ''))
        factors = condition if isinstance(condition, dict) else {}
        label = factors.get('condition', '') if factors else condition or ''
        if not isinstance(label, str):
            label = str(label)
        cid = 'condition-' + sha256(label.encode()).hexdigest()[:12] if label else ''
        rows.append(dict(id=_id(root, movie['stem']), run=str(root), stem=movie['stem'],
                         include=True, condition=cid, subject=movie.get('subject') or spec.get('subject') or '',
                         slice='', pair='', factors={k: str(v) for k, v in factors.items() if k != 'condition'},
                         analysis_ids={}, cell_conditions={}))
        if cid and cid not in {row['id'] for row in conditions}:
            definition=declared.get(label)
            conditions.append(dict(id=cid, label=definition.label if definition else label,
                colour=resolve_colour(definition.colour) if definition and definition.colour else _colour(len(conditions)),
                control=bool(definition and definition.control)))
    return rows, conditions


def _colour(index):
    from ..visualisation.panels import colour
    return colour(('teal', 'orange', 'blue', 'plum', 'okabe_bluish_green', 'muted')[index % 6])


def _paths(design, root, *, relative=False):
    design = json.loads(json.dumps(design))
    for row in design['recordings']:
        path = Path(row['run'])
        path = path.resolve() if path.is_absolute() else (root/path).resolve()
        row['run'] = os.path.relpath(path, root) if relative else str(path)
    return design


def workspace(run):
    root, _ = manifest(run)
    path = _folder(root) / 'workspace.json'
    if path.is_file():
        design = _paths(json.loads(path.read_text(encoding='utf-8')), root)
        return validate_design(design)
    rows, conditions = _recordings(root)
    return dict(schema_version=1, recordings=rows, conditions=conditions, plots={}, layout={})


def validate_design(design):
    if design.get('schema_version') != 1:
        raise ValueError('Unsupported comparison design version')
    from matplotlib.colors import is_color_like
    rows = design.get('recordings')
    conditions = design.get('conditions')
    if not isinstance(rows, list) or not rows or len(rows) > 256 or not isinstance(conditions, list):
        raise ValueError('Choose between 1 and 256 recordings and a condition list')
    ids = [str(row['id']) for row in conditions]
    if len(ids) != len(set(ids)) or any(not cid for cid in ids):
        raise ValueError('Condition identifiers must be unique and nonempty')
    for condition in conditions:
        if not condition.get('label', '').strip() or not is_color_like(condition.get('colour', '')):
            raise ValueError('Each condition needs a label and valid colour')
    seen = set()
    for row in rows:
        root, _, movie = recording(row['run'], row['stem'])
        row['run'] = str(root)
        expected = _id(root, movie['stem'])
        # A portable workspace preserves its assigned keys across relocation.
        row.setdefault('id', expected)
        if row['id'] in seen or (str(root), movie['stem']) in seen:
            raise ValueError('The same recording is included twice')
        seen.update([row['id'], (str(root), movie['stem'])])
        if row.get('condition', '') and row['condition'] not in ids:
            raise ValueError(f"Unknown condition assigned to {row['stem']}")
        row['cell_conditions'] = validate_cell_conditions(row.get('cell_conditions', {}), ids)
        row.setdefault('include', True)
        for key in ('subject', 'slice', 'pair'):
            row[key] = str(row.get(key) or '')
        if not isinstance(row.get('factors', {}), dict) or not isinstance(row.get('analysis_ids', {}), dict):
            raise ValueError('Factors and saved analysis selections must be mappings')
    return design


def save_workspace(run, design):
    root, _ = manifest(run)
    design = validate_design(_paths(design, root))
    _atomic(_folder(root)/'workspace.json', _paths(design, root, relative=True))
    return design


def add_run(design, run):
    return add_runs(design, [run])


def add_runs(design, runs):
    """Combine saved runs in one design; invalid batches leave the original intact."""
    if not isinstance(runs, list) or not 1 <= len(runs) <= 256 or any(not isinstance(run, str) or not run.strip() for run in runs):
        raise ValueError('Choose 1 to 256 saved measurement folders')
    design = validate_design(json.loads(json.dumps(design)))
    for run in runs:
        rows, conditions = _recordings(run.strip())
        existing = {(row['run'], row['stem']) for row in design['recordings']}
        design['recordings'].extend(row for row in rows if (row['run'], row['stem']) not in existing)
        known = {row['id'] for row in design['conditions']}
        design['conditions'].extend(row for row in conditions if row['id'] not in known)
    return validate_design(design)


@lru_cache(maxsize=128)
def _fingerprint(path, size, modified):
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=12)
def _table(path, size, modified, metric):
    import pandas as pd
    if shutil.disk_usage(Path(path).parent).free < size*4 + 64*1024**2:
        raise ValueError('Insufficient local space to load saved comparison measurements')
    columns={'identity','frame_index','hours','stem',metric}
    digest=_fingerprint(path,size,modified)
    frame=pd.read_csv(path,usecols=lambda column:column in columns)
    after=Path(path).stat()
    if (after.st_size,after.st_mtime_ns)!=(size,modified):
        raise ValueError('Saved measurements changed while loading the comparison')
    return frame,digest


def selection_preview(payload):
    """Resolve assignment targets from full saved IDs, without changing labels."""
    import numpy as np
    import pandas as pd
    design = validate_design(json.loads(json.dumps(payload['design'])))
    selection = payload.get('selection', {})
    if not isinstance(selection, dict):
        raise ValueError('Choose cells, recordings or a recording name pattern')
    mode = selection.get('mode', 'cells')
    if mode not in ('cells', 'recordings', 'pattern'):
        raise ValueError('Choose cells, recordings or a recording name pattern')
    selected = selection.get('recording_ids', [])
    if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        raise ValueError('Choose recording identifiers from the assignment table')
    if set(selected) - {row['id'] for row in design['recordings']}:
        raise ValueError('A selected recording is no longer in this design')
    identities = parse_cell_ids(selection.get('ids', '')) if mode == 'cells' else []
    pattern = None
    if mode == 'pattern':
        expression = selection.get('pattern', '')
        if not isinstance(expression, str) or len(expression) > 512:
            raise ValueError('Use a recording name pattern of at most 512 characters')
        if expression:
            try:
                pattern = re.compile(expression, re.IGNORECASE if selection.get('ignore_case', True) else 0)
            except re.error as exc:
                raise ValueError(f'Invalid regular expression: {exc}') from None
    targets = []
    for row in design['recordings']:
        if mode == 'pattern':
            if pattern is None or pattern.search(row['stem']) is None:
                continue
        elif row['id'] not in selected:
            continue
        root, _, movie = recording(row['run'], row['stem'])
        path = table_path(root, movie, 'cell_frame')
        if path is None:
            raise ValueError(f"{row['stem']}: no saved cell-frame table")
        stat = path.stat()
        frame, digest = _table(str(path), stat.st_size, stat.st_mtime_ns, 'identity')
        if 'stem' in frame:
            frame = frame[frame.stem.eq(row['stem'])]
        if 'identity' not in frame:
            raise ValueError(f"{row['stem']}: no saved cell identities")
        values = pd.to_numeric(frame.identity, errors='coerce')
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1).ne(0).any():
            raise ValueError('Invalid saved cell identities')
        available = set(values.astype('int64'))
        matched = sorted(available.intersection(identities) if mode == 'cells' else available)
        targets.append(dict(recording_id=row['id'], stem=row['stem'], identities=matched,
                            missing_ids=sorted(set(identities)-available), sha256=digest))
    signature = sha256(_json(dict(selection=selection, targets=targets,
        sources=[{key:row[key] for key in ('id','run','stem')} for row in design['recordings']])).encode()).hexdigest()
    return dict(mode=mode, targets=targets, cells=sum(len(row['identities']) for row in targets),
                recordings=sum(bool(row['identities']) for row in targets), signature=signature)


def assign_condition(payload):
    """Apply one condition to the previewed cells or entire recordings and save."""
    preview = selection_preview(payload)
    if preview['signature'] != payload.get('expected_signature'):
        raise ValueError('The selection changed; preview it again before setting a condition')
    if not preview['cells']:
        raise ValueError('Select at least one saved cell or recording')
    label = payload.get('condition', '')
    if not isinstance(label, str) or not label.strip() or len(label) > 200:
        raise ValueError('Choose or type a condition name (at most 200 characters)')
    label = label.strip()
    design = validate_design(json.loads(json.dumps(payload['design'])))
    existing = next((row for row in design['conditions'] if row['label'].casefold() == label.casefold()), None)
    if label.casefold() == 'unassigned':
        condition = ''
    elif existing:
        condition = existing['id']
    else:
        condition = 'condition-' + uuid4().hex
        design['conditions'].append(dict(id=condition, label=label, colour=_colour(len(design['conditions'])), control=False))
    targets = {row['recording_id']:row for row in preview['targets']}
    for row in design['recordings']:
        target = targets.get(row['id'])
        if target is None or not target['identities']:
            continue
        if preview['mode'] != 'cells':
            row['condition'] = condition
            row['cell_conditions'] = {}
        else:
            for identity in target['identities']:
                if condition == row.get('condition', ''):
                    row['cell_conditions'].pop(str(identity), None)
                else:
                    row['cell_conditions'][str(identity)] = condition
    return dict(design=save_workspace(payload['run'], design), selection=preview, condition=label)


def prepare(payload):
    import numpy as np
    import pandas as pd
    from ..figure_tables.comparison import prepare_frames, settings
    design = validate_design(json.loads(json.dumps(payload['design'])))
    config = settings(payload.get('options', {}))
    blocks, inputs, warnings = [], [], []
    all_cells = excluded = 0
    for source in design['recordings']:
        if not source['include']:
            continue
        root, _, movie = recording(source['run'], source['stem'])
        path = table_path(root, movie, 'cell_frame')
        if path is None:
            raise ValueError(f"{source['stem']}: no saved cell-frame table")
        stat = path.stat()
        original, digest = _table(str(path), stat.st_size, stat.st_mtime_ns, config['metric'])
        block = original.copy()
        if 'stem' in block:
            block = block[block.stem.eq(source['stem'])].copy()
        required = {'identity', 'frame_index', 'hours'}
        if not required.issubset(block.columns) or block.duplicated(['identity', 'frame_index']).any():
            raise ValueError('Comparison observations need unique identity/frame pairs and hours')
        for key in required:
            block[key] = pd.to_numeric(block[key], errors='coerce')
        if not np.isfinite(block[list(required)].to_numpy()).all() or (block.identity <= 0).any() or (block.frame_index < 0).any() or (block[['identity','frame_index']] % 1).ne(0).any().any():
            raise ValueError('Invalid saved cell identities or observation times')
        if block.groupby('frame_index').hours.nunique().gt(1).any():
            raise ValueError('Cells in the same saved frame must share one observation time')
        count = block.identity.nunique()
        all_cells += count
        filters = payload.get('exclusions', [])
        for rule in filters:
            if not rule:
                continue
            from .exploration_filters import preview_filters
            selection = preview_filters(root, source['stem'], **rule,
                analysis_ids=source.get('analysis_ids', {}))
            block = block[block.identity.isin(selection['identities'])]
        excluded += count-block.identity.nunique()
        if config['metric'] not in block:
            block[config['metric']] = np.nan
            warnings.append(f"{source['stem']}: {config['metric']} was not measured")
        block['run'] = str(root)
        block['stem'] = source['stem']
        block['recording_id'] = source['id']
        block['condition'] = frame_conditions(source, block.identity)
        block['subject'] = source.get('subject', '')
        block['pair'] = source.get('pair', '')
        block['recording'] = source['stem'] + ' · ' + source['id'][:6]
        for key, value in source.get('factors', {}).items():
            block['factor:' + key] = value
        if config['group_by'] == 'rhythmicity':
            aid = source.get('analysis_ids', {}).get(config['metric'])
            if not aid:
                raise ValueError(f"Choose a completed {config['metric']} rhythm analysis for {source['stem']}")
            from .exploration_rhythms import load_attached_rhythms
            result = load_attached_rhythms(root, movie, aid)
            if result is None:
                raise ValueError('The selected completed rhythm analysis is unavailable or changed')
            classes = {int(row['identity']): row.get('status', 'Untestable') for row in result['rows'] if row['metric'] == config['metric']}
            block['rhythmicity'] = block.identity.map(classes).fillna('Untestable')
        blocks.append(block)
        inputs.append(dict(run=str(root), recording_id=source['id'], stem=source['stem'], path=str(path), sha256=digest))
    if not blocks:
        raise ValueError('Include at least one recording')
    combined = pd.concat(blocks, ignore_index=True)
    from .exploration_data import column_metadata
    definition = column_metadata().get(config['metric'])
    label = definition.label + (f' ({definition.unit})' if definition.unit else '') if definition else config['metric'].replace('_',' ')
    if label==config['metric']:label=label.replace('_',' ')
    data = prepare_frames(combined, config, conditions=design['conditions'], metric_label=label)
    data['counts'].update(total_cells=int(all_cells), excluded_cells=int(excluded))
    data.update(inputs=inputs, design=design, warnings=warnings)
    return data


def analysis_key(data, exclusions=()):
    config = {k: v for k, v in data['config'].items() if k not in ('view', 'bins', 'bandwidth', 'band', 'unit', 'aggregate')}
    rows = [{k: row.get(k) for k in ('id', 'condition', 'cell_conditions', 'subject', 'pair', 'factors', 'include', 'analysis_ids')}
            for row in data['design']['recordings'] if row['include']]
    inputs=[{key:row[key] for key in ('recording_id','sha256')} for row in data['inputs']]
    values=data['cells'][['recording_id','identity','group','value','subject','pair']].sort_values(['recording_id','identity'])
    return sha256(_json(dict(inputs=inputs, config=config, assignments=rows,
        prepared_values=clean_records(values), exclusions=exclusions)).encode()).hexdigest()


def clean_records(table):
    return json.loads(table.to_json(orient='records', double_precision=15))


def run_comparison(payload):
    from .. import workbench
    from ..measure.contrasts import parse_contrasts
    data = prepare(payload)
    test = dict(payload['test'])
    groups = test.get('groups') or data['groups']
    config = data['config']
    unit = test['unit']
    aggregate = test.get('aggregate')
    frame = data['cells'].rename(columns={'recording_id': 'stem', 'stem': 'recording_name'})
    frame[config['metric']] = frame['value']
    if test['test'] not in ('wilcoxon', 'paired_t') and unit in ('movie', 'subject'):
        import numpy as np
        key = 'stem' if unit == 'movie' else 'subject'
        eligible = frame[frame.group.isin(groups) & np.isfinite(frame.value)]
        if eligible.groupby(key).group.nunique().gt(1).any():
            raise ValueError('The same sample contributes cells to multiple conditions. '
                             'Use a paired test with pairing identifiers, or select separate samples.')
    if test['test'] in ('wilcoxon', 'paired_t'):
        if frame.pair.eq('').any():
            raise ValueError('Assign pairing identifiers before running a paired test')
        frame['subject'] = frame['pair']
        unit = 'subject'
    declaration = dict(name='comparison', table='comparison_cells', metrics=[config['metric']],
        group_by='group', groups=groups, unit=unit, aggregate=None if unit == 'cell' else aggregate,
        test=test['test'], correction=test.get('correction', 'benjamini_hochberg'),
        alpha=float(test.get('alpha', .05)), family='comparison')
    specs = parse_contrasts([declaration], {})
    statistics = workbench.group_contrasts.group_contrasts({'comparison_cells': frame}, specs)
    from .. import __version__
    identity = analysis_key(data, payload.get('exclusions', []))
    key = uuid4().hex[:24]
    result = dict(schema_version=1, id=key, created_at=datetime.now(timezone.utc).isoformat(),
                  input_key=identity, request=payload, statistics=clean_records(statistics),
                  counts=data['counts'], inputs=data['inputs'], version=__version__,
                  workbench_version=workbench.WORKBENCH_VERSION)
    _atomic(_folder(payload['run'])/'results'/f'{key}.json', result)
    return result


def result(run, identifier):
    if len(identifier) != 24 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Invalid comparison result identifier')
    return json.loads((_folder(run)/'results'/f'{identifier}.json').read_text(encoding='utf-8'))


def history(run):
    files=sorted((_folder(run)/'results').glob('*.json'),key=lambda path:path.stat().st_mtime,reverse=True)[:100]
    rows=[]
    for path in files:
        saved=json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(id=saved['id'],created_at=saved['created_at'],input_key=saved['input_key'],
                         metric=saved['request']['options']['metric'],test=saved['request']['test'],
                         counts=saved['counts']))
    return dict(comparisons=rows)


def render(payload, *, format='svg'):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from ..visualisation.figures import get_figure
    import pandas as pd
    if format not in ('svg', 'png'):
        raise ValueError('Choose SVG or PNG')
    data = prepare(payload)
    input_key = analysis_key(data, payload.get('exclusions', []))
    if payload.get('expected_input_key') and payload['expected_input_key']!=input_key:
        raise ValueError('Saved measurements changed since this plot was displayed; redraw before exporting')
    completed = payload.get('completed_result') or (result(payload['run'], payload['result_id']) if payload.get('result_id') else None)
    stale = bool(completed and completed['input_key'] != input_key)
    if data['config']['view'] == 'effects':
        if not completed or stale:
            raise ValueError('Run a comparison with these data settings to show effect sizes')
        from ..figure_tables.contrast import prepare as prepare_effects
        class Source:
            interval = None
            def table(self, *args, **kwargs):
                return pd.DataFrame(completed['statistics'])
        effects, _ = prepare_effects(Source(), dict(order='family', metrics=[]))
        data['effects'] = effects['effects']
        if 'rows' in data['effects']:
            labels=dict(zip(data['groups'],data['labels']))
            rows=data['effects']['rows']
            rows['label']=[f"{labels.get(a,a)} vs {labels.get(b,b)}" if a and b else data['ylabel']
                           for a,b in zip(rows.group_a.fillna(''),rows.group_b.fillna(''))]
    appearance = payload.get('appearance', {})
    if appearance.get('colourOverride') and data['colours']:
        data['colours'][0]=appearance['colourOverride']
    width = max(400, min(1800, int(payload.get('width', 900))))
    height = max(160, min(720, int(appearance.get('height', 320))))
    background = appearance.get('background', '#ffffff')
    from matplotlib.colors import to_rgb
    r,g,b=to_rgb(background)
    foreground = '#e4edf4' if .2126*r+.7152*g+.0722*b < .4 else '#283540'
    with DRAW_LOCK, matplotlib.rc_context({'svg.fonttype': 'none', 'font.size': float(appearance.get('fontSize', 12)),
            'axes.labelcolor': foreground, 'text.color': foreground, 'xtick.color': foreground,
            'ytick.color': foreground, 'axes.edgecolor': foreground, 'axes.facecolor': background,
            'figure.facecolor': background, 'lines.linewidth': float(appearance.get('lineWidth', 1.25))}):
        figure = Figure(figsize=(width/100, height/100), dpi=100, layout='constrained')
        FigureCanvasAgg(figure)
        drawn = get_figure('cell_group_comparison').view(data['config']['view']).draw(figure, data)
        axis = drawn.axes
        if hasattr(axis, 'set_ylim'):
            bounds = [appearance.get('yMin'), appearance.get('yMax')]
            if any(value not in (None, '') for value in bounds):
                axis.set_ylim(*[None if value in (None, '') else float(value) for value in bounds])
        figure.canvas.draw()
        hits = []
        if data['config']['view'] == 'cells':
            for point in data['hits']:
                x, y = axis.transData.transform([point['x'], point['y']])
                hits.append({**point, 'x': float(x), 'y': float(height-y)})
        stream = BytesIO()
        receipt=dict(schema_version=1, request=payload, input_key=input_key, inputs=data['inputs'],
                     cells=clean_records(data['cells']), samples=clean_records(data['samples']),
                     timeline=[dict(group=row['group'],values=clean_records(row['table'])) for row in data['timeline']],
                     completed_result=completed)
        figure.savefig(stream, format=format, dpi=100, metadata={'Description':json.dumps(receipt,allow_nan=False)})
        image = stream.getvalue()
        figure.clear()
    return dict(image=image, width=width, height=height, hits=hits, counts=data['counts'],
                input_key=input_key, stale=stale, warnings=data['warnings'],
                cells=clean_records(data['cells']), samples=clean_records(data['samples']),
                timeline=[dict(group=row['group'], values=clean_records(row['table'])) for row in data['timeline']],
                inputs=data['inputs'], groups=[dict(id=k, label=v) for k, v in zip(data['groups'], data['labels'])])


def replay_script(payload):
    """A standalone native render with frozen inputs and optional test evidence."""
    rendered=render(payload)
    request=json.loads(json.dumps(payload))
    if payload.get('result_id'):
        request['completed_result']=result(payload['run'],payload['result_id'])
    return ("from pathlib import Path\nfrom hashlib import sha256\nimport json\n"
            "from pymicroglia.ui.exploration_comparison import render\n\n"
            f"REQUEST = {request!r}\nINPUTS = {rendered['inputs']!r}\n"
            "for source in INPUTS:\n"
            "    if sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:\n"
            "        raise ValueError('Saved comparison measurements changed')\n"
            "reproduced = render(REQUEST)\n"
            f"if reproduced['input_key'] != {rendered['input_key']!r}:\n"
            "    raise ValueError('Comparison assignments or preparation changed')\n"
            "Path('comparison.svg').write_bytes(reproduced['image'])\n"
            "Path('comparison-values.json').write_text(json.dumps({k:v for k,v in reproduced.items() if k != 'image'}, indent=2), encoding='utf-8')\n")
