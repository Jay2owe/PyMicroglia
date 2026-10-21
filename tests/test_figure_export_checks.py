"""The visual parity gate must reject changed marks, pixels, and placement."""
import base64
from io import BytesIO
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def checker():
    path = Path(__file__).resolve().parents[1]/'tools/check_figure_svg_parity.py'
    spec = importlib.util.spec_from_file_location('figure_svg_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_svg(path, *, native, changed_pixel=False, x=0, line=10):
    pixels = np.array([[[255, 0, 0, 255], [0, 128, 0, 255], [0, 0, 255, 255]]], dtype='uint8')
    if changed_pixel:
        pixels[0, 1, 0] = 100
    if not native:
        pixels = np.repeat(np.repeat(pixels, 4, axis=0), 4, axis=1)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    data = base64.b64encode(stream.getvalue()).decode()
    attributes = (f'width="3" height="1" transform="matrix(4 0 0 4 {x} 0)" '
                  'style="image-rendering:pixelated"') if native else f'width="12" height="4" x="{x}"'
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
                    f'<path d="M 0 0 L {line} 10"/><image {attributes} xlink:href="data:image/png;base64,{data}"/></svg>')


def test_svg_check_accepts_native_pixels_but_rejects_changed_plot(tmp_path):
    before, after = tmp_path/'before.svg', tmp_path/'after.svg'
    check = checker().compare
    write_svg(before, native=False)
    write_svg(after, native=True)
    assert check(before, after)['parity']
    for change in [{'changed_pixel': True}, {'x': 5}, {'line': 20}]:
        write_svg(after, native=True, **change)
        assert not check(before, after)['parity']
    write_svg(after, native=True)
    after.write_text(after.read_text().replace('<image ', '<image opacity="0.5" '))
    assert not check(before, after)['parity']


def test_native_image_respects_the_same_visible_clip(tmp_path):
    before, after = tmp_path/'before.svg', tmp_path/'after.svg'
    write_svg(before, native=False)
    write_svg(after, native=True)
    import xml.etree.ElementTree as ET
    module = checker()
    for path in [before, after]:
        root = ET.parse(path).getroot()
        picture = root.find(module.SVG+'image')
        root.remove(picture)
        definitions = ET.SubElement(root, module.SVG+'defs')
        clip = ET.SubElement(definitions, module.SVG+'clipPath', {'id': 'crop'})
        ET.SubElement(clip, module.SVG+'rect', {'width': '12', 'height': '4'})
        group = ET.SubElement(root, module.SVG+'g', {'clip-path': 'url(#crop)'})
        group.append(picture)
        if path == after:
            picture.set('width', '4')
            source = Image.open(BytesIO(base64.b64decode(picture.attrib[module.HREF].split(',')[1]))).convert('RGBA')
            pixels = np.concatenate([np.asarray(source), np.array([[[255, 255, 0, 255]]], dtype='uint8')], axis=1)
            stream = BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
            picture.set(module.HREF, 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
        path.write_bytes(ET.tostring(root))
    assert module.compare(before, after)['parity']


def test_statistics_check_only_normalizes_creation_time():
    import json
    import pytest
    path = Path(__file__).resolve().parents[1]/'tools/check_figure_export_parity.py'
    spec = importlib.util.spec_from_file_location('figure_data_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalize = module.normalized_statistics

    def rows(created, period=24):
        return [{'results_json': json.dumps([{'data': {'period': period},
                                              'provenance': {'generated_utc': created}}])}]

    before = rows('2026-10-20T08:00:00+00:00')
    assert normalize(before) == normalize(rows('2026-10-20T09:00:00+00:00'))
    assert normalize(before) != normalize(rows('2026-10-20T09:00:00+00:00', 23))
    assert '08:00:00' in before[0]['results_json']
    with pytest.raises(ValueError):
        normalize(rows('invalid'))


def test_vector_rounding_tolerance_does_not_hide_changed_marks_or_text():
    same = checker().vectors_equivalent
    assert same({'d': 'M 1 3.123456 L 2 4'}, {'d': 'M 1 3.123457 L 2 4'})
    assert not same({'d': 'M 1 3.123456 L 2 4'}, {'d': 'M 1 3.123466 L 2 4'})
    assert not same({'style': 'fill: #000000'}, {'style': 'fill: #000001'})
    assert not same('Measured 1.123456', 'Measured 1.123457')


def test_completion_gate_rejects_missing_changed_and_unresolved_evidence(tmp_path):
    from copy import deepcopy
    path = Path(__file__).resolve().parents[1]/'tools/check_figure_speed_completion.py'
    spec = importlib.util.spec_from_file_location('figure_completion_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/'receipt.json').write_text('{}')
    valid = {'required_actions': ['figure'], 'cases': [{'action': 'figure', 'executed': True,
             'before_seconds': [2], 'after_seconds': [1], 'parity': True, 'material_regression': False}],
             'source_stable': True, 'functions': [{'disposition': 'improved', 'coverage_note': 'executed'}],
             'residuals': [{'status': 'closed', 'evidence': 'receipt.json'}], 'receipts': ['receipt.json'],
             'focused_no_new_failures': True, 'project_gate_passed': True}
    assert module.check(valid, tmp_path)['complete']
    invalid = []
    for key, value in [('source_stable', False), ('cases', []), ('receipts', ['missing']),
                       ('project_gate_passed', False), ('focused_no_new_failures', False)]:
        changed = deepcopy(valid); changed[key] = value; invalid.append(changed)
    for key, value in [('parity', False), ('material_regression', True)]:
        changed = deepcopy(valid); changed['cases'][0][key] = value; invalid.append(changed)
    changed = deepcopy(valid); changed['residuals'][0]['status'] = 'open'; invalid.append(changed)
    assert all(not module.check(value, tmp_path)['complete'] for value in invalid)
    import json
    changed = deepcopy(valid)
    changed['cases'][0].update(material_regression=True, reviewed_evidence='review.json')
    review = [{'action': 'figure', 'status': 'open', 'reason': 'Still investigating',
               'evidence': ['receipt.json']}]
    (tmp_path/'review.json').write_text(json.dumps(review))
    assert not module.check(changed, tmp_path)['complete']
    review[0].update(status='resolved', reason='Controlled follow-up below threshold')
    (tmp_path/'review.json').write_text(json.dumps(review))
    assert module.check(changed, tmp_path)['complete']
    review[0]['evidence'] = ['missing.json']
    (tmp_path/'review.json').write_text(json.dumps(review))
    assert not module.check(changed, tmp_path)['complete']
