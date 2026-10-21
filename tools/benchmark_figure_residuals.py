"""Bound remaining repeated settings work and inspect the existing native grid alternative."""
import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from ai_research_assistant.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args()
    from circadian_workbench import actions, arguments, analysis_grid
    from circadian_workbench.figure_definitions import FigureDefinition
    from pymicroglia import workbench
    counts = {'default_config': 0, '_to_config': 0}
    for file in (args.evidence/'catalogue-profile').glob('*.json'):
        for row in json.loads(file.read_text()).get('profile', []):
            if 'circadian_workbench' in row['file'] and row['function'] in counts:
                counts[row['function']] += row['calls']
    report = {'settings': [], 'note': 'Kernel timings are diagnostic bounds, not full export speedups. Nested costs are not added.'}
    with RunTracker('Measure remaining figure export costs', project='PyMicroglia') as tracker:
        with tracker.stage('Settings, decoding and native grid alternatives', total=4):
            with tracker.item('Saved table scalar decoding'):
                from pymicroglia.pipelines._tables import _cell_decoder
                table_path = Path(json.loads((args.evidence/'table-timings.json').read_text())['input'])
                with table_path.open(encoding='utf-8', newline='') as stream:
                    reader = csv.reader(stream)
                    next(reader)
                    cells = [cell for row in reader for cell in row]
                reference = [json.loads(cell) for cell in cells]
                samples = []
                for trial in range(4):
                    for mode in (['before', 'after'] if trial % 2 == 0 else ['after', 'before']):
                        decode = json.loads if mode == 'before' else _cell_decoder()
                        started = time.perf_counter()
                        actual = [decode(cell) for cell in cells]
                        elapsed = time.perf_counter()-started
                        assert actual == reference
                        samples.append({'trial': trial, 'mode': mode, 'seconds': elapsed})
                report['table_decode'] = {'input': str(table_path), 'cells': len(cells),
                                         'input_sha256': hashlib.sha256(table_path.read_bytes()).hexdigest(),
                                         'samples': samples, 'exact': True,
                                         'scope': 'Decode all saved CSV cells; excludes file reading and DataFrame construction.'}
            for name, function in [('default_config', arguments.default_config), ('_to_config', actions._to_config)]:
                with tracker.item(name):
                    expected = function()
                    samples = []
                    with workbench.analysis_session():
                        for _ in range(5):
                            started = time.perf_counter()
                            for index in range(100):
                                actual = function()
                            samples.append((time.perf_counter()-started)/100)
                            assert actual == expected
                    report['settings'].append({'function': name, 'seconds_per_call': samples,
                                               'profiled_catalogue_calls': counts[name],
                                               'empirical_upper_seconds_if_entire_cost_removed': max(samples)*counts[name],
                                               'assumption': 'Recorded call counts and ordinary default-sized settings; slowest measured trial, no profiler.'})
            with tracker.item('Native vector grid composition'):
                receipt = json.loads((args.evidence/'full-before1/cell_analysis_grid.json').read_text())
                path = Path(receipt['parameters']['output_dir'])/'parity_A1_statistics.csv'
                csv.field_size_limit(100_000_000)
                rows = list(csv.DictReader(path.open(encoding='utf-8', newline='')))
                accepted = {field.name for field in fields(FigureDefinition)}
                panels = []
                for row in rows:
                    data = json.loads(row['definition_json'])
                    definition = FigureDefinition(**{k: v for k, v in data.items() if k in accepted})
                    panels.append((definition.title, definition))
                shape = analysis_grid.grid_shape(len(panels), None, None)
                started = time.perf_counter()
                rendered = analysis_grid.render_svg(panels, rows=shape[0], columns=shape[1])
                seconds = time.perf_counter()-started
                native = ET.fromstring(rendered)
                current = ET.parse(receipt['files'][0]).getroot()
                current_text = [node.text for node in current.iter() if node.tag.endswith('}text')]
                native_text = [node.text for node in native.iter() if node.tag.endswith('}text')]
                report['native_grid_alternative'] = {
                    'seconds': seconds, 'bytes': len(rendered), 'panels': len(panels),
                    'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'current_viewbox': current.attrib.get('viewBox'), 'native_viewbox': native.attrib.get('viewBox'),
                    'same_text_sequence': current_text == native_text,
                    'current_text_missing_in_alternative': [text for text in current_text if text not in native_text],
                    'decision': 'Not substituted: native composition has its own grid sizing and wording placement; it does not preserve the current complete figure layout.',
                    'scope': 'Existing renderer exercised in memory; no new scientific analysis or deliverable written.'}
    (args.evidence/'residual-measurements.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
