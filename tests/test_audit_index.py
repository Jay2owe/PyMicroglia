"""Saved audit reports preserve decision states, complete identities and portable links."""
from pymicroglia._results import read_document
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines.audit.index import copy_evidence, render_index
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines._runner import SavedResult
from pymicroglia.pipelines._screening import _write_json, file_hash, write_table

class Links(HTMLParser):

    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if 'href' in values:
            self.links.append(values['href'])
        if 'id' in values:
            self.ids.add(values['id'])

def validate_links(output):
    parser = Links()
    parser.feed((output / 'index.html').read_text(encoding='utf-8'))
    for href in parser.links:
        parts = urlsplit(href)
        assert not parts.scheme and (not parts.netloc)
        if parts.path:
            target = (output / unquote(parts.path)).resolve()
            assert target.is_relative_to(output.resolve()) and target.is_file()
        if parts.fragment:
            assert parts.fragment in parser.ids
    return parser.links

def report_fixture(root):
    candidates = [{'candidate_id': name, 'labels': ['same label <script>alert("test")</script>'], 'analysis_options': {'fit_method': 'lomb', 'significance_method': 'f'}} for name in ('recipe-a', 'recipe-b')]
    score = {'recovery_rate': 0.8, 'recovery_lower': 0.7, 'recovery_upper': 0.9, 'recovery_numerator': 80, 'recovery_denominator': 100, 'recovery_independent_units': 50, 'false_alarm_rate': 0.01, 'false_alarm_lower': 0, 'false_alarm_upper': 0.04, 'false_alarm_numerator': 1, 'false_alarm_denominator': 100, 'false_alarm_independent_units': 50}
    decisions = []
    for index, (state, confirmation) in enumerate((('confirmed_choice', 'supported'), ('tie', 'supported'), ('confirmation_failed', 'failed'), ('insufficient_evidence', 'skipped'))):
        decisions.append({'decision_id': str(index), 'measurement': 'metric ' + str(index), 'final_state': state, 'reason': 'Saved reason ' + str(index), 'candidate_ids': ['recipe-a'] if index != 1 else ['recipe-a', 'recipe-b'], 'confirmation_status': confirmation, 'confirmation_checks': [{'candidate_id': 'recipe-a', 'state': confirmation, 'score': score}] if index < 3 else []})
    assessment = [{'candidate_id': c['candidate_id'], 'measurement': 'metric 0', 'state': 'admissible', 'score': score, 'gates': [{'requirement': 'false_alarm_control', 'state': 'pass'}]} for c in candidates]
    definitions = {'audit-design': {'audit_design': {'scientific_id': 'audit', 'profile_count': 2, 'request': {'source': {'workbench_version': 'saved-version'}, 'request': {'population': {'mode': 'all'}}}}}, 'candidate-shortlist': {'frozen_selection': {'selection_id': 'selection', 'candidates': candidates}, 'candidate_assessments': pd.DataFrame(assessment)}, 'independent-confirmation': {'confirmation_record': {'selection_id': 'selection', 'confirmation_id': 'confirmation', 'family_compatibility': {'state': 'matched'}}, 'final_decisions': pd.DataFrame(decisions), 'confirmation_checks': pd.DataFrame()}, 'focused-pages': {'focused_manifest.json': {'rendered_pages': [], 'reason': 'Explicitly skipped examples'}}}
    saved = {}
    for step, artifacts in definitions.items():
        folder = root / step
        folder.mkdir(parents=True)
        refs = []
        for name, data in artifacts.items():
            path = folder / (name if name.endswith('.json') else name + '.json')
            path = (write_table if isinstance(data, pd.DataFrame) else _write_json)(path, data)
            refs.append(ArtifactRef(name, path.name, file_hash(path), 'saved-id'))
        saved[step] = SavedResult(folder, StepResult(step, 'saved-id', 'completed', 'Saved fixture', tuple(refs)))
    return saved

def test_all_states_and_distinct_recipes_remain_portable_without_science(tmp_path, monkeypatch):
    saved = report_fixture(tmp_path / 'sources')
    for name in ('estimate_one', 'adjust_pvalues', 'generate_benchmark_cases', 'benchmark_score_interval'):
        monkeypatch.setattr(circadian, name, lambda *a, **k: pytest.fail('index recomputed science'))
    output = tmp_path / 'report'
    output.mkdir()
    inventory = copy_evidence(saved, output)
    manifest = render_index(output, inventory)
    links = validate_links(output)
    text = (output / 'index.html').read_text(encoding='utf-8')
    for term in ('confirmed choice', 'tie', 'confirmation failed', 'insufficient evidence', '80/100', 'Explicitly skipped examples'):
        assert term in text
    assert '<script>' not in text and '&lt;script&gt;' in text
    assert not manifest['settings_applied'] and (not manifest['settings_exported'])
    assert len({r['target_id'] for r in manifest['candidate_targets']}) == 2
    for target in manifest['candidate_targets']:
        payload = read_document(output / target['path'])
        assert payload['target_id'] == content_id({k: v for k, v in payload.items() if k != 'target_id'})
    moved = tmp_path / 'moved report'
    shutil.copytree(output, moved)
    assert validate_links(moved) == links
    assert render_index(moved, manifest['inputs'])['report_id'] == manifest['report_id']
    for result in saved.values():
        for ref in result.outcome.artifacts:
            result.artifact(ref.name)

def test_changed_evidence_cannot_be_reported_as_a_saved_choice(tmp_path):
    saved = report_fixture(tmp_path / 'sources')
    output = tmp_path / 'report'
    output.mkdir()
    inventory = copy_evidence(saved, output)
    path = output / inventory['candidate-shortlist']['artifacts']['frozen_selection']['path']
    path.write_text('{}')
    with pytest.raises(ValueError, match='missing or changed'):
        render_index(output, inventory)
