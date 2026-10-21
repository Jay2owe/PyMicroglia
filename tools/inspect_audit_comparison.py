"""Inspect exact native confirmation differences in the retained speed trials."""
from pathlib import Path
import json
from pymicroglia._results import read_document
from pymicroglia.pipelines._contracts import content_id

root = Path(__file__).resolve().parents[1] / 'development/speed-pass-20261011/complete'
records = []
for mode in ['before', 'after']:
    trial = root / f'pipeline-audit-{mode}/audit'
    receipt = json.loads((trial / 'receipt.json').read_text())
    name = next(k for k in receipt['outputs'] if '/science/' in k and k.endswith('/confirmation_record.json'))
    data = read_document(trial / name)
    print(mode, 'valid digest', data['confirmation_id'] == content_id({k:v for k,v in data.items() if k != 'confirmation_id'}))
    records.append(data)

def compare(a, b, key=''):
    if type(a) != type(b):
        print(key, 'type'); return
    if isinstance(a, dict):
        for name in set(a) | set(b): compare(a.get(name), b.get(name), key + '.' + name)
    elif isinstance(a, list):
        if len(a) != len(b): print(key, 'length')
        for i, (left, right) in enumerate(zip(a,b)): compare(left, right, key + f'[{i}]')
    elif a != b: print(key, str(a)[:100], str(b)[:100])

compare(*records)

decisions=[]
for mode in ['before','after']:
    trial=root/f'pipeline-behaviour-{mode}/behaviour'
    receipt=json.loads((trial/'receipt.json').read_text())
    name=next(k for k in receipt['outputs'] if '/science/' in k and k.endswith('/support_decision.json'))
    data=read_document(trial/name)
    print(mode,'behaviour decision valid',data['decision_id']==content_id({k:v for k,v in data.items() if k!='decision_id'}))
    decisions.append(data)
compare(*decisions)
