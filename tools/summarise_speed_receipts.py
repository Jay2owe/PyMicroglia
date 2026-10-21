"""Inventory locally generated action receipts; failures never become coverage."""
import json
from benchmark_pipeline_execution import ROOT

base=ROOT/'development/speed-pass-20261011'
required=json.loads((base/'scope.json').read_text())['required']
seen=set();success=set();failures={};rows=[]
for path in (base/'complete').rglob('receipt.json'):
    data=json.loads(path.read_text(encoding='utf-8'))
    for row in data.get('operations',[data]):
        if 'id' not in row:continue
        seen.add(row['id'])
        if row.get('status')=='measured':success.add(row['id'])
        elif row.get('error'):failures[row['id']]=row['error'].splitlines()[-1]
        rows.append({'receipt':str(path.relative_to(base)),**{k:row[k] for k in
            ('id','status','seconds','peak_rss_bytes','label') if k in row}})
summary={'measured':len(success),'unattempted':[r for r in required if r not in seen],
         'failed':{k:v for k,v in failures.items() if k not in success},'receipts':rows}
(base/'complete/execution-inventory.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='receipts'},indent=2))
