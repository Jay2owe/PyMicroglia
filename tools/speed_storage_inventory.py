"""Metadata-only sizes for this task's own benchmark products."""
from collections import defaultdict
import json
import shutil
from benchmark_pipeline_execution import ROOT
root=ROOT/'development/speed-pass-20261011'
sizes=defaultdict(int);files=[]
for p in root.rglob('*'):
    if not p.is_file():continue
    n=p.stat().st_size;relative=p.relative_to(root)
    sizes[relative.parts[0]]+=n;files.append((n,relative.as_posix()))
print(json.dumps({'free_bytes':shutil.disk_usage(root).free,'groups':dict(sizes),
                  'largest':sorted(files,reverse=True)[:20]},indent=2))
