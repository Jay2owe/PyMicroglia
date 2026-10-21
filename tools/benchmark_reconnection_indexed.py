"""Run the existing complete reconnection benchmark with indexed nomination."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys


def main():
    adapter=Path(sys.argv[sys.argv.index('--adapter')+1]).resolve()
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    sys.path.insert(0,str(adapter))
    import reconnection_area_reuse as areas
    import reconnection_candidate_index as index
    from benchmark_reconnection_partition import main as benchmark
    original=areas.install
    @contextmanager
    def combined(module):
        with index.install(module) as nominations, original(module) as measurements:
            assert nominations['supported'] and measurements['supported']
            yield dict(supported=True,nominations=nominations,measurements=measurements)
    areas.install=combined
    try:benchmark()
    finally:
        areas.install=original
        path=output/'report.json'
        if path.exists():
            report=json.loads(path.read_text())
            for source in (Path(__file__),Path(index.__file__)):
                report['sources'][str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
            report['scope']+='; indexed resident nomination and scalar area reuse'
            path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
