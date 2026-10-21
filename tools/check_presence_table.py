"""Check the working-array presence table against direct per-identity scans.

tools/presence_probe/site_probe.py is registered as an audited site for this
check only; results, dtypes and types must equal the plain NumPy answers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--storage', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    here = Path(__file__).parent
    sys.path[:0] = [str(here / 'presence_probe'), str(here)]
    from benchmark_working_storage import load
    st = load(a.storage.resolve())
    probe = here / 'presence_probe' / 'site_probe.py'
    st.PRESENCE_SITES['site_probe.py'] = (hashlib.sha256(probe.read_bytes()).hexdigest(), (3, 4, 5, 6))
    import site_probe
    rng = np.random.default_rng(3)
    values = [0, 1, 5, 436, 437, 65535, 70000, -1, np.int64(5), np.uint16(7), np.int32(-3),
              np.uint64(2**63 + 5), 2**70, True, 3.0]
    results = {}
    with tempfile.TemporaryDirectory(prefix='presence-check-') as scratch:
        pool = st.Pool(Path(scratch), minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
        for dtype in (np.uint16, np.int32, np.int64):
            raw = rng.integers(0, 437, (40, 120, 130)).astype(dtype)
            raw[3] = 0
            if dtype is np.int32:
                raw[5, 0, 0] = -3
            expected = site_probe.loops(raw, values)
            with st.install(pool):
                stack = pool.array(raw, copy=True)
                assert isinstance(stack, st.WorkingArray)
                actual = site_probe.loops(stack, values)
                del stack
            for e, f in zip(expected, actual):
                for x, y in zip(e, f):
                    assert type(np.asarray(x)) is type(np.asarray(y))
                    assert np.asarray(x).dtype == np.asarray(y).dtype and np.array_equal(x, y)
            results[np.dtype(dtype).name] = dict(st.presence_stats)
        pool.close()
    report = dict(status='pass', counters=results)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
