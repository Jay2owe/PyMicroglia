"""Check the Boolean-array nonzero positions in three adapters against np.flatnonzero, exactly.

python tools/check_boolean_positions.py --adapter <identity_candidate> --output <report.json>

raw_support_memo, observation_cache and optical_physical_paths find nonzero
positions with np.flatnonzero(x != 0) where x is numeric. Each expression is
read from the adapter's own source and evaluated against np.flatnonzero(x)
for every NumPy numeric kind (Boolean, signed and unsigned integers, floats,
complex) holding zeros, negative zero, NaN, infinities, subnormals, extremes
and complex numbers with one zero part, in empty, 1-D, 2-D, 3-D,
non-contiguous, memmap and working-storage arrays, and for the non-numeric
inputs each guard sends to the original call (object arrays, strings, lists,
scalars). Positions, dtype and raised errors must be identical.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

SITES = {
    'raw_support_memo': (r'where = (np\.flatnonzero\(owners != 0\))\n', 'owners'),
    'observation_cache': (r'where = (np\.flatnonzero\(flat != 0\) if .*?)\n', 'flat'),
    'optical_physical_paths': (r'positions = (\(np\.flatnonzero\(nodes != 0\).*?\))\n    if', 'nodes'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    sys.path.insert(0, str(a.adapter.resolve()))
    from array_storage import Pool
    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    pool = Pool(Path(temporary.name) / 'pool', minimum_bytes=64, temporary_ram_bytes=0)
    functions = {}
    for module, (pattern, name) in SITES.items():
        source = (a.adapter / (module + '.py')).read_text(encoding='utf-8')
        found = re.findall(pattern, source, re.S)
        assert len(found) == 1, (module, found)
        functions[module] = eval('lambda %s: %s' % (name, ' '.join(found[0].split())), {'np': np})

    def outcome(function, value):
        try:
            result = function(value)
            return ('ok', result.dtype.str, result.tolist())
        except Exception as error:
            return ('error', type(error).__name__, str(error))

    specials = [0, 1, -1, 2, 0.5, -0.0, np.nan, np.inf, -np.inf, 5e-324, 1e-45, 1e300, 1j, complex(0, -0.0),
                complex(np.nan, 0), complex(0, np.nan), True, False]
    kinds = [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
             np.float16, np.float32, np.float64, np.longdouble, np.complex64, np.complex128]
    rng = np.random.default_rng(3)
    report = dict(status='pass', compared=0, arrays=0, errors=0)
    cases = []
    for number, kind in enumerate(kinds):
        info = np.iinfo(kind) if np.dtype(kind).kind in 'iu' else None
        values = []
        for value in specials + ([info.min, info.max] if info else []):
            try:
                with np.errstate(all='ignore'):
                    cast = np.array(value).astype(kind)
            except (TypeError, ValueError, OverflowError):
                continue
            values.append(cast)
        base = np.array(values, dtype=kind)
        mixed = rng.choice(base, size=(5, 7, 9))
        mixed[rng.random(mixed.shape) < .6] = 0
        cases += [('empty', np.zeros((0,), kind)), ('values', base), ('2-D', mixed[0]), ('3-D', mixed),
                  ('strided', mixed[:, ::2, 1::3]), ('fortran', np.asfortranarray(mixed[0])),
                  ('zeros', np.zeros((4, 4), kind)), ('scalar array', np.array(base[-1]))]
        store = np.memmap(Path(temporary.name) / ('m_%d' % number), dtype=kind, mode='w+',
                          shape=mixed.shape)
        store[:] = mixed
        cases.append(('memmap', store))
        cases.append(('working', pool.array(mixed, copy=True, force=True)))
    cases += [('object', np.array([0, 1, None, 0.0, 'a'], dtype=object)), ('strings', np.array(['', 'a', '0'])),
              ('list', [0, 1, 0, 2]), ('scalar', 3), ('zero scalar', 0), ('None', None)]
    for label, value in cases:
        for module, function in functions.items():
            expected = outcome(np.flatnonzero, value)
            actual = outcome(function, value)
            numeric = isinstance(value, np.ndarray) and value.dtype.kind in 'biufc'
            if module == 'raw_support_memo' and not numeric:
                # Reached only for plain contiguous integer ndarrays.
                continue
            if module == 'observation_cache' and not isinstance(value, np.ndarray):
                # flat is always a flattened ndarray there.
                continue
            assert expected == actual, (module, label, getattr(value, 'dtype', type(value)), expected, actual)
            report['compared'] += 1
            report['errors'] += expected[0] == 'error'
        report['arrays'] += 1
    del cases
    pool.close()
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
