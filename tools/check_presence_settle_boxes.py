"""Compare boxed carve settlement and changed-frame validation with presence.py.

Carves are discs and grown blobs placed over real accepted-label frames (hosts
cut in two, swallowed whole, cut below the host floor, touching the frame
edge, several hosts at once), plus random masks over synthetic frames, on
uint16, int32 and memmap frames. Validation runs on real completions with
recovered pixels, unchanged movies, and movies that alter an unclaimed pixel,
leave a claimed pixel empty, or carve a name away. Results, refusal strings and
raised errors must be equal.
"""
import argparse, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def outcome(function, *args):
    try:
        value = function(*args)
    except Exception as error:
        return ('error', type(error).__name__, str(error))
    if value is None:
        return ('none',)
    mask, refused = value
    return ('ok', mask.dtype.str, mask.shape, mask.tobytes(), refused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import presence as module
    import presence_settle_boxes as adapter
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    rng = np.random.default_rng(20261021)
    yy, xx = np.indices(stack.shape[1:])
    folder = Path(tempfile.mkdtemp(prefix='settle_check_'))
    cases = []
    for t in rng.choice(len(stack), 60, replace=False):
        frame = stack[t]
        for _ in range(12):
            y, x = int(rng.integers(0, frame.shape[0])), int(rng.integers(0, frame.shape[1]))
            if rng.random() < .5:
                ids = np.unique(frame)[1:]
                cell = frame == int(rng.choice(ids))
                cy, cx = np.argwhere(cell)[int(rng.integers(0, cell.sum()))]
                y, x = int(cy), int(cx)
            radius = float(rng.choice([2, 3, 5, 8, 14]))
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
            if rng.random() < .3:
                mask = ndi.binary_dilation(mask, iterations=int(rng.integers(1, 4))) & (rng.random(mask.shape) < .9)
            params = dict(minimum_host_area_px=int(rng.choice([1, 10, 40, 120])))
            cases.append((mask, frame, params))
            if len(cases) % 25 == 0:
                cases.append((mask, frame.astype(np.int32), params))
                mapped = np.memmap(folder / f'f{len(cases)}.dat', mode='w+', dtype=frame.dtype, shape=frame.shape)
                mapped[:] = frame
                cases.append((mask, mapped, params))
    for _ in range(300):
        shape = (int(rng.integers(3, 30)), int(rng.integers(3, 30)))
        frame = rng.integers(0, 6, shape).astype(np.uint16) * (rng.random(shape) < .7)
        mask = rng.random(shape) < rng.choice([.05, .3, .7])
        cases.append((mask, frame, dict(minimum_host_area_px=int(rng.integers(0, 6)))))
    cases.append((np.zeros((5, 5), bool), np.zeros((5, 5), np.uint16), dict(minimum_host_area_px=1)))
    cases.append((np.ones((5, 5), int), np.ones((5, 5), np.uint16), dict(minimum_host_area_px=1)))
    original_settle, original_validate = module._settle_carves, module._validate
    failures, seconds = [], dict(original=0.0, boxed=0.0)
    with adapter.install(module) as stats:
        boxed_settle, boxed_validate = module._settle_carves, module._validate
        for number, case in enumerate(cases):
            t0 = time.perf_counter(); e = outcome(original_settle, *case)
            t1 = time.perf_counter(); f = outcome(boxed_settle, *case)
            t2 = time.perf_counter()
            seconds['original'] += t1 - t0; seconds['boxed'] += t2 - t1
            if e != f:
                failures.append(dict(kind='settle', number=number, expected=str(e)[:120], actual=str(f)[:120]))
        validations = []
        window = stack[100:160]
        for _ in range(30):
            recovered = np.zeros(window.shape, bool)
            output = window.copy()
            for t in rng.choice(len(window), int(rng.integers(0, 6)), replace=False):
                frame = output[t]
                ids = np.unique(frame)[1:]
                victim = int(rng.choice(ids))
                taker = int(rng.choice(ids))
                pick = frame == victim
                if rng.random() < .7:
                    pick &= rng.random(pick.shape) < .5
                output[t][pick] = taker
                recovered[t][pick] = True
            choice = rng.random()
            if choice < .1:
                output[int(rng.integers(0, len(window)))][0, 0] += 1
            elif choice < .2 and recovered.any():
                where = np.argwhere(recovered)[0]
                output[tuple(where)] = 0
            validations.append((output, window, recovered))
        validations.append((window.copy(), window, np.zeros(window.shape, bool)))
        validations.append((window.copy(), window, np.zeros(window.shape, np.uint8)))
        for number, case in enumerate(validations):
            t0 = time.perf_counter(); e = outcome(original_validate, *case)
            t1 = time.perf_counter(); f = outcome(boxed_validate, *case)
            t2 = time.perf_counter()
            seconds['original'] += t1 - t0; seconds['boxed'] += t2 - t1
            if e != f:
                failures.append(dict(kind='validate', number=number, expected=str(e)[:200], actual=str(f)[:200]))
        counters = dict(stats)
    assert module._settle_carves is original_settle and module._validate is original_validate
    report = dict(status='pass' if not failures else 'fail', settle_cases=len(cases),
                  validate_cases=len(validations), seconds=seconds, counters=counters, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
