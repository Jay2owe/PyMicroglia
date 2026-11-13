"""Compare the windowed whole-soma component with complete_stationary_soma_path._component.

python tools/check_stationary_component_window.py <engine code folder> <report.json>

Random owner frames (small blobs, thin long processes that cross windows,
components touching the frame edges, frames filled by one owner) and tracker
points (on the owner, beside it, outside the frame, non-finite, half-pixel
centres) are given to the original and the adapted function; the masks (type,
dtype, shape, values), None results and raised error types must be identical.
The installed adapter must replace the module function and restore it.
"""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

code, report_path = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path[:0] = [str(code.resolve()), str(here)]
module = importlib.import_module('complete_stationary_soma_path')
import stationary_component_window as adapter

# Held in a list: the install also re-points module-level aliases, this script's included.
held = [module._component]
stats = dict(versions=0, windowed=0, grown=0, fallback=0)
with adapter.install(stats):
    adapted = module._component
    installed = adapted is not held[0] and adapted.__wrapped__ is held[0]
original = held[0]
restored = module._component is original
rng = np.random.default_rng(1152)


def outcome(function, *args):
    try:
        return function(*args)
    except Exception as error:
        return type(error)


def same(a, b):
    if isinstance(a, type) or a is None:
        return a is b
    return type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def frame_of(kind, h, w, dtype):
    if kind == 'blobs':
        frame = ndi.label(ndi.binary_opening(rng.random((h, w)) < .55))[0] % 7
    elif kind == 'processes':
        frame = np.zeros((h, w), int)
        for owner in range(1, 6):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            for _ in range(int(rng.integers(20, 400))):
                frame[y, x] = owner
                y = int(np.clip(y + rng.integers(-1, 2), 0, h - 1))
                x = int(np.clip(x + rng.integers(-1, 2), 0, w - 1))
    elif kind == 'full':
        frame = np.full((h, w), 3)
        frame[rng.random((h, w)) < .02] = 0
    else:
        frame = (rng.random((h, w)) < .5).astype(int) * 2
    return frame.astype(dtype)


cases = equal = 0
for trial in range(400):
    h, w = int(rng.choice([1, 2, 9, 40, 150, 471])), int(rng.choice([1, 3, 40, 200, 471]))
    kind = ['blobs', 'processes', 'full', 'noise'][trial % 4]
    dtype = [np.uint16, np.int32, np.uint8, np.int64][trial % 4 if trial % 7 else 0]
    frame = frame_of(kind, h, w, dtype)
    if trial % 25 == 0:
        frame = frame.astype(float)
    on = np.argwhere(frame > 0)
    for _ in range(12):
        if len(on) and rng.random() < .6:
            y, x = on[int(rng.integers(0, len(on)))]
            y, x = float(y) + float(rng.choice([0, .49, .5, -.5, .51])), float(x) + float(rng.choice([0, .5, -.49]))
            owner = int(frame[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))]) or 1
        else:
            y, x = float(rng.uniform(-5, h + 5)), float(rng.uniform(-5, w + 5))
            owner = int(rng.integers(-1, 8))
        if rng.random() < .03:
            y = float(rng.choice([np.nan, np.inf]))
        if rng.random() < .03:
            owner = int(rng.choice([0, -2, 70000, 2**40]))
        point = SimpleNamespace(y=y, x=x, radius_px=float(rng.choice([0., 1.5, 4.])))
        cases += 1
        equal += same(outcome(original, frame, owner, point), outcome(adapted, frame, owner, point))
    for bad in (SimpleNamespace(y='a', x=1.0, radius_px=2.0), SimpleNamespace(x=1.0, radius_px=2.0)):
        cases += 1
        equal += same(outcome(original, frame, 2, bad), outcome(adapted, frame, 2, bad))
checks = dict(installed=bool(installed and restored), cases=cases, equal=int(equal), stats=stats)
passed = checks['installed'] and cases == equal and stats['versions'] == 1 and stats['windowed'] > 0 and stats['grown'] > 0
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
