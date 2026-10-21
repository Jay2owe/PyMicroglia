"""Paired saved-view action timings on deterministic, local synthetic data.

No laboratory input is read. Baseline files are a copy of the pre-edit checkout.
Retains raw timings, exact response hashes and source identities; not an analysis
pipeline or a claim about model inference / remote storage performance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import tifffile


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    recipe = {"seed": 111026, "frames": 48, "shape": [512, 768], "cells": 96}
    if (root / 'recipe.json').exists():
        assert json.loads((root / 'recipe.json').read_text()) == recipe
        return root
    rng = np.random.default_rng(recipe['seed'])
    raw = rng.integers(0, 4000, (48, 512, 768), dtype=np.uint16)
    labels = np.zeros_like(raw)
    for f in range(48):
        for i in range(96):
            y, x = 12 + i // 12 * 60, 10 + i % 12 * 62 + f % 4
            labels[f, y:y+16, x:x+18] = i+1
    for name, data in [('raw', raw), ('labels', labels)]:
        tifffile.imwrite(root / (name+'.tif'), data, photometric='minisblack', compression='zlib')
    with (root/'cell_frame.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['identity','frame_index','hours','integrated_density','area_px','centroid_x','centroid_y'])
        for i in range(1,97):
            for f in range(48):
                writer.writerow([i,f,f,100+20*np.cos(2*np.pi*(f-i)/24),288,19+(i-1)%12*62+f%4,20+(i-1)//12*60])
    (root/'manifest.json').write_text(json.dumps({'movies':[{'stem':'synthetic',
        'spec':{'raw':'raw.tif','labels':'labels.tif'},'display':{'contrast':[0,4000]},
        'tables':{'cell_frame':{'path':'cell_frame.csv'}}}]}))
    (root/'recipe.json').write_text(json.dumps(recipe))
    return root


def baseline(path, name):
    spec = importlib.util.spec_from_file_location('pymicroglia.ui._speed_'+name, path/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run(out, repeats):
    from pymicroglia.ui import exploration_images as after, exploration_previews as previews
    root = fixture(out/'fixture')
    before = baseline(out/'baseline','exploration_images')
    before_preview = baseline(out/'baseline','exploration_previews')
    actions = {
        'full_field': lambda m: m.load_frame(root,'synthetic',20),
        'source_frame': lambda m: m.load_frame_image(root,'synthetic',20),
        'selected_cell': lambda m: m.load_cell_frame(root,'synthetic',1,20),
        'changed_pixels': lambda m: m.load_change(root,'synthetic',1,20),
        'outline_history': lambda m: m.load_history(root,'synthetic',1,20),
        'cell_strip': lambda m: m.load_cell_images(root,'synthetic',1),
    }
    result = {'python':platform.python_version(),'platform':platform.platform(),'repeats':repeats,'actions':{}}
    # Warm imports and compressed file pages; cache-miss trials still clear the
    # application plane cache. No OS cache manipulation on the shared host.
    after.load_cell_frame(root,'synthetic',1,0)
    for name, call in actions.items():
        samples = {'before':[], 'after':[]}
        equality = []
        for iteration in range(repeats):
            values = {}
            for label, module in ([('before',before),('after',after)] if iteration%2==0 else [('after',after),('before',before)]):
                module._plane.cache_clear(); module._contrast.cache_clear()
                start = time.perf_counter(); values[label] = call(module)
                samples[label].append(time.perf_counter()-start)
            equality.append(digest(values['before']) == digest(values['after']))
        result['actions'][name] = {'samples_seconds':samples,'response_exact':all(equality),
            'median_seconds':{k:statistics.median(v) for k,v in samples.items()}}
        print(name,result['actions'][name]['median_seconds'], 'exact',all(equality),flush=True)
    samples={'before':[],'after':[]};values={}
    for i in range(repeats):
        for label,module in ([('before',before_preview),('after',previews)] if i%2==0 else [('after',previews),('before',before_preview)]):
            module._phases.cache_clear()
            start=time.perf_counter(); values[label]=module.load_phase_preview(root,'synthetic')
            samples[label].append(time.perf_counter()-start)
    result['actions']['phase_preview']={'samples_seconds':samples,'response_exact':digest(values['before'])==digest(values['after']),
        'median_seconds':{k:statistics.median(v) for k,v in samples.items()}}
    result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    result['code_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(after.__file__),Path(previews.__file__)]}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--label',required=True);parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args();result=run(args.out.resolve(),args.repeats)
    (args.out/(args.label+'.json')).write_text(json.dumps(result,indent=2))
