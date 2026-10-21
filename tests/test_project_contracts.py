"""Bindings to the live catalogue and linked image/trace selection helper."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_catalogue_uses_live_registries():
    from pymicroglia.ui.catalog import load_map, validate_map
    assert validate_map(load_map()) == []


def test_linked_selection_roundtrip_and_missing_observations():
    source = r"""
    const assert = require('node:assert/strict');
    const C = require('./src/pymicroglia/ui/static/exploration-core.js');
    const data = {run:'first',stem:'recording',identities:[11,22],frames:[0,1,2,3]};
    const original = {identity:11,frame:2,metric:'signal_mean'};
    const changed = C.selection(data, original, {identity:22});
    assert.equal(changed.identity,22);assert.equal(changed.frame,2);
    assert.equal(original.identity,11);
    const restored = C.selection(data, {}, JSON.parse(JSON.stringify(changed)));
    assert.deepEqual(restored,changed);
    assert.equal(C.selection(data,restored,{frame:500}).frame,3);
    assert.equal(C.selection({...data,run:'second',identities:[33]},restored).identity,33);
    const raw=[1,null,3,4],fit=[1,2,3,4];
    const scaled=C.scaleSeries(raw,fit,true);
    assert.equal(scaled.values[1],null);assert.deepEqual(raw,[1,null,3,4]);
    assert.deepEqual(C.alignFit({indices:[0,1,2,3],values:fit},data.frames,raw),[1,null,3,4]);
    """
    subprocess.run(["node", "-e", source], cwd=ROOT, check=True, timeout=15)
