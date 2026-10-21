"""Check exact immutable component caching and mapping alias lifetime."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-component-cache-check'
    sys.path[:0]=[str(a.code.resolve()),str(a.adapter.resolve())]
    import numpy as np
    native=importlib.import_module('component_accounting')
    adapter=importlib.import_module('frame_component_reuse')
    original=native.label_value_components
    calls=[0]
    def counted(frame):
        calls[0]+=1
        return original(frame)
    accounting=SimpleNamespace(label_value_components=counted)
    rng=np.random.default_rng(23)
    labels=rng.integers(0,7,(8,31,37),dtype=np.int32)
    labels[0]=0
    def discover(movie):
        for repeat in range(3):
            for index in reversed(range(len(movie))):
                expected=original(movie[index])
                actual=accounting.label_value_components(movie[index])
                for left,right in zip(expected,actual):
                    np.testing.assert_array_equal(left,right)
                    assert left.dtype==right.dtype
        return True
    module=SimpleNamespace(component_accounting=accounting,discover=discover)
    with tempfile.TemporaryDirectory(prefix='component-reuse-',dir=a.output.parent) as folder:
        with adapter.install(module,folder) as counters:
            assert module.discover(labels)
            assert calls[0]==len(labels)
            labels[2]=3
            assert module.discover(labels)
            assert calls[0]==2*len(labels)
            assert not list(Path(folder).iterdir())
            # A detached copy is not the same immutable input view.
            def unrelated(movie):
                frame=movie[0].copy()
                frame[:]=4
                actual=accounting.label_value_components(frame)
                expected=original(frame)
                for left,right in zip(expected,actual):
                    np.testing.assert_array_equal(left,right)
            plain=SimpleNamespace(component_accounting=accounting,discover=unrelated)
            with adapter.install(plain,folder) as other:
                plain.discover(labels)
                assert other['uncached_calls']==1
        assert accounting.label_value_components is counted
        # Callers may retain ndarray views after discovery; mappings live until
        # the last alias is released, then remove their single scratch file.
        def aliases(movie):
            mapped=accounting.label_value_components(movie[1])[0]
            return np.asarray(mapped)[::2,::-1]
        module.discover=aliases
        with adapter.install(module,folder):
            view=module.discover(labels)
        np.testing.assert_array_equal(view,original(labels[1])[0][::2,::-1])
        assert not view.flags.writeable
        assert len(list(Path(folder).iterdir()))==1
        del view
        gc.collect()
        assert not list(Path(folder).iterdir())
        def arithmetic(movie):
            mapped=accounting.label_value_components(movie[1])[0]
            mask=mapped==1
            assert type(mask) is np.ndarray and mask.dtype==np.bool_
            return mask
        module.discover=arithmetic
        with adapter.install(module,folder):
            mask=module.discover(labels)
        np.testing.assert_array_equal(mask,original(labels[1])[0]==1)
        assert not list(Path(folder).iterdir())
        def failure(movie):
            mapped=accounting.label_value_components(movie[0])[0]
            raise ValueError('original failure preserved')
        module.discover=failure
        with adapter.install(module,folder):
            try:
                module.discover(labels)
            except ValueError as error:
                assert str(error)=='original failure preserved'
            else:
                raise AssertionError('Expected original error')
        gc.collect()
        assert not list(Path(folder).iterdir())
    report=dict(status='pass',coverage=['exact arrays/dtypes','empty frame','fresh call after edits',
        'foreign copies bypass cache','nested contexts','retained ndarray aliases','readonly results',
        'original exception preserved','scratch released after last alias',
        'arithmetic uses plain arrays and releases mapped storage'],counters=counters)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
