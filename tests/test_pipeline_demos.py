"""All six public workflows verify their complete synthetic reports and saved replay."""
import importlib
import pytest


@pytest.mark.parametrize('family',['rhythm','audit','relationships','behaviour','coordination','intervention'])
def test_public_pipeline_demo(family,tmp_path):
    demo=importlib.import_module(f'tests.pipelines.{family}.demo')
    root=demo.prepare(tmp_path/family)
    proof=demo.verify(root)
    assert proof
    assert (root/'verification.json').is_file()
