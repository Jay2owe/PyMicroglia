from pathlib import Path
import importlib.util
import pytest


@pytest.fixture
def configuration():
    source=Path(__file__).parents[1]/'development/cd68_full_red_green/configure_rigid_batch.py'
    spec=importlib.util.spec_from_file_location('rigid_dependency_test',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_motion_preparation_uses_original_channels_without_learned_mask(configuration):
    roles=['red_original','green_original','tracking_proxy','validity_canvas','source_to_canvas','frame_clock']
    manifest={'outputs':[{'role':role,'path':role+'.file'} for role in roles]}
    assert configuration.preparation_inputs(manifest)=={
        role:role+'.file' for role in ['red_original','green_original','source_to_canvas','frame_clock']}
    assert [row['role'] for row in manifest['outputs']]==roles


def test_missing_mask_input_is_rejected(configuration):
    with pytest.raises(ValueError,match='Missing original-channel'):
        configuration.preparation_inputs({'outputs':[{'role':'tracking_proxy','path':'red.tif'}]})
