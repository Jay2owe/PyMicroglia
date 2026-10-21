"""Result assembly uses the same input directories as the frozen tracker."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia.tracking import engine


@pytest.mark.parametrize('absolute',[False,True])
@pytest.mark.parametrize('bounded',[False,True])
def test_completed_result_resolves_separate_registered_and_motion_directories(
    tmp_path,monkeypatch,absolute,bounded,
):
    metadata=tmp_path/'metadata';metadata.mkdir()
    registered=tmp_path/'inputs'/'camera';registered.mkdir(parents=True)
    motion=tmp_path/'inputs'/'motion';motion.mkdir()
    qc=tmp_path/'inputs'/'qc';qc.mkdir()
    raw=registered/'sample.tif'
    evidence=qc/'sample_evidence.tif'
    tifffile.imwrite(raw,np.ones((4,7,9),np.uint16),photometric='minisblack')
    tifffile.imwrite(evidence,np.ones((4,7,9),np.uint16),photometric='minisblack')
    config=metadata/'config.json'
    config.write_text(json.dumps(dict(
        registered_input_dir=str(registered) if absolute else '../inputs/camera',
        motion_input_dir=str(motion) if absolute else '../inputs/motion',
        source_frame_offset=2,
    )))
    pins=dict(
        registered_raw={'relative_to_registered_input_dir':'sample.tif'},
        motion_composite={'relative_to_motion_input_dir':'../qc/sample_evidence.tif'},
    )
    inputs=metadata/'motion_inputs.json'
    inputs.write_text(json.dumps(dict(stems=['sample'],prepared=dict(
        sample=dict(config=str(config),pins=pins),
    ))))
    output=tmp_path/'output';output.mkdir()
    root=output/'engine';root.mkdir()
    calls=[]
    def finished_calculation(root,stem,run_name,config,log,**options):
        calls.append(options)
        accepted=root/'m23_completed_feedback'/run_name/'out';accepted.mkdir(parents=True)
        tifffile.imwrite(accepted/f'{stem}.tif',np.ones((2,7,9),np.uint16),
                         photometric='minisblack')
    monkeypatch.setattr(engine,'_engine',lambda _:root)
    monkeypatch.setattr(engine,'_run_frozen',finished_calculation)
    result=engine.run(inputs,output,bounded_memory=bounded)
    assert result.raw==raw and result.evidence==evidence
    assert result.source_frame_offset==2
    assert len(calls)==1
    assert calls[0].get('bounded_memory',False)==bounded
