"""Display plans retain the original source byte identity and seeded row ordering."""
import hashlib,json
from pathlib import Path
import pytest
from pymicroglia._results import measurement_identity,read_document,write_document,figure_plans
from pymicroglia.pipelines._runner import register_figure_plan


@pytest.mark.parametrize("hidden",[False,True])
def test_original_byte_identity_survives_figure_plan_and_reopening(tmp_path,hidden):
    path=tmp_path/"manifest.json"
    manifest={"movies":[{"stem":"synthetic"}],"settings":{"seed":17}}
    if hidden: physical=write_document(path,manifest)
    else:
        path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        physical=path
    before=hashlib.sha256(physical.read_bytes()).hexdigest()
    assert measurement_identity(path)==before
    register_figure_plan(tmp_path,[{"name":"view","figure":"clock-face","options":{},"pipeline":{}}])
    assert measurement_identity(path)==before
    register_figure_plan(tmp_path,[{"name":"other","figure":"clock-face","options":{},"pipeline":{}}])
    assert measurement_identity(path)==before
    assert hashlib.sha256(physical.read_bytes()).hexdigest()==before
    assert set(figure_plans(tmp_path))=={"view","other"}
    current=read_document(path)
    current["settings"]["seed"]=99
    write_document(path,current)
    assert measurement_identity(path)!=before


def test_earlier_inline_plans_remain_readable_and_conflicts_are_refused(tmp_path):
    from pymicroglia._results import SOURCE_IDENTITY_KEY
    from pymicroglia.pipelines._contracts import content_id
    base={"movies": [], "settings": {"seed": 17}}
    old={"name":"view","figure":"clock-face","options":{},"pipeline":{}}
    manifest={**base,"figure_plans":{"view":old}, SOURCE_IDENTITY_KEY:{
        "manifest_sha256":"original-byte-hash","content_sha256":content_id(base)}}
    path=write_document(tmp_path/"manifest.json",manifest)
    before=path.read_bytes()
    register_figure_plan(tmp_path,[old])
    assert path.read_bytes()==before
    assert measurement_identity(path)=="original-byte-hash"
    assert figure_plans(tmp_path)=={"view":old}
    changed={**old,"options":{"bins":5}}
    write_document(tmp_path/"figure-plans.json",{"figure_plans":{"view":changed}})
    with pytest.raises(ValueError,match="Conflicting saved figure plan"):
        figure_plans(tmp_path)
    manifest["settings"]["seed"]=99
    write_document(tmp_path/"manifest.json",manifest)
    with pytest.raises(ValueError,match="Measurement manifest changed"):
        measurement_identity(tmp_path/"manifest.json")
