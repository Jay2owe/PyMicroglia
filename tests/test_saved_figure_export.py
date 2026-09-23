"""A saved failure remains a failure when drawn through the packaged exporter."""
from pathlib import Path
from types import SimpleNamespace
import json


def test_saved_failure_exports_without_refitting(tmp_path, monkeypatch):
    from tests.test_behaviour_figures import unavailable_context
    from pymicroglia.pipelines.behaviour import figures as display
    from pymicroglia.pipelines._contracts import Settings
    from pymicroglia.pipelines._saved_figures import draw_batch
    from pymicroglia import workbench
    from auto_organotypic.store.ledger import read
    def refused(*args, **kwargs):
        raise AssertionError("Reopening a saved result attempted fresh analysis")
    monkeypatch.setattr(workbench,"estimate_trace",refused)
    context=unavailable_context(tmp_path,"failed")
    values,statistics,metadata=display.support_data(context)
    saved,metadata=display.snapshot(context,values,statistics,metadata)
    options,_=display.options(Settings())
    pages=display.pages(values,metadata,options)
    run=tmp_path/"run"
    run.mkdir()
    (run/"manifest.json").write_text(json.dumps({"movies":[]}),encoding="utf-8")
    target=tmp_path/"rendered"
    context=SimpleNamespace(output=target,presentation_id="controlled-display",
        dependencies={"state-display":saved},table_paths={},
        request=SimpleNamespace(inputs=SimpleNamespace(source_run=str(run))))
    result=draw_batch(context,slug="state-support-summary",options=[options],
        aliases=display.ALIASES,data={"values":values,"statistics":statistics,"metadata":metadata,"pages":pages},
        cache_name="_behaviour_display_sources",sources=[display.__file__],text={},
        claim="Saved failed analysis remains explicitly failed.",grammar="small-multiples")
    assert len(result["masters"])==1
    master=target/result["masters"][0]
    assert master.is_file() and master.with_suffix(".csv").is_file()
    assert "failed" in master.with_suffix(".csv").read_text(encoding="utf-8")
    assert not list(target.glob("*_provenance.json"))
    assert not list(target.glob("*_bundle"))
    assert read(target)


def test_clock_face_views_export_recorded_estimates(tmp_path):
    import pandas as pd
    from pymicroglia.figure_tables.actions import draw
    from pymicroglia.visualisation.figures import available_views
    run=tmp_path/"run"
    (run/"pooled").mkdir(parents=True)
    (run/"manifest.json").write_text(json.dumps({"movies":[{"stem":"A1","modules":[]}]}),encoding="utf-8")
    pd.DataFrame({"identity":[1,2],"metric":["area_px"]*2,"best_period_hours":[8.,12.],
                  "best_phase_hours":[2.,4.],"best_p_value":[.01,.02],"rhythmic":[True,True],
                  "free_cosinor_amplitude":[3.,4.]}).to_csv(run/"pooled/rhythms.csv",index=False)
    assert available_views("clock_face")==("dial","rose","histogram")
    result=draw("clock_face",run,view="rose",metrics="area_px")
    assert result["figures"][0].name=="A1_rose.svg"
    values=pd.read_csv(result["table"])
    assert values["count"].sum()==2 and set(values.view)=={"rose"}
