"""Embedded evidence survives normalization; failed promotion keeps the old file."""
import pytest


def test_statistics_with_missing_values_roundtrip(tmp_path):
    from matplotlib import pyplot as plt
    from pymicroglia.visualisation.panels import save
    from reprofig import validate_artifact
    fig,ax=plt.subplots()
    try:
        ax.plot([0,1],[1,2])
        save(fig,tmp_path/'evidence',table={'x':[0,1],'y':[1,2]},
             statistics=[{'p_value':float('nan'),'estimate':2.}],statistics_status='complete',formats=('svg',))
        assert validate_artifact(tmp_path/'evidence.svg').valid
    finally:
        plt.close(fig)


def test_failed_delivery_preserves_existing_figure(tmp_path,monkeypatch):
    from pymicroglia.visualisation._delivery import render_destination
    import auto_organotypic.io as io
    target=tmp_path/'existing.svg';target.write_bytes(b'original')
    def refuse(*args,**kwargs):
        raise PermissionError('Controlled persistent lock')
    monkeypatch.setattr(io,'replace_with_retry',refuse)
    with pytest.raises(PermissionError):
        with render_destination(target) as staged:
            staged.write_bytes(b'replacement')
    assert target.read_bytes()==b'original'
    assert sorted(p.name for p in tmp_path.iterdir())==['existing.svg']
