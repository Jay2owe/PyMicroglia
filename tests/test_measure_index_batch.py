"""Measurement keeps its results when the shared store batches or is older."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tifffile


def test_measure_batches_durable_results_and_supports_older_store(tmp_path, monkeypatch):
    from pymicroglia import config, store
    from pymicroglia.measure import measure
    monkeypatch.setattr(config, "index_root", lambda: tmp_path / "index")
    monkeypatch.setattr(config, "store_root", lambda: tmp_path / "store")
    raw = np.full((6, 24, 24), 100, np.uint16)
    labels = np.zeros_like(raw)
    labels[:, 8:16, 8:16] = 1
    tifffile.imwrite(tmp_path / "raw.tif", raw, photometric="minisblack", metadata={"axes": "TYX"})
    tifffile.imwrite(tmp_path / "labels.tif", labels, photometric="minisblack", metadata={"axes": "TYX"})
    movies = [{"stem": "test", "raw": tmp_path / "raw.tif", "labels": tmp_path / "labels.tif"}]
    real = store.manifest.batch_updates
    calls = []
    @contextmanager
    def counted():
        calls.append("enter")
        with real():
            yield
        calls.append("exit")
    monkeypatch.setattr(store.manifest, "batch_updates", counted)
    measure(movies, output_dir=tmp_path / "new", run_label="test", enabled_modules=["intensity"],
            frame_interval_min=30, claim="Synthetic storage parity test")
    assert calls == ["enter", "exit"]
    monkeypatch.delattr(store.manifest, "batch_updates")
    measure(movies, output_dir=tmp_path / "old", run_label="test", enabled_modules=["intensity"],
            frame_interval_min=30, claim="Synthetic storage parity test")
    first = tmp_path / "new/test/measure/test"
    second = tmp_path / "old/test/measure/test"
    for name in ("cell_frame.csv", "cell_summary.csv", "frame_summary.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    indexed = {Path(row["path"]).resolve() for row in store.manifest.load()["artefacts"].values()}
    assert (first / "cell_frame.csv").resolve() in indexed
    assert (second / "cell_frame.csv").resolve() in indexed
