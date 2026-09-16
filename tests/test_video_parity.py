"""The legacy PyMicroglia video address delegates to Auto-Organotypic."""

from pymicroglia import video
from auto_organotypic import video as moved_video


def test_video_adapter_does_not_import_removed_private_modules():
    """Importing the adapter exposes the consolidated public video API."""
    assert video.stack_to_video is moved_video.stack_to_video
    assert video.encode is moved_video.encode


def test_video_adapter_exposes_only_names_owned_by_moved_module():
    assert {"encode", "stack_to_video", "available"} <= set(video.__all__)
    assert not {"annotate", "exports", "luts", "render"} & set(video.__all__)


def test_red_only_translates_to_the_consolidated_renderer(tmp_path, monkeypatch):
    calls = []

    def save(source, **options):
        calls.append((source, options))
        return {"output": str(tmp_path / "red.mp4")}

    monkeypatch.setattr(video, "stack_to_video", save)
    source = tmp_path / "registered.tif"
    result = video.red_only(source, frame_interval_minutes=30.0)

    assert result["output"].endswith("red.mp4")
    assert calls[0][0] == source
    assert calls[0][1]["channels"] == 2
    assert calls[0][1]["lut"] == "red"
    assert calls[0][1]["frame_interval_h"] == 0.5
