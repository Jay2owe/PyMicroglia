import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from reprofig import (
    classify_figure,
    extract_figure,
    extract_record,
    formats as reprofig_formats,
)
from pymicroglia.visualisation import panels


def test_pymicroglia_svg_is_a_package_neutral_complete_master(tmp_path):
    source = tmp_path / "trace.csv"
    source.write_text("time,value\n0,1\n1,2\n", encoding="utf-8")
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 2.0])
    try:
        result = panels.save(
            fig,
            tmp_path / "trace.svg",
            table={"time": [0.0, 1.0], "value": [1.0, 2.0]},
            sources=[source],
            claim="A descriptive trace",
            formats=("svg",),
            bundle=False,
            statistics_status="not_applicable",
        )
    finally:
        plt.close(fig)

    svg = result["figures"][0]
    record = extract_record(svg)
    assert record.producer["package"] == "PyMicroglia"
    assert record.data_tables[0].contents == (tmp_path / "trace_plotted.csv").read_text(
        encoding="utf-8"
    )
    assert record.sources[0].sha256
    assert classify_figure(svg)["provenance_level"] == "complete"
    extracted = extract_figure(svg, tmp_path / "extracted")
    assert any(path.name.endswith("plotted-data.csv") for path in extracted)


def test_pymicroglia_minimal_public_uses_the_safe_table_only_in_its_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 2.0])
    try:
        result = panels.save(
            fig,
            tmp_path / "public-trace.svg",
            table={
                "time": [0.0, 1.0],
                "value": [1.0, 2.0],
                "subject": ["private-1", "private-2"],
            },
            formats=("svg",),
            bundle=False,
            statistics_status="not_applicable",
            figure_profile="minimal_public",
            figure_safe_columns=["time", "value"],
        )
    finally:
        plt.close(fig)

    record = extract_record(result["figures"][0])
    assert record.distribution_profile == "minimal_public"
    assert record.data_tables[0].contents is None
    sidecar = (tmp_path / "public-trace_plotted.csv").read_text(encoding="utf-8")
    assert sidecar == "time,value\n0.000000,1.000000\n1.000000,2.000000\n"
    assert "subject" not in sidecar


def test_public_bundle_does_not_copy_private_sources_or_values(tmp_path):
    source = tmp_path / "private-subject.csv"
    source.write_text("time,value,subject\n0,1,private-1\n", encoding="utf-8")
    fig, ax = plt.subplots()
    ax.plot([0.0], [1.0])
    try:
        result = panels.save(
            fig,
            tmp_path / "public-bundle.svg",
            table={"time": [0.0], "value": [1.0], "subject": ["private-1"]},
            sources=[source],
            formats=("svg",),
            bundle=True,
            statistics_status="not_applicable",
            figure_profile="minimal_public",
            figure_safe_columns=["time", "value"],
        )
    finally:
        plt.close(fig)

    bundle = result["bundle"]
    text_files = [
        path for path in bundle.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
    assert "private-subject.csv" not in combined
    assert "private-1" not in combined
    assert not any(path.name == source.name for path in bundle.rglob("*"))


def test_pymicroglia_writes_every_available_direct_carrier_with_one_identity(tmp_path):
    direct = {"svg", "pdf", "png", "jpeg", "tiff", "webp", "avif", "heif"}
    available = [
        row["format"] for row in reprofig_formats()
        if row["format"] in direct and row["available"]
    ]
    extension = {"jpeg": "jpg", "tiff": "tif", "heif": "heif"}
    suffixes = [extension.get(name, name) for name in available]

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([0.0, 1.0], [1.0, 2.0])
    try:
        result = panels.save(
            fig,
            tmp_path / f"multi.{suffixes[0]}",
            table={"time": [0.0, 1.0], "value": [1.0, 2.0]},
            formats=suffixes[1:],
            dpi=123,
            bundle=False,
            statistics_status="not_applicable",
        )
    finally:
        plt.close(fig)

    assert {path.suffix.lstrip(".") for path in result["figures"]} == set(suffixes)
    identities = {extract_record(path).figure_id for path in result["figures"]}
    assert len(identities) == 1


def test_publication_module_exposes_all_carrier_operations():
    from pymicroglia import publication

    assert callable(publication.embed_file)
    assert callable(publication.publish_artifacts)
    assert callable(publication.extract_artifact)
    assert callable(publication.validate_artifact)
    assert len(publication.formats()) >= 16
