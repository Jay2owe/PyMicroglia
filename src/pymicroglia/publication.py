"""Format-neutral ReproFig publication tools for PyMicroglia outputs.

Matplotlib figures are rendered by :func:`pymicroglia.visualisation.panels.save`.
These exports cover the second half of the workflow: embedding, inspecting,
validating, extracting, scanning and publishing direct figures plus existing
documents, web pages, archives and scientific containers.
"""

from reprofig import (
    ArtifactPublicationResult,
    bundle_artifacts,
    build_publication_workbook,
    embed_file,
    extract_artifact,
    extract_record,
    formats,
    inspect_artifact,
    publish_artifacts,
    save_figure,
    scan_artifacts,
    validate_artifact,
    verify_proof,
)


def publication_workbook(
    source=None,
    *,
    artifacts=(),
    output_path,
    statistics_ledger_path=None,
    profile="master",
    safe_columns=None,
    public_sources=None,
    declare_ledger_complete=False,
    overwrite=False,
):
    """Build one canonical journal workbook from an explicit figure batch."""

    inputs = ([source] if source is not None else []) + list(artifacts or [])
    if not inputs:
        raise ValueError("publication_workbook requires source or artifacts")
    result = build_publication_workbook(
        inputs,
        output_path,
        profile=profile,
        experiment_statistics=statistics_ledger_path,
        declare_ledger_complete=bool(declare_ledger_complete),
        safe_columns=safe_columns,
        public_sources=public_sources,
        overwrite=bool(overwrite),
    )
    return result.to_dict()

__all__ = [
    "ArtifactPublicationResult",
    "bundle_artifacts",
    "build_publication_workbook",
    "embed_file",
    "extract_artifact",
    "extract_record",
    "formats",
    "inspect_artifact",
    "publish_artifacts",
    "publication_workbook",
    "save_figure",
    "scan_artifacts",
    "validate_artifact",
    "verify_proof",
]
