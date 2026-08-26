"""Shared action-facing figure save wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def default_output_dir(source, suffix: str = "_qc") -> Path:
    """Beside the results, in the folder every protocol in this project uses."""
    path = Path(source).resolve()
    for parent in (path.parent, *path.parents):
        if parent.name.lower() == "ai_exports":
            return parent / f"{path.stem}{suffix}"
    return path.parent / "AI_Exports" / f"{path.stem}{suffix}"


def save_for(figure, source, table: Mapping[str, Sequence[Any]], *, stage: str,
             output_name: str, output_dir=None, overwrite: bool = False,
             dpi: int = 150, artefacts: Iterable[Any] = (), claim: str = "",
             settings: Mapping[str, Any] | None = None,
             output_formats: Sequence[str] = ("png",),
             figure_profile: str = "master",
             figure_safe_columns: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
             public_sources: Mapping[str, str] | None = None,
             dpi_preset: str | None = None, render_preset: str | None = None,
             render_width_in: float | None = None,
             render_height_in: float | None = None,
             format_options: Mapping[str, Any] | None = None,
             allow_reencode: bool = False, proof: bool = False,
             statistical_specs: Sequence[Mapping[str, Any]] = (),
             required_grades: Sequence[str] = (),
             signing_key_path: str | None = None,
             signing_password_env: str | None = None,
             trust_policy_path: str | None = None,
             encrypted_sections: Sequence[str] = (),
             encryption_password_env: str | None = None,
             recipient_file: str | None = None,
             broker_policy_path: str | None = None,
             suffix: str = "_qc") -> dict[str, Any]:
    """Save and close one figure through the package's single audited path."""
    from matplotlib import pyplot as plt
    from .panels import save

    folder = Path(output_dir) if output_dir else default_output_dir(source, suffix)
    written = save(
        figure, folder / f"{output_name}.png", table=table,
        sources=[{"path": source, "role": "pixels"}], claim=claim,
        artefacts=[item for item in artefacts if item is not None],
        settings=settings or {}, formats=output_formats, dpi=dpi,
        figure_profile=figure_profile, figure_safe_columns=figure_safe_columns,
        public_sources=public_sources, dpi_preset=dpi_preset,
        render_preset=render_preset, width=render_width_in,
        height=render_height_in, format_options=format_options,
        allow_reencode=allow_reencode, overwrite=overwrite, slug=output_name,
        proof=proof, statistical_specs=statistical_specs,
        required_grades=required_grades, signing_key_path=signing_key_path,
        signing_password_env=signing_password_env,
        trust_policy_path=trust_policy_path,
        encrypted_sections=encrypted_sections,
        encryption_password_env=encryption_password_env,
        recipient_file=recipient_file, broker_policy_path=broker_policy_path)
    plt.close(figure)
    return {
        "stage": stage,
        "figures": [str(path) for path in written["figures"]],
        "table": str(written["table"]),
        "provenance": str(written["provenance"]),
        "bundle": str(written["bundle"]) if written["bundle"] else None,
        "proof": written.get("proof"),
    }


__all__ = ["default_output_dir", "save_for"]
