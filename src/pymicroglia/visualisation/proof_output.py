"""Optional rigorous ReproFig policy helpers for figure actions."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


def prepare_proof(
    figure: Any,
    record: Any,
    table: Any,
    *,
    claim: str,
    statistical_specs: Sequence[Mapping[str, Any]],
    proof: bool,
    proof_policy: Mapping[str, Any] | None,
    required_grades: Sequence[str],
    signing_key_path: str | None,
    signing_password_env: str | None,
    trust_policy_path: str | None,
    encrypted_sections: Sequence[str],
    encryption_password_env: str | None,
    recipient_file: str | None,
    broker_policy_path: str | None,
) -> tuple[bool, dict[str, Any]]:
    """Attach typed claims and semantic bindings only when explicitly enabled."""

    from reprofig import ScientificClaim, StatisticalSpecification, bind_artist

    policy = dict(proof_policy or {})
    policy.update({
        key: value
        for key, value in {
            "required_grades": list(required_grades),
            "signing_key_path": signing_key_path,
            "signing_password_env": signing_password_env,
            "trust_policy_path": trust_policy_path,
            "encrypt_sections": list(encrypted_sections),
            "encryption_password_env": encryption_password_env,
            "recipient_file": recipient_file,
            "broker_policy_path": broker_policy_path,
        }.items()
        if value not in (None, [], ())
    })
    enabled = bool(proof or policy or statistical_specs)
    if not enabled:
        return False, policy

    specifications = [
        value
        if isinstance(value, StatisticalSpecification)
        else StatisticalSpecification.from_dict(value)
        for value in statistical_specs
    ]
    claims = (
        [ScientificClaim(
            text=claim,
            statistic_ids=[str(value.statistic_id) for value in specifications],
        ).to_dict()]
        if claim
        else []
    )
    record.extensions["proof"] = {
        "statistical_specifications": [value.to_dict() for value in specifications],
        "claims": claims,
    }
    table_id = f"table:{table.sha256}"
    columns = [column.name for column in table.columns]
    try:
        from reprofig.transformations import canonical_rows

        row_ids = [str(row["__reprofig_row_id"]) for row in canonical_rows(table)]
    except Exception:
        row_ids = []
    for axes_index, axes in enumerate(figure.axes):
        for kind, artists in (
            ("line", axes.lines),
            ("collection", axes.collections),
            ("patch", axes.patches),
            ("image", axes.images),
        ):
            for artist_index, artist in enumerate(artists):
                if getattr(artist, "_reprofig_binding", None):
                    continue
                bind_artist(
                    artist,
                    semantic_id=f"pymicroglia-axes-{axes_index}-{kind}-{artist_index}",
                    table_id=table_id,
                    row_ids=row_ids,
                    columns=columns,
                    role="plotted_evidence",
                )
    return True, policy


def proof_summary(outputs: Sequence[Path], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return honest per-carrier meanings and the shared scientific root."""

    from reprofig import extract_record, graph_from_record, verify_proof

    required = policy.get("required_meanings", policy.get("required_grades", []))
    trust_store = policy.get("trust_store", policy.get("trust_policy_path"))
    reports = {}
    roots = {}
    for output in outputs:
        verification = verify_proof(
            output, required=required, trust_store=trust_store
        )
        reports[str(output)] = verification.to_dict()
        roots[str(output)] = graph_from_record(extract_record(output)).root_sha256
    return {
        "reports": reports,
        "evidence_roots": roots,
        "shared_evidence_root": (
            next(iter(set(roots.values())))
            if roots and len(set(roots.values())) == 1
            else None
        ),
    }


def promote_outputs(
    outputs: Sequence[Path], *, policy_path: str, workspace_parent: Path, stem: str
) -> dict[str, Any]:
    """Verify candidate artifacts before controlled promotion."""

    from reprofig.guard.broker import OutputBroker
    from reprofig.guard.policy import OutputPolicy
    from reprofig.guard.workspace import GuardWorkspace

    policy = OutputPolicy.from_json(policy_path)
    if not policy.destination:
        raise ValueError("broker policy requires a controlled destination")
    workspace = GuardWorkspace.create(workspace_parent / ".reprofig-broker" / stem)
    broker = OutputBroker(workspace, policy.destination, policy, mode="hard")
    receipts = []
    promoted = []
    for output in outputs:
        candidate = workspace.candidates / output.name
        shutil.copyfile(output, candidate, follow_symlinks=False)
        receipt = broker.promote(candidate)
        receipts.append(receipt.to_dict())
        promoted.append(str(Path(policy.destination) / receipt.output_name))
    return {"outputs": promoted, "receipts": receipts}


__all__ = ["prepare_proof", "proof_summary", "promote_outputs"]
