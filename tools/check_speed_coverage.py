"""Expose missing execution coverage without turning descriptor timings into it.

Freezes the original action inventory once. Run the shared performance skill's
check-action-coverage.py on the generated scope and evidence files for its exit
status. Execution receipts must be reviewed and added explicitly; this helper
never infers coverage merely from a test or action name.
"""
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-receipt", help="Reviewed comparison receipt, relative to the speed-pass folder")
    args = parser.parse_args()
    from pymicroglia.registry import REGISTRY
    root = Path(__file__).resolve().parents[1] / "development/speed-pass-20261011"
    original = json.loads((root / "public-actions.json").read_text(encoding="utf-8"))[0]["actions"]
    if set(REGISTRY.names()) != {row["action"] for row in original}:
        raise ValueError("Live actions differ from the frozen scope; reconcile additions before continuing")
    scope = {"required": ["action:" + row["action"] for row in original] + [
        "workflow:initial-usable-screen", "workflow:selected-item-full-detail",
        "workflow:growth", "workflow:restart", "workflow:external-edit"]}
    path = root / "scope.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != scope:
        raise ValueError("Frozen scope changed; review additions without silently reducing it")
    path.write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
    evidence = root / "coverage-evidence.json"
    if not evidence.exists():
        evidence.write_text(json.dumps({"operations": [{"id": "action:" + row["action"],
            "kind": "description", "status": "measured", "receipt": "public-actions.json"}
            for row in original]}, indent=2) + "\n", encoding="utf-8")
    if args.measurement_receipt:
        receipt_path = (root / args.measurement_receipt).resolve()
        if not receipt_path.is_relative_to(root.resolve()):
            raise ValueError("Receipt must belong to this speed pass")
        measured = json.loads(receipt_path.read_text(encoding="utf-8"))
        if measured.get("id") != "action:measure" or measured.get("parity") is not True:
            raise ValueError("Expected a reviewed measurement comparison with exact output parity")
        measured["receipt"] = args.measurement_receipt
        data = json.loads(evidence.read_text(encoding="utf-8"))
        data["operations"] = [measured if row["id"] == measured["id"] else row for row in data["operations"]]
        evidence.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"{len(original)} actions and five additional workflow requirements; scope preserved.")


if __name__ == "__main__":
    main()
