"""Portable saved audit report, complete evidence links and explicit choice targets."""
from pymicroglia._results import report_name, read_document, copy_artifact

import html
import json
from pathlib import Path
import shutil
from urllib.parse import quote

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table


def escaped(value):
    return html.escape(str(value), quote=True)


def link(path, label):
    return f'<a href="{quote(str(path).replace(chr(92), "/"), safe="/#")}">{escaped(label)}</a>'


def readable(value):
    return escaped(str(value).replace("_", " "))


def rate(score, prefix):
    value = score.get(prefix + "_rate")
    count = score.get(prefix + "_numerator", 0)
    denominator = score.get(prefix + "_denominator", 0)
    units = score.get(prefix + "_independent_units", 0)
    if value is None:
        return f"Unavailable; {denominator} eligible cases, {units} independent simulation groups"
    bounds = (score.get(prefix + "_lower"), score.get(prefix + "_upper"))
    interval = f" [{bounds[0]:.1%}, {bounds[1]:.1%}]" if all(v is not None for v in bounds) else " (interval unavailable)"
    return f"{value:.1%}{interval}; {count}/{denominator} cases, {units} independent simulation groups"


def inventory_artifact(output, inventory, step, name):
    """Resolve a logical table name against current CSV or original JSON records."""
    records = inventory.get(step, {}).get("artifacts", {})
    found = records.get(name)
    if found is None and name.endswith(".json"):
        from .._tables import schema
        candidate = records.get(name[:-5] + ".csv")
        if candidate:
            root = Path(output).resolve()
            path = (root / candidate["path"]).resolve()
            if path.is_relative_to(root) and schema(path) is not None:
                found = candidate
    return found


def copy_evidence(saved, output):
    """Copy verified artifacts with their original names inside each step folder."""
    inventory = {}
    for step, result in sorted(saved.items()):
        files = {}
        for ref in result.outcome.artifacts:
            source = result.artifact(ref.name)
            relative = Path("evidence") / step / ref.path
            target = (output / relative).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError("Copied audit artifact escapes its report")
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_artifact(source, target)
            if file_hash(target) != ref.sha256:
                raise IOError("Copied audit evidence changed")
            files[ref.name] = {"path": relative.as_posix(), "sha256": ref.sha256, "scientific_id": ref.scientific_id}
        inventory[step] = {"status": result.outcome.status, "reason": result.outcome.reason,
                           "scientific_id": result.outcome.scientific_id, "artifacts": files}
    return inventory


def render_index(output, inventory, *, title="Method-selection audit"):
    """Assemble linked saved decisions; rendering cannot run scientific operations."""
    def source(step, name):
        item = inventory[step]["artifacts"][name]
        path = (output / item["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or file_hash(path) != item["sha256"]:
            raise ValueError("Report source is missing or changed")
        return path
    def artifact(step, name, label=None):
        item = inventory.get(step, {}).get("artifacts", {}).get(name)
        return link(item["path"], label or name) if item else "Unavailable: " + escaped(inventory.get(step, {}).get("reason", "not produced"))
    design = read_document(source("audit-design", "audit_design"))
    selection = read_document(source("candidate-shortlist", "frozen_selection"))
    confirmation = read_document(source("independent-confirmation", "confirmation_record"))
    if selection["selection_id"] != confirmation["selection_id"]:
        raise ValueError("Report combines different frozen choices")
    decisions = read_table(source("independent-confirmation", "final_decisions")).to_dict("records")
    assessments = read_table(source("candidate-shortlist", "candidate_assessments")).to_dict("records")
    targets, target_files, sections = [], [], []
    for candidate in selection["candidates"]:
        record = {"schema_version": 1, "audit_id": design["scientific_id"], "selection_id": selection["selection_id"],
            "confirmation_id": confirmation["confirmation_id"], "candidate_id": candidate["candidate_id"],
            "candidate_settings_id": content_id({k: v for k, v in candidate.items() if k != "labels"}),
            "candidate": candidate, "action": "Explicit selection reference only; no profile has been exported or applied"}
        record["target_id"] = content_id(record)
        path = Path("choices") / (record["target_id"] + ".json")
        (output / path).parent.mkdir(exist_ok=True)
        path = _write_json(output / path, record).relative_to(output)
        target_files.append(path.as_posix()); targets.append(record)
        key = "candidate-" + candidate["candidate_id"]
        settings = escaped(json.dumps(candidate, indent=2, ensure_ascii=False))
        matched = [r for r in assessments if r["candidate_id"] == candidate["candidate_id"]]
        rows = []
        for row in matched:
            failed = "; ".join(g["requirement"] + ": " + g["state"] for g in row["gates"])
            rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
                escaped(row.get("measurement") or "Whole dataset"), readable(row["state"]),
                escaped(rate(row["score"], "recovery")), escaped(rate(row["score"], "false_alarm")), readable(failed))) + "</tr>")
        sections.append(f'<section id="{escaped(key)}"><h3>{escaped(" / ".join(candidate["labels"]))}</h3>'
            f'<p>Complete recipe <code>{escaped(candidate["candidate_id"])}</code>. {link(path, "Save this explicit choice reference")}</p>'
            '<table><tr><th>Scope</th><th>Development state</th><th>Correct recovery</th><th>False alarms</th><th>Gates</th></tr>'
            + "".join(rows) + f'</table><details><summary>Every resolved setting</summary><pre>{settings}</pre></details></section>')
    decision_html = []
    for decision in decisions:
        alternatives = decision.get("candidate_ids", [])
        choices = ", ".join(link("#candidate-" + cid, cid[:12]) for cid in alternatives) or "No selected recipe"
        checks = []
        for check in decision.get("confirmation_checks", []):
            score = check.get("score", {})
            checks.append(f'<li>{link("#candidate-" + check["candidate_id"], check["candidate_id"][:12])}: '
                + readable(check.get("state", check.get("status", "see saved check")))
                + "; recovery " + escaped(rate(score, "recovery")) + "; false alarms " + escaped(rate(score, "false_alarm")) + "</li>")
        decision_html.append(f'<section id="decision-{escaped(decision["decision_id"])}"><h3>{escaped(decision.get("measurement") or "Whole dataset")}</h3>'
            f'<p><strong>{readable(decision["final_state"])}</strong>. {escaped(decision["reason"])}</p>'
            f'<p>{choices}. Confirmation: <strong>{readable(decision["confirmation_status"])}</strong>.</p>'
            + ("<ul>" + "".join(checks) + "</ul>" if checks else "<p>No independent confirmation score is claimed for this decision.</p>")
            + "<p>" + artifact("independent-confirmation", "confirmation_checks", "All confirmation checks and frozen thresholds")
            + " · " + artifact("candidate-shortlist", "candidate_assessments", "Development support and failed gates") + "</p></section>")
    figures = []
    for step, name, collection in (("performance-figures", "figure_manifest.json", "pages"), ("focused-pages", "focused_manifest.json", "rendered_pages")):
        if name not in inventory.get(step, {}).get("artifacts", {}):
            figures.append("<p>" + escaped(step) + ": " + escaped(inventory.get(step, {}).get("reason", "not produced")) + "</p>")
            continue
        manifest = read_document(source(step, name))
        pages = manifest.get(collection, [])
        if not pages:
            figures.append("<p>" + escaped(step) + ": " + escaped(manifest.get("reason", "No pages requested")) + "</p>")
        for page in pages:
            label = page.get("figure", page.get("item", "Saved page"))
            if "movie" in page:
                label = f"{page['movie']} / cell {page['identity']} / {page['measurement']} — part {page['part']}/{page['parts']}"
            figures.append("<li>" + artifact(step, page["master"], label) + " · "
                + artifact(step, page["data"], "Exact plotted data") + " · " + artifact(step, page["statistics"], "Saved statistics")
                + ("<br>Selected for: " + escaped("; ".join(page["reasons"])) if "reasons" in page else "") + "</li>")
    evidence = []
    for step, record in inventory.items():
        links = ["<li>" + link(item["path"], name) + "</li>" for name, item in record["artifacts"].items()]
        evidence.append(f'<details><summary>{escaped(step)} — {escaped(record["status"])}</summary>'
                        + f'<p>{escaped(record["reason"])}</p><ul>' + "".join(links) + "</ul></details>")
    request = design["request"]
    input_policy = request.get("request", {}).get("population", {})
    version = request["source"]["workbench_version"]
    policy = escaped(json.dumps(input_policy, indent=2, ensure_ascii=False))
    compatibility = escaped(json.dumps(confirmation.get("family_compatibility", {}), indent=2, ensure_ascii=False))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped(title)}</title><style>body{{font:16px Arial,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:black}}a{{color:steelblue}}h1,h2,h3{{line-height:1.3}}section{{border-top:1px solid silver;padding:12px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid silver;padding:8px;text-align:left;vertical-align:top}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:whitesmoke;padding:12px}}code{{overflow-wrap:anywhere}}li{{margin:8px 0}}details{{margin:12px 0}}nav{{display:flex;gap:20px;flex-wrap:wrap}}</style>
<h1>{escaped(title)}</h1><nav><a href="#decisions">Decisions</a><a href="#figures">Figures</a><a href="#candidates">Complete recipes</a><a href="#evidence">All evidence</a></nav>
<p>Workbench {escaped(version)}. {design.get('profile_count', 0)} frozen recording profiles; {len(selection['candidates'])} complete candidate recipes.</p>
<p>Recovery measures agreement with the declared synthetic cases. Fresh-case confirmation checks that benchmark; it does not independently validate the biology of these cells. Real recordings informed the audit, so their downstream analysis remains exploratory. No daily period or common waveform is assumed.</p>
<p>Intervals apply separately to fixed, weighted simulation groups. Profiles and repeated omissions are not independent biological replicates. Missing or insufficient evidence is retained.</p>
<details><summary>Input population rule</summary><pre>{policy}</pre></details>
<details><summary>Correction-family compatibility</summary><pre>{compatibility}</pre></details>
<h2 id="decisions">Saved decisions</h2>{''.join(decision_html)}
<h2 id="figures">Saved figures and inspection pages</h2><ul>{''.join(figures)}</ul>
<h2 id="candidates">Every complete candidate</h2><p>Choice references identify exact settings for a later explicit export. They do not change the active configuration or make an unconfirmed manual choice validated.</p>{''.join(sections)}
<h2 id="evidence">Complete saved evidence</h2>{''.join(evidence)}
</html>'''
    (output / "index.html").write_text(document, encoding="utf-8")
    manifest = {"schema_version": 1, "audit_id": design["scientific_id"], "selection_id": selection["selection_id"],
        "confirmation_id": confirmation["confirmation_id"], "index": "index.html", "inputs": inventory,
        "candidate_targets": [{"path": path, "target_id": target["target_id"], "candidate_id": target["candidate_id"]}
                              for path, target in zip(target_files, targets)],
        "analysis_recomputed": False, "settings_applied": False, "settings_exported": False}
    manifest["report_id"] = content_id(manifest)
    _write_json(output / "audit_report.json", manifest)
    return manifest


def produce_index(context):
    appearance = context.presentation.as_dict().get("report", {})
    if not isinstance(appearance, dict) or set(appearance) - {"title"} or not isinstance(appearance.get("title", ""), str):
        raise ValueError("Report presentation accepts a title string")
    context.output.mkdir(parents=True)
    inventory = copy_evidence(context.dependencies, context.output)
    manifest = render_index(context.output, inventory, **appearance)
    refs = tuple(ArtifactRef(report_name(path,context.output), path.relative_to(context.output).as_posix(),
                            file_hash(path), context.scientific_id)
                 for path in sorted(context.output.rglob("*")) if path.is_file() and path.name!="artefacts.json")
    return StepResult(context.step.name, context.scientific_id, "completed", "Saved portable audit report and complete choice references",
        refs, provenance=Settings({"report_id": manifest["report_id"], "analysis_recomputed": False, "settings_applied": False}))
