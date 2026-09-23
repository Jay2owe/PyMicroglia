"""Portable navigation from verified cell, pair and figure inventories."""
from pymicroglia._results import workings_link
from pymicroglia._results import report_name, read_document, copy_artifact
from .._saved_figures import table_name
from pymicroglia._sources import source_file
import html
import json
import os
import shutil
from uuid import uuid4
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from pymicroglia.pipelines.audit.index import copy_evidence
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, cell_number, content_id, result_to_dict
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table

KEYS = ("source_run", "movie", "identity")
MANIFESTS = {
    "screening-overview": "overview_manifest.json",
    "selected-cell-evidence": "evidence_manifest.json",
    "time-matrices": "time_matrix_manifest.json",
    "group-comparison-figures": "group_comparison_manifest.json",
    "detection-agreement-figures": "agreement_manifest.json",
    "timing-relationship-figures": "relationship_manifest.json",
}
LABELS = {
    "rhythm-screen": "Complete rhythm screening",
    "screening-overview": "Screening overview",
    "selected-cell-evidence": "Selected cell reports and trace grids",
    "time-matrix-values": "Saved time-matrix values",
    "time-matrices": "Time matrices",
    "group-comparisons": "Rhythm-group comparison results",
    "group-comparison-figures": "Rhythm-group comparison figures",
    "detection-agreement": "Detection-agreement results",
    "detection-agreement-figures": "Detection-agreement figures",
    "within-cell-timing": "Timing within each cell",
    "timing-across-samples": "Timing across cells and samples",
    "timing-relationship-figures": "Timing matrices and pair reports",
}


def version():
    return content_id({name: file_hash(source_file(name)) for name in ("rhythm_index.py", "audit_index.py")})


def openable_report(saved, *, directory=None):
    """Give Windows browsers a verified short copy of the immutable report."""
    index = saved.artifact("index.html")
    if os.name != "nt": return index
    # Store Python may virtualize LOCALAPPDATA into its package directory. A
    # short profile path is also visible to an ordinary desktop browser.
    base = Path(directory) if directory is not None else Path.home() / ".motion" / "reports"
    base.mkdir(parents=True, exist_ok=True)
    destination = base / content_id({"copy_version": 1, "result": str(saved.root.resolve()), "index": file_hash(index)})[:20]
    sources = [(ref, saved.artifact(ref.name)) for ref in saved.outcome.artifacts]
    if destination.exists():
        if all((destination / ref.path).is_file() and file_hash(destination / ref.path) == ref.sha256 for ref, _ in sources):
            return (destination / "index.html").resolve()
        # Preserve a user's edited copy, and leave interrupted copies untouched.
        destination = destination.with_name(destination.name + "-" + uuid4().hex[:8])
    temporary = base / ("copy-" + uuid4().hex[:12])
    temporary.mkdir()
    for ref, source in sources:
        target = (temporary / ref.path).resolve()
        if not target.is_relative_to(temporary.resolve()): raise ValueError("Report copy escapes its destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_artifact(source, target)
        if file_hash(target) != ref.sha256: raise ValueError("Browser report copy differs from verified evidence")
    temporary.rename(destination)
    return (destination / "index.html").resolve()


def escaped(value):
    return html.escape(str(value), quote=True)


def link(path, label):
    destination = quote(path, safe="/") if not path.startswith("#") else "#" + quote(path[1:], safe="")
    return f'<a href="{destination}">{escaped(label)}</a>'


def cell_id(row):
    return "cell-" + content_id({key: row[key] for key in KEYS})


def cell_label(row):
    return f"{row['movie']} / cell {row['identity']}"


def build(output, inventory, *, title="Rhythm discovery", requested=None):
    """Build only from immutable artifacts; indexes never infer new results."""
    output = Path(output)
    def source(step, name):
        ref = inventory.get(step, {}).get("artifacts", {}).get(name)
        if ref is None: return None
        path = (output / ref["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or file_hash(path) != ref["sha256"]:
            raise ValueError(f"Index source is missing or changed: {step}/{name}")
        return path
    def table(step, name):
        path = source(step, name)
        return read_table(path) if path else pd.DataFrame()
    def metadata(step, name):
        path = source(step, name)
        return read_document(path) if path else {}
    def path_for(step, name):
        if source(step, name) is None: raise ValueError(f"Figure manifest references an absent artifact: {step}/{name}")
        return inventory[step]["artifacts"][name]["path"]

    provenance = metadata("rhythm-screen", "provenance")
    screen_id = inventory.get("rhythm-screen", {}).get("scientific_id")
    for step in inventory:
        source_provenance = metadata(step, "provenance")
        if "screen_id" in source_provenance and source_provenance["screen_id"] != screen_id:
            raise ValueError("Index inputs refer to different saved rhythm screens")
    resolved = requested if requested is not None else provenance.get("resolved_request", {})
    population = resolved.get("inputs", {}).get("cells", [])
    results = table("rhythm-screen", "rhythm_results")
    cells = {cell_id(row): {**row, "id": cell_id(row), "pages": [], "report_pages": [], "results": []} for row in population}
    for row in _json_value(results.to_dict("records")):
        key = cell_id(row)
        if key not in cells: raise ValueError("Screen result is outside the requested cell population")
        cells[key]["results"].append({name: row.get(name) for name in ("measurement", "status", "reason", "test_status",
            "significant", "p_value", "q_value", "period_hours", "period_available", "period_underdetermined", "period_at_search_edge",
            "method", "significance_method", "family_id", "observations", "span_hours", "sample", "sample_confirmed")})
    pairs = {}
    for step, name in (("detection-agreement", "summary"), ("within-cell-timing", "pairs"), ("timing-across-samples", "summary")):
        for row in _json_value(table(step, name).to_dict("records")):
            key = row["pair_id"]
            if key in pairs and (pairs[key]["reference"], pairs[key]["target"]) != (row["reference"], row["target"]):
                raise ValueError("The same pair identity has conflicting direction")
            pair = pairs.setdefault(key, {"pair_id": key, "id": "pair-" + key, "reference": row["reference"],
                "target": row["target"], "pages": [], "report_pages": [], "evidence": {}})
            # Full native arrays remain linked in their scientific table; these
            # are existing typed summary/outcome rows, without new calculations.
            pair["evidence"].setdefault(step, []).append(row)
    # A failed screen still retains every explicitly requested pair destination.
    for row in resolved.get("request", {}).get("pairs", []):
        if not any((p["reference"], p["target"]) == (row["reference"], row["target"]) for p in pairs.values()):
            key = content_id({"requested_pair": row})
            pairs[key] = {"pair_id": key, "id": "pair-" + key, **row, "pages": [], "report_pages": [], "evidence": {}}

    pages, branches = [], []
    for step, name in MANIFESTS.items():
        manifest = metadata(step, name)
        if "screen_id" in manifest and manifest["screen_id"] != screen_id:
            raise ValueError("Figure manifest belongs to a different saved rhythm screen")
        branch = inventory.get(step, {"status": "not-requested", "reason": "This branch was not requested"})
        branches.append({"step": step, "status": branch["status"], "reason": branch["reason"],
            "empty_reason": manifest.get("empty_reason", ""), "page_count": len(manifest.get("pages", []))})
        for item in manifest.get("pages", []):
            master = item["master"]
            figure_path = path_for(step, master)
            key = "figure-" + content_id({"step": step, "master": master, "scientific_id": inventory[step]["scientific_id"]})
            data_name = table_name(master,'values',lambda filename:source(step,filename))
            data_path = source(step, data_name)
            plotted = pd.read_csv(data_path, dtype={"source_run": str, "movie": str}) if data_path else pd.DataFrame()
            member_ids = []
            if set(KEYS) <= set(plotted):
                for row in plotted[list(KEYS)].dropna().drop_duplicates().to_dict("records"):
                    # CSVs with mixed cell/sample rows read integral cell keys as
                    # floats. Restore the declared cell-key type before hashing.
                    row["identity"] = cell_number(row["identity"])
                    member = cell_id(row)
                    if member not in cells: raise ValueError("Figure contains an unknown source/movie/cell key")
                    member_ids.append(member)
            # Manifest membership also covers cells with no observed values.
            declared = item.get("members", item.get("cells", []))
            if isinstance(declared, list):
                for row in declared:
                    row = row if isinstance(row, dict) else {"source_run": item.get("source_run"), "movie": item.get("movie"), "identity": row}
                    if set(KEYS) <= row.keys() and all(row[k] is not None for k in KEYS):
                        member = cell_id(row)
                        if member not in cells: raise ValueError("Page manifest contains an unknown cell")
                        member_ids.append(member)
            if item.get("kind") == "reports": member_ids.append(cell_id(item))
            pair_ids = [item["pair_id"]] if "pair_id" in item else item.get("pair_ids", [])
            if "pair_id" in plotted:
                pair_ids = [*pair_ids, *plotted.pair_id.dropna().unique().tolist()]
            pair_ids = list(dict.fromkeys(value for value in pair_ids if value))
            if any(value not in pairs for value in pair_ids): raise ValueError("Figure contains an unknown ordered pair")
            member_ids = list(dict.fromkeys(member_ids))
            title_parts = [LABELS[step], str(item.get("kind", item.get("figure", "page"))).replace("-", " ")]
            for field in ("measurement", "comparison", "level", "stratum"):
                if field in item: title_parts.append(str(item[field]))
            if "part" in item: title_parts.append(f"page {item['part']}/{item.get('parts', 1)}")
            page = {"id": key, "step": step, "path": figure_path, "title": " | ".join(title_parts),
                "cells": member_ids, "pairs": pair_ids, "semantics": item, "data": []}
            for name, label in ((data_name, "Plotted values"), (table_name(master,'statistics',lambda filename:source(step,filename)), "Saved statistics"),
                                (table_name(master,'display',lambda filename:source(step,filename)), "Display settings")):
                if source(step, name): page["data"].append({"path": path_for(step, name), "label": label})
            pages.append(page)
            for member in member_ids:
                cells[member]["pages"].append(key)
                if item.get("kind") == "reports": cells[member]["report_pages"].append(key)
            for pair_id in pair_ids:
                pairs[pair_id]["pages"].append(key)
                if item.get("kind") in {"pair", "pair-summary", "pair-traces"}: pairs[pair_id]["report_pages"].append(key)

    navigation = {"schema_version": 1, "screen_id": inventory.get("rhythm-screen", {}).get("scientific_id"),
        "index": "index.html", "analysis_recomputed": False, "cells": list(cells.values()), "pairs": list(pairs.values()),
        "pages": pages, "branches": branches, "inputs": inventory, "resolved_settings": resolved,
        "screen_resolved_settings": provenance.get("resolved_request", {})}
    _write_json(output / "navigation.json", navigation)
    document = _document(navigation, title)
    (output / "index.html").write_text(document, encoding="utf-8")
    return navigation


def _document(data, title):
    by_page = {page["id"]: page for page in data["pages"]}
    by_cell = {cell["id"]: cell for cell in data["cells"]}
    by_pair = {pair["pair_id"]: pair for pair in data["pairs"]}
    def page_links(keys):
        return "<ul>" + "".join("<li>" + link("#" + key, by_page[key]["title"]) + "</li>" for key in dict.fromkeys(keys)) + "</ul>"
    def details(label, value):
        return f'<details><summary>{escaped(label)}</summary><pre>{escaped(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details>'
    def pair_label(pair): return pair["reference"] + " to " + pair["target"]
    def unavailable(step, default):
        branch = data["inputs"].get(step)
        if branch is None: return "Not requested: this branch was not included in this execution."
        return escaped(branch["status"] + ": " + branch["reason"] if branch and branch["status"] not in {"completed", "reused"} else default)
    metrics = [m["column"] for m in data["resolved_settings"].get("test_measurements", [])]
    overview = []
    for cell in data["cells"]:
        outcomes = {row["measurement"]: row for row in cell["results"]}
        entries = ["<td>" + link("#" + cell["id"], cell_label(cell)) + "</td>"]
        for metric in metrics:
            row = outcomes.get(metric, {})
            entries.append("<td>" + link("#" + cell["id"], str(row.get("status", "No saved result"))) + "</td>")
        overview.append("<tr>" + "".join(entries) + "</tr>")
    cell_sections = []
    for cell in data["cells"]:
        cards = page_links(cell["report_pages"]) if cell["report_pages"] else "<p>" + unavailable("selected-cell-evidence", "No report card was selected by the saved rhythm decisions.") + "</p>"
        cell_sections.append(f'<section id="{cell["id"]}"><h3>{escaped(cell_label(cell))}</h3>'
            f'<p>Source run: {escaped(cell["source_run"])}</p>{cards}'
            + details("Every saved measurement outcome, method and correction family", cell["results"])
            + "<details><summary>All figures containing this cell</summary>" + page_links(cell["pages"]) + "</details></section>")
    pair_sections = []
    for pair in data["pairs"]:
        cards = page_links(pair["report_pages"]) if pair["report_pages"] else "<p>" + unavailable("timing-relationship-figures", "No pair figures were produced; inspect the branch states below.") + "</p>"
        pair_sections.append(f'<section id="{pair["id"]}"><h3>{escaped(pair_label(pair))}</h3><p>Reference to target; positive timing offset means target follows reference.</p>'
            + cards + details("Detection, cell timing and sample results remain separate", pair["evidence"])
            + "<details><summary>All figures containing this pair</summary>" + page_links(pair["pages"]) + "</details></section>")
    figures = []
    for page in data["pages"]:
        members = ", ".join(link("#" + key, cell_label(by_cell[key])) for key in page["cells"])
        pairs = ", ".join(link("#" + by_pair[key]["id"], pair_label(by_pair[key])) for key in page["pairs"])
        figure = link(page["path"], "Open saved figure")
        records = " | ".join(link(item["path"], item["label"]) for item in page["data"])
        figures.append(f'<details id="{page["id"]}"><summary>{escaped(page["title"])}</summary><p>{figure} | {records}</p>'
            f'<p>Cells: {members or "No individual cell rows on this page"}</p><p>Pairs: {pairs or "No ordered pair rows on this page"}</p>'
            + f'<img loading="lazy" src="{quote(page["path"], safe="/")}" alt="{escaped(page["title"])}">'
            + details("Exact page membership and settings", page["semantics"]) + "</details>")
    branches = []
    for step, row in data["inputs"].items():
        files = ["<li>" + link(ref["path"], name) + "</li>" for name, ref in row["artifacts"].items()]
        branches.append(f'<details><summary>{escaped(LABELS.get(step, step))}: {escaped(row["status"])}</summary><p>{escaped(row["reason"])}</p>'
            f'<p>Scientific result: <code>{escaped(row["scientific_id"])}</code>. '
            + link(row["record"], "Execution outcome, provenance and exact selections") + "</p><ul>" + "".join(files) + "</ul></details>")
    empty = "".join(f'<p>{escaped(LABELS[row["step"]])}: {escaped(row["status"])}. {escaped(row["empty_reason"] or row["reason"])}</p>'
                    for row in data["branches"] if not row["page_count"])
    heading = "".join("<th>" + escaped(value) + "</th>" for value in ["Movie / cell", *metrics])
    pair_nav = " | ".join(link("#" + pair["id"], pair_label(pair)) for pair in data["pairs"])
    failure = [row for row in data["inputs"].values() if row["status"] in {"failed", "unavailable"}]
    banner = '<p class="status">Some requested branches are unfinished. Their exact states and reasons are retained below.</p>' if failure else ""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped(title)}</title><style>body{{font:16px Arial,sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;color:black}}a{{color:steelblue}}nav{{display:flex;gap:22px;flex-wrap:wrap}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid silver;text-align:left}}details{{margin:14px 0;padding:9px;border:1px solid gainsboro}}summary{{cursor:pointer;font-weight:bold}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:whitesmoke;padding:10px}}code{{overflow-wrap:anywhere}}img{{width:100%;height:auto}}section{{border-top:1px solid silver;padding:12px 0}}li{{margin:8px 0}}.status{{background:cornsilk;padding:14px}}</style></head><body>
<h1>{escaped(title)}</h1>{banner}<nav><a href="#overview">Cell overview</a><a href="#pairs">Measurement pairs</a><a href="#figures">Figures</a><a href="#settings">Settings</a><a href="#evidence">Execution and evidence</a></nav>
<p>Results retain the complete requested population. A significant test, a supported period and consistent timing are separate outcomes. Not significant does not establish biological absence. Timing comparisons require the recorded period support; no daily or shared tissue clock is assumed.</p>
<h2 id="overview">Linked cell overview</h2><p>Select any cell or outcome to open its report links and complete saved results.</p><table><tr>{heading}</tr>{''.join(overview)}</table>
<h2>Cell evidence</h2>{''.join(cell_sections)}<h2 id="pairs">Measurement pairs</h2><p>{pair_nav or "No pairs requested or resolved"}</p>{''.join(pair_sections)}
<h2 id="figures">Saved figures</h2><p>Every figure has links to its exact cells and ordered pairs. The copied figure files and their frozen data are unchanged.</p>{empty}{''.join(figures)}
<h2 id="settings">Resolved scientific settings</h2>{details("Current request: measurements, sample mapping, methods and all applied choices", data['resolved_settings'])}{details("Original saved screen settings and full correction population", data['screen_resolved_settings'])}
<h2 id="evidence">Execution and complete evidence</h2>{''.join(branches)}<p>{link(workings_link('navigation.json'), 'Complete portable navigation inventory')}</p>
<script>function reveal(){{var e=document.getElementById(decodeURIComponent(location.hash.slice(1)));for(var p=e;p;p=p.parentElement){{if(p.tagName==='DETAILS')p.open=true}}if(e)e.scrollIntoView()}}addEventListener('hashchange',reveal);addEventListener('DOMContentLoaded',reveal)</script></body></html>'''


def produce(context):
    appearance = context.presentation.as_dict().get("report", {})
    if not isinstance(appearance, dict) or set(appearance) - {"title"} or not isinstance(appearance.get("title", ""), str):
        raise ValueError("Report presentation accepts a title string")
    context.output.mkdir(parents=True, exist_ok=True)
    inventory = copy_evidence(context.dependencies, context.output)
    records = context.output / "execution-records"
    records.mkdir()
    for step, saved in context.dependencies.items():
        path = records / (step + ".json")
        path = _write_json(path, result_to_dict(saved.outcome))
        inventory[step]["record"] = path.relative_to(context.output).as_posix()
    requested = context.request.as_dict() if context.request is not None else None
    navigation = build(context.output, inventory, requested=requested, **appearance)
    refs = tuple(ArtifactRef(report_name(path,context.output), path.relative_to(context.output).as_posix(),
        file_hash(path), context.scientific_id) for path in sorted(context.output.rglob("*")) if path.is_file() and path.name!="artefacts.json")
    return StepResult(context.step.name, context.scientific_id, "completed", "Saved portable linked cell, pair and figure evidence", refs,
        provenance=Settings({"screen_id": navigation["screen_id"], "analysis_recomputed": False}))
