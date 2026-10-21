"""Exercise cell filters against the real UI without running or editing analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

from playwright.sync_api import sync_playwright
from verify_exploration_rhythms_ui import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:62855")
    args = parser.parse_args()
    real = args.run.resolve()
    files = [real / "measure/95_A3/cell_frame.csv", *real.glob("AI_Exports/exploration_rhythms/**/statistics.csv")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    output = Path(tempfile.mkdtemp(prefix="pymicroglia-cell-filters-"))
    errors, jobs = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("request", lambda r: jobs.append(r.url) if r.method == "POST" and "jobs" in r.url else None)
            page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
            started = perf_counter();load(page, args.url, real)
            page.wait_for_function("window.pyMicrogliaExplorer.exact && window.pyMicrogliaExplorer.history && window.pyMicrogliaExplorer.image")
            first_usable = perf_counter()-started
            original = page.evaluate("window.pyMicrogliaExplorer.data.identities")
            assert page.locator("[data-plot-type]").count() == 9
            assert page.locator("#xp-filter-panel").is_hidden()
            page.locator("#xp-filter-toggle").click()
            page.wait_for_function("window.pyMicrogliaExplorer.filters.schema")
            page.locator('[data-filter-recipe="nonrhythmic"]').click()
            page.locator("#xp-filter-preview").click()
            page.wait_for_function("window.pyMicrogliaExplorer.filters.preview")
            assert page.evaluate("window.pyMicrogliaExplorer.data.identities") == original
            page.locator("#xp-filter-apply").click()
            page.wait_for_function("window.pyMicrogliaExplorer.filters.applied")
            expected = page.evaluate("window.pyMicrogliaExplorer.filters.applied.identities")
            assert page.evaluate("window.pyMicrogliaExplorer.data.identities") == expected
            assert len(expected) < len(original) and expected
            page.wait_for_function("window.pyMicrogliaExplorer.timeline && window.pyMicrogliaExplorer.gridRows.size")
            assert page.evaluate("window.pyMicrogliaExplorer.timeline.cells.map(c=>c.identity).sort((a,b)=>a-b)") == expected
            assert page.locator(".xp-grid-cell").count() == len(expected)
            assert page.evaluate("window.pyMicrogliaExplorer.hits['phase-clock'].every(p=>window.pyMicrogliaExplorer.inCohort(p.identity))")
            assert page.evaluate("window.pyMicrogliaExplorer.sourceData.identities") == original
            page.screenshot(path=str(output / "filter-preview.png"))
            page.get_by_role("button", name="Events", exact=True).click()
            page.wait_for_function("window.pyMicrogliaEvents?.data")
            assert page.evaluate("window.pyMicrogliaEvents.filtered().every(r=>window.pyMicrogliaExplorer.inCohort(r.identity))")
            page.get_by_role("button", name="Follow a cell", exact=True).click()
            # Draft changes do not change the applied cohort. Nested NOT is saved intact.
            page.locator("#xp-filter-clear").click()
            page.locator('#xp-filter-tree [data-edit="group"][data-path=""]').click()
            page.locator('#xp-filter-tree [data-edit="not"][data-path="0"]').click()
            assert page.evaluate("window.pyMicrogliaExplorer.filters.tree.children[0].op") == "NOT"
            assert page.evaluate("window.pyMicrogliaExplorer.data.identities") == expected
            page.reload();page.wait_for_function("window.pyMicrogliaExplorer?.filters?.applied")
            assert page.evaluate("window.pyMicrogliaExplorer.data.identities") == expected
            assert page.evaluate("window.pyMicrogliaExplorer.filters.tree.children[0].op") == "NOT"
            # Off restores every original identity and every cell grid.
            page.locator("#xp-filter-off").click()
            assert page.evaluate("window.pyMicrogliaExplorer.data.identities") == original
            # An empty cohort must never trigger cell-0 requests or leave stale images.
            page.evaluate("(()=>{const f=window.pyMicrogliaExplorer.filters;f.tree={field:'identity',cmp:'gt',value:1000000};f.mode='keep';f.render();f.dirty();})()")
            page.locator("#xp-filter-apply").click()
            page.wait_for_function("window.pyMicrogliaExplorer.filters.applied?.included===0")
            assert page.locator("#xp-cell-select").is_disabled()
            assert page.locator(".xp-grid-cell").count() == 0
            page.locator("#xp-filter-off").click()
            page.wait_for_function("window.pyMicrogliaExplorer.exact && window.pyMicrogliaExplorer.image")
            # A warm selection includes exact history and the selected-frame image.
            started = perf_counter()
            page.select_option("#xp-cell-select", str(original[1]))
            page.wait_for_function("(()=>{const e=window.pyMicrogliaExplorer;return e.exact?.cell.identity===e.state.identity && e.history?.identity===e.state.identity && (e.cellImageCache.get(e.state.frame)?.image || e.image && e.observation());})()")
            selection_seconds = perf_counter()-started
            for theme in ("light", "dark"):
                page.evaluate("theme=>document.documentElement.dataset.pmTheme=theme", theme)
                page.screenshot(path=str(output / f"filters-{theme}.png"))
            page.set_viewport_size({"width": 800, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
            page.screenshot(path=str(output / "filters-narrow.png"))
            assert not jobs and not errors, {"jobs": jobs, "errors": errors}
        finally:
            browser.close()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    receipt = {"first_usable_seconds": first_usable, "selected_cell_seconds": selection_seconds,
               "original_cells": len(original), "filtered_cells": len(expected), "errors": errors,
               "analysis_jobs": jobs, "source_unchanged": True}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **receipt}))


if __name__ == "__main__": main()
