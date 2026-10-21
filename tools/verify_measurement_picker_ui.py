"""Verify grouped metric menus on saved data; no analyses or tracking edits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

from playwright.sync_api import sync_playwright
from verify_exploration_rhythms_ui import load


def fixture(root):
    folder = root / "measure" / "picker_fixture"
    folder.mkdir(parents=True)
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "integrated_density", "signal_mean",
                         "area_px", "branch_count", "speed", "cumulative_unique_px", "nearest_edge_px"])
        for identity in (19, 26):
            for frame in range(16):
                if identity == 19 and frame == 3:
                    continue
                writer.writerow([identity, frame, frame / 2, 10 * frame + identity, frame + 2,
                                 20 + frame, 3 + frame % 2, frame / 3, 20 + frame * 4, 12 - frame / 2])
    (root / "manifest.json").write_text(json.dumps({"movies": [{"stem": "picker_fixture"}]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:62857")
    args = parser.parse_args()
    real = args.run.resolve()
    files = [real / "measure/95_A3/cell_frame.csv", *real.glob("AI_Exports/exploration_rhythms/**/statistics.csv")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    output = Path(tempfile.mkdtemp(prefix="pymicroglia-measurement-picker-"))
    fixture_root = output / "fixture"
    fixture_root.mkdir()
    fixture(fixture_root)
    fixture_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in fixture_root.rglob("*") if path.is_file()}
    errors, jobs = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: jobs.append(request.url) if request.method == "POST" and "jobs" in request.url else None)
            page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
            started = perf_counter()
            load(page, args.url, real)
            page.wait_for_function("window.pyMicrogliaExplorer.exact && window.pyMicrogliaExplorer.history && window.pyMicrogliaExplorer.image")
            first_usable = perf_counter() - started
            original = page.evaluate("window.pyMicrogliaExplorer.data.identities")
            page.locator("#xp-rhythm-measurements > summary").click()
            page.locator('#xp-rhythm-metric-list [data-measurement-group="shape"]').click()
            assert page.locator('#xp-rhythm-metric-list input[value="area_px"]').is_disabled()
            assert page.locator('#xp-rhythm-metric-list input[value="area_px"]').locator("..").inner_text().endswith("Not measured yet")
            selected_before = page.evaluate("window.pyMicrogliaExplorer.selectedRhythmMetrics()")
            page.locator('#xp-rhythm-metric-list [data-measurement-group="movement"]').click()
            assert page.evaluate("window.pyMicrogliaExplorer.selectedRhythmMetrics()") == selected_before
            page.screenshot(path=str(output / "real-missing-measurements.png"))
            page.locator("#xp-rhythm-measurements > summary").click()
            started = perf_counter()
            page.locator("#xp-cell-select").select_option(str(original[1]))
            page.wait_for_function("(()=>{const e=window.pyMicrogliaExplorer;return e.exact?.cell.identity===e.state.identity && e.history?.identity===e.state.identity && (e.cellImageCache.get(e.state.frame)?.image || e.image && e.observation());})()")
            selection_seconds = perf_counter() - started
            assert page.evaluate("window.pyMicrogliaExplorer.data.identities") == original

            load(page, args.url, fixture_root)
            page.locator("#xp-rhythm-measurements > summary").click()
            for group, metric in (("shape", "area_px"), ("movement", "speed"), ("signal", "signal_mean")):
                page.locator(f'#xp-rhythm-metric-list [data-measurement-group="{group}"]').click()
                page.locator(f'#xp-rhythm-metric-list input[value="{metric}"]').check()
            chosen = page.evaluate("window.pyMicrogliaExplorer.selectedRhythmMetrics()")
            assert {"area_px", "speed", "signal_mean"} <= set(chosen)
            assert page.locator("#xp-rhythm-metric-list .xp-measurement-selected").inner_text().count("Area") == 1
            page.locator('#xp-rhythm-metric-list [data-measurement-group="movement"]').click()
            page.locator('#xp-rhythm-metric-list-panel-movement').get_by_role("button", name="Clear group").click()
            assert "speed" not in page.evaluate("window.pyMicrogliaExplorer.selectedRhythmMetrics()")
            assert page.locator('#xp-rhythm-metric-list input[value="area_px"]').is_checked()
            page.locator('#xp-rhythm-metric-list input[value="speed"]').check()
            page.locator("#xp-rhythm-select-all").click()
            assert page.evaluate("document.querySelectorAll('#xp-rhythm-metric-list input:disabled:checked').length") == 0
            page.locator("#xp-rhythm-measurements > summary").click()

            page.locator('#xp-trace-panel .settings:not(.xp-plot-settings)').evaluate("el=>el.open=true")
            page.locator("#xp-trace-measurements-picker > summary").click()
            for group, metric in (("signal", "signal_mean"), ("shape", "area_px")):
                page.locator(f'#xp-trace-measurements-list [data-measurement-group="{group}"]').click()
                page.locator(f'#xp-trace-measurements-list input[value="{metric}"]').check()
            page.wait_for_function("window.pyMicrogliaExplorer.gridRows.get(19)?.values_by_metric?.area_px")
            assert page.locator('#xp-trace-grid-measurements-list input[value="area_px"]').is_checked()
            assert page.evaluate("window.pyMicrogliaExplorer.traceMetrics()") == ["integrated_density", "signal_mean", "area_px"]
            page.locator("#xp-trace-measurements-picker > summary").click()
            page.locator("#xp-barcode-metric-picker > summary").click()
            page.locator('#xp-barcode-metric-browser [data-measurement-group="shape"]').click()
            page.locator('#xp-barcode-metric-browser input[value="area_px"]').check()
            assert page.locator("#xp-barcode-metric").input_value() == "area_px"
            assert page.locator("#xp-trace-metric").input_value() == "light"
            assert not page.locator("#xp-barcode-metric-picker").evaluate("el=>el.open")
            page.reload()
            page.wait_for_function("window.pyMicrogliaExplorer?.rhythmSchema && window.pyMicrogliaExplorer.exact")
            assert page.evaluate("window.pyMicrogliaExplorer.tracePicks('measurements')") == ["signal_mean", "area_px"]
            assert page.locator("#xp-barcode-metric").input_value() == "area_px"
            assert page.locator('#xp-rhythm-metric-list [aria-selected="true"]').get_attribute("data-measurement-group") == "movement"
            assert page.locator('#xp-trace-measurements-list [aria-selected="true"]').get_attribute("data-measurement-group") == "shape"
            assert page.locator("[data-plot-type]").count() == 9
            ids = page.locator(".xp-measurement-browser [id]").evaluate_all("elements=>elements.map(el=>el.id)")
            assert len(ids) == len(set(ids))
            for theme, width in (("dark", 1440), ("light", 800)):
                page.set_viewport_size({"width": width, "height": 1000})
                page.evaluate("theme=>document.documentElement.dataset.pmTheme=theme", theme)
                page.locator("#xp-rhythm-measurements > summary").click()
                tab = page.locator('#xp-rhythm-metric-list [data-measurement-group="shape"]')
                tab.click()
                tab.press("ArrowRight")
                assert page.locator('#xp-rhythm-metric-list [aria-selected="true"]').get_attribute("data-measurement-group") == "branches"
                box = page.locator(".xp-rhythm-choices").bounding_box()
                assert box["x"] >= 0 and box["x"] + box["width"] <= width
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                page.screenshot(path=str(output / f"groups-{theme}-{width}.png"))
                page.locator('#xp-rhythm-metric-list [aria-selected="true"]').press("Escape")
            # Old workspaces have no browsing state and retain their metric values.
            page.evaluate("()=>{const key='pm-exploration-session-v1',old=JSON.parse(localStorage.getItem(key));delete old.measurementTabs;localStorage.setItem(key,JSON.stringify(old));}")
            page.reload()
            page.wait_for_function("window.pyMicrogliaExplorer?.rhythmSchema && window.pyMicrogliaExplorer.exact")
            assert page.locator("#xp-barcode-metric").input_value() == "area_px"
            assert not errors and not jobs, {"errors": errors, "jobs": jobs}
        except Exception:
            page.screenshot(path=str(output / "failed.png"))
            print(json.dumps({"output": str(output), "errors": errors, "jobs": jobs}))
            raise
        finally:
            browser.close()
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in {**hashes, **fixture_hashes}.items())
    receipt = {"first_usable_seconds": first_usable, "selected_cell_seconds": selection_seconds,
               "cross_group_selection": True, "independent_plot_metrics": True, "saved_state": True,
               "missing_metrics_disabled": True, "keyboard_tabs": True, "themes_and_narrow_width": True,
               "canonical_views": 9, "source_unchanged": True, "errors": errors, "analysis_jobs": jobs}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"output": str(output), **receipt}))


if __name__ == "__main__":
    main()
