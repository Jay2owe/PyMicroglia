"""Check clear, swappable rhythm settings and shared multi-metric trace controls."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

from playwright.sync_api import sync_playwright
from verify_exploration_rhythms_ui import fixture, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:62855")
    args = parser.parse_args()
    real = args.run.resolve()
    paths = [real / "measure/95_A3/cell_frame.csv",
             *real.glob("AI_Exports/exploration_rhythms/**/result.json"),
             *real.glob("AI_Exports/exploration_rhythms/**/statistics.csv")]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = Path(tempfile.mkdtemp(prefix="pymicroglia-rhythm-settings-"))
    fixture(output)
    errors, requests = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: requests.append(json.loads(request.post_data))
                    if request.method == "POST" and "/rhythm-jobs" in request.url else None)
            page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
            started = perf_counter();load(page, args.url, real)
            first_usable = perf_counter() - started
            assert page.locator("#xp-rhythm-recipe").is_visible()
            assert page.locator("#xp-rhythm-detrend").is_visible()
            assert not page.locator("#xp-rhythm-method").is_visible()
            assert page.locator("#xp-rhythm-recipe").input_value() == "fft_component"
            shown = page.locator("#xp-rhythm-applied").inner_text()
            assert "robust linear" in shown.lower()
            result_id = page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms.analysis_id")
            started = perf_counter()
            page.locator("#xp-rhythm-recipe").select_option("lomb")
            page.locator("#xp-rhythm-detrend").select_option("none")
            swap_seconds = perf_counter() - started
            assert page.locator("#xp-rhythm-method").input_value() == "lomb"
            assert page.locator("#xp-rhythm-test").input_value() == "lomb"
            assert page.locator("#xp-rhythm-applied").inner_text() == shown
            assert "Pending" in page.locator("#xp-rhythm-pending").inner_text()
            assert page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms.analysis_id") == result_id
            assert not requests
            page.screenshot(path=str(output / "pending-method.png"))
            page.reload();page.wait_for_function("window.pyMicrogliaExplorer?.rhythmSchema")
            assert page.locator("#xp-rhythm-recipe").input_value() == "lomb"
            assert page.locator("#xp-rhythm-detrend").input_value() == "none"
            page.locator("#xp-rhythm-use-shown").click()
            assert page.locator("#xp-rhythm-recipe").input_value() == "fft_component"
            assert page.locator("#xp-rhythm-detrend").input_value() == "robust_linear"
            assert page.locator("#xp-rhythm-pending").is_hidden()
            page.locator("#xp-rhythm-detrend").select_option("polynomial")
            assert page.locator("#xp-rhythm-detrend").input_value() == "polynomial"
            assert page.locator("#xp-rhythm-recipe").input_value() == "fft_nlls"
            assert page.locator("#xp-rhythm-test").input_value() == "lomb"
            assert "requires" in page.locator("#xp-rhythm-pending").inner_text()
            page.locator("#xp-rhythm-use-shown").click()
            # Multi-metric controls are usable from the grid, with one shared state.
            page.locator("#xp-trace-grid-panel").scroll_into_view_if_needed()
            page.locator("#xp-trace-grid-measurements-picker").evaluate("e=>e.open=true")
            page.locator('#xp-trace-grid-measurements-list input[value="signal_mean"]').check()
            page.locator("#xp-trace-grid-measurements-picker").evaluate("e=>e.open=false")
            page.wait_for_function("window.pyMicrogliaExplorer.gridRows.size===94 && window.pyMicrogliaExplorer.gridRows.get(6)?.values_by_metric?.signal_mean")
            page.locator("#xp-trace-grid-measurement-layout").select_option("overlay")
            assert page.locator("#xp-trace-layout").input_value() == "overlay"
            assert page.evaluate("window.pyMicrogliaExplorer.traceMetrics().length") == 2
            page.wait_for_function("[...document.querySelectorAll('#xp-trace-grid canvas')].some(c=>c.dataset.traceLayout==='overlay')")
            page.screenshot(path=str(output / "grid-metrics-overlay.png"))
            page.locator("#xp-trace-grid-measurement-layout").select_option("stack")
            assert page.locator("#xp-trace-layout").input_value() == "stack"
            assert len(json.loads(page.locator("#xp-trace").get_attribute("data-trace-series"))) == 2
            page.locator("#xp-trace-cells-picker").evaluate("e=>e.open=true")
            page.locator("#xp-trace-cells-all").click()
            page.locator("#xp-trace-cells-picker").evaluate("e=>e.open=false")
            page.locator("#xp-trace-grid-layout").select_option("stack")
            page.locator("#xp-trace-grid-panel").scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#xp-trace-grid canvas').dataset.traceSeries")
            assert page.locator("#xp-trace-grid canvas").first.bounding_box()["height"] <= 200
            assert page.locator("#xp-trace-grid .xp-grid-cell:not([hidden])").count() == 94
            page.locator("#xp-trace-grid-layout").select_option("grid")
            assert page.evaluate("(()=>{const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return ids.length===new Set(ids).size;})()")
            assert not requests
            for theme, width in (("dark", 1440), ("light", 800)):
                page.set_viewport_size({"width": width, "height": 1000})
                page.locator("#micro-theme").select_option(theme)
                page.evaluate("scrollTo(0,0)")
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                for selector in ("#xp-rhythm-recipe", "#xp-rhythm-detrend"):
                    box = page.locator(selector).bounding_box()
                    assert box["x"] >= 0 and box["x"] + box["width"] <= width
                page.screenshot(path=str(output / f"controls-{theme}-{width}.png"))
            # An actual native job uses the new choices; real inputs are never run.
            page.set_viewport_size({"width": 1440, "height": 1000})
            load(page, args.url, output)
            page.locator("#xp-rhythm-recipe").select_option("lomb")
            page.locator("#xp-rhythm-detrend").select_option("none")
            page.locator("#xp-rhythm-measurements").evaluate("e=>e.open=true")
            page.locator("#xp-rhythm-clear").click()
            for metric in ("integrated_density_mean3", "signal_mean"):
                page.locator(f'#xp-rhythm-metric-list input[value="{metric}"]').check()
            page.locator("#xp-rhythm-measurements").evaluate("e=>e.open=false")
            page.locator("#xp-run-rhythms").click()
            page.wait_for_function("window.pyMicrogliaExplorer.data.results.rhythms.analysis_id && !window.pyMicrogliaExplorer.rhythmRunning", timeout=90000)
            result = page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms")
            assert result["settings"]["detrend"] == "none"
            assert all(r["method"] == "lomb" and r["significance_method"] == "lomb" for r in result["rows"])
            assert len(result["rows"]) == 4
            assert requests[-1]["options"]["fit_method"] == "lomb"
            assert requests[-1]["options"]["fft_component_test"] is False
            assert page.locator("#xp-rhythm-pending").is_hidden()
            first_id = result["analysis_id"]
            page.locator("#xp-rhythm-detrend").select_option("polynomial")
            assert "Apply" in page.locator("#xp-run-rhythms").inner_text()
            page.locator("#xp-run-rhythms").click()
            page.wait_for_function(f"window.pyMicrogliaExplorer.data.results.rhythms.analysis_id!=='{first_id}' && !window.pyMicrogliaExplorer.rhythmRunning", timeout=90000)
            assert page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms.settings.detrend") == "polynomial"
            assert len(requests) == 2 and all(r["run"] == str(output) for r in requests)
            page.reload();page.wait_for_function("window.pyMicrogliaExplorer?.rhythmSchema")
            assert page.locator("#xp-rhythm-recipe").input_value() == "lomb"
            assert page.locator("#xp-rhythm-detrend").input_value() == "polynomial"
            assert page.locator("#xp-rhythm-pending").is_hidden()
            assert not errors, errors
        except Exception:
            page.screenshot(path=str(output / "failed.png"))
            print(json.dumps({"screenshots": str(output), "browser_errors": errors}))
            raise
        finally:
            browser.close()
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    receipt = {"first_usable_seconds": first_usable, "method_swap_seconds": swap_seconds,
               "real_inputs_and_results_unchanged": True, "synthetic_jobs": len(requests),
               "browser_errors": errors, "screenshots": str(output)}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
