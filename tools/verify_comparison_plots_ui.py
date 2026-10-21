"""Check repeated plots, linked selection and recovery against saved measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

from playwright.sync_api import sync_playwright
from verify_exploration_rhythms_ui import load
from verify_measurement_picker_ui import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:62857")
    args = parser.parse_args()
    output = Path(tempfile.mkdtemp(prefix="pymicroglia-comparison-plots-"))
    real = args.run.resolve()
    source = real / "measure/95_A3/cell_frame.csv"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    errors, jobs = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: jobs.append(request.url) if request.method == "POST" and "jobs" in request.url else None)
        page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
        try:
            started = perf_counter()
            load(page, args.url, real)
            page.wait_for_function("window.pyMicrogliaExplorer.exact && window.pyMicrogliaExplorer.image && [...window.pyMicrogliaExplorer.tissueCache.values()].some(row=>row.image)")
            first_usable = perf_counter() - started
            for family in ("barcode", "tissue-map"):
                page.locator(f"#xp-{family}-comparisons").get_by_role("button", name="Add plot", exact=True).click()
            page.wait_for_function("window.pyMicrogliaExplorer.hits['barcode-2']?.cells && window.pyMicrogliaExplorer.hits['tissue-map-2']?.length")
            assert page.locator("[data-plot-type]").count() == 12
            page.locator("#xp-map-layer-tissue-map-2").select_option("tectonics:cumulative_occupancy")
            page.wait_for_function("document.querySelector('#xp-map-caption-tissue-map-2').textContent.toLowerCase().includes('cumulative')")
            assert page.locator("#xp-map-layer").input_value() == "tectonics:first_coverage"
            for family in ("barcode", "tissue-map"):
                primary = page.locator(f"#xp-{family}").bounding_box()
                second = page.locator(f"#xp-{family}-2").bounding_box()
                assert second["x"] > primary["x"] + primary["width"] - 1, (family, primary, second)
            page.evaluate("""() => {
              const e=window.pyMicrogliaExplorer,api=e.options.api;
              e.comparisonTimings=[];
              e.options.api=async(...args)=>{const start=performance.now();try{return await api(...args);}finally{e.comparisonTimings.push({endpoint:args[0].split('?')[0],milliseconds:performance.now()-start});}};
            }""")
            started = perf_counter()
            selection = page.evaluate("""() => {
              const e=window.pyMicrogliaExplorer,id='barcode-2',p=e.hits[id],c=e.$(id),r=c.getBoundingClientRect(),row=p.cells.length-1;
              const x=p.left+.4*(p.right-p.left),y=p.top+(row+.5)*p.rowHeight;
              c.dispatchEvent(new MouseEvent('click',{clientX:r.left+x,clientY:r.top+y,bubbles:true}));
              return {identity:p.cells[row].identity,frame:e.nearestFrame(.4)};
            }""")
            page.wait_for_function("expected=>{const e=window.pyMicrogliaExplorer;return e.state.identity===expected.identity && e.state.frame===expected.frame && e.exact?.cell.identity===expected.identity && e.history?.identity===expected.identity && e.frameData?.frame_index===expected.frame && !e.secondaryPending;}", arg=selection)
            selection_seconds = perf_counter() - started
            request_timings = page.evaluate("window.pyMicrogliaExplorer.comparisonTimings")
            page.locator("#xp-barcode-comparisons").screenshot(path=str(output / "real-barcodes.png"))
            page.locator("#xp-tissue-map-comparisons").screenshot(path=str(output / "real-tissue-maps.png"))
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("window.pyMicrogliaExplorer?.exact && window.pyMicrogliaExplorer.comparisons.views.has('tissue-map-2')")
            assert page.locator("#xp-map-layer-tissue-map-2").input_value() == "tectonics:cumulative_occupancy"
            assert page.evaluate("window.pyMicrogliaExplorer.state.identity") == selection["identity"]
            synthetic = output / "fixture"
            synthetic.mkdir()
            fixture(synthetic)
            load(page, args.url, synthetic)
            assert page.locator("[data-plot-type]").count() == 10
            page.locator("#xp-barcode-comparisons").get_by_role("button", name="Add plot", exact=True).click()
            page.locator("#xp-barcode-metric-barcode-2-picker > summary").click()
            page.locator('#xp-barcode-metric-barcode-2-browser [data-measurement-group="shape"]').click()
            page.locator('#xp-barcode-metric-barcode-2-browser input[value="area_px"]').check()
            assert page.locator("#xp-barcode-metric").input_value() == "light"
            assert page.locator("#xp-barcode-metric-barcode-2").input_value() == "area_px"
            page.locator("#xp-style-button-barcode-2").click()
            page.locator("#xp-style-barcode-2-height").fill("240")
            page.locator("#xp-style-barcode-2-height").dispatch_event("change")
            assert page.evaluate("window.pyMicrogliaExplorer.styles['barcode-2'].height") == 240
            page.locator("#xp-barcode-comparisons").get_by_role("button", name="Add plot", exact=True).click()
            assert page.locator("#xp-barcode-3").count() == 1
            page.locator("#xp-barcode-panel-barcode-3").get_by_role("button", name="Remove", exact=True).click()
            assert page.locator("#xp-barcode-2").count() == 1
            assert not page.evaluate("window.pyMicrogliaExplorer.styles.barcode?.height")
            page.locator("#xp-style-button-barcode-2").click()
            page.locator("#xp-barcode-columns").select_option("1")
            first = page.locator("#xp-barcode").bounding_box()
            second = page.locator("#xp-barcode-2").bounding_box()
            assert second["y"] > first["y"] + first["height"]
            page.locator("#xp-barcode-columns").select_option("2")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("window.pyMicrogliaExplorer?.exact && window.pyMicrogliaExplorer.hits['barcode-2']")
            assert page.locator("#xp-barcode-metric-barcode-2").input_value() == "area_px"
            assert page.locator("#xp-barcode-columns").input_value() == "2"
            assert page.evaluate("window.pyMicrogliaExplorer.styles['barcode-2'].height") == 240
            for theme, width in (("dark", 1440), ("light", 800)):
                page.set_viewport_size({"width": width, "height": 1000})
                page.evaluate("theme=>document.documentElement.dataset.pmTheme=theme", theme)
                page.locator("#xp-barcode-comparisons").scroll_into_view_if_needed()
                page.wait_for_function("document.documentElement.scrollWidth<=innerWidth")
                page.locator("#xp-barcode-comparisons").screenshot(path=str(output / f"comparison-{theme}-{width}.png"))
            page.locator("#xp-barcode-panel-barcode-2").get_by_role("button", name="Remove", exact=True).click()
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("window.pyMicrogliaExplorer?.exact")
            assert page.locator("[data-plot-type]").count() == 10
            ids = page.locator("#exploration-host [id]").evaluate_all("rows=>rows.map(row=>row.id)")
            assert len(ids) == len(set(ids))
            assert not errors and not jobs, {"errors": errors, "jobs": jobs}
        except Exception as error:
            print("Failure:",repr(error),flush=True)
            try:
                page.screenshot(path=str(output / "failed.png"),timeout=5000)
            except Exception:
                pass
            print(json.dumps({"output": str(output), "errors": errors, "jobs": jobs}))
            raise
        finally:
            browser.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    receipt = {"first_usable_seconds": first_usable, "selected_cell_seconds": selection_seconds,
               "selection_request_timings": request_timings, "independent_metrics_and_layers": True,
               "linked_selection": True, "saved_layout_and_appearance": True,
               "themes_and_narrow_width": True, "source_unchanged": True, "errors": errors, "analysis_jobs": jobs}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"output": str(output), **receipt}))


if __name__ == "__main__":
    main()
