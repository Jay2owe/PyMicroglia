"""Exercise cell and measurement comparisons against a saved, immutable run."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:62855")
    args = parser.parse_args()
    run = args.run.resolve()
    inputs = [run / "measure/95_A3/cell_frame.csv",
              *run.glob("AI_Exports/exploration_rhythms/**/statistics.csv"),
              *run.glob("AI_Exports/exploration_rhythms/**/result.json")]
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    output = Path(tempfile.mkdtemp(prefix="pymicroglia-trace-layouts-"))
    errors, jobs = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: jobs.append(request.url)
                    if request.method == "POST" and "rhythm-jobs" in request.url else None)
            page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
            started = perf_counter()
            page.goto(args.url + "/?" + urlencode({"run": str(run)}))
            page.wait_for_function("window.pyMicrogliaExplorer?.exact && window.pyMicrogliaExplorer.gridRows.size===94")
            first_usable = perf_counter() - started
            grid_panel=page.locator('#xp-trace-grid-panel')
            grid_panel.scroll_into_view_if_needed()
            assert grid_panel.locator('.xp-panel-main h3').inner_text() == 'All cell traces'
            for name in ('Edit plot','Settings','Export'):
                assert grid_panel.locator('.xp-panel-actions').get_by_role('button',name=name,exact=True).count() == 1
            assert not grid_panel.locator('#xp-trace-grid-columns').count()
            switch=page.get_by_role("switch",name="Overlay cells",exact=True)
            assert switch.is_visible()
            assert not grid_panel.locator("#xp-trace-grid-overlay-toggle").count()
            assert switch.bounding_box()["y"]+switch.bounding_box()["height"] <= grid_panel.bounding_box()["y"]
            assert page.locator("#xp-trace-grid-layout").is_hidden()
            columns=page.locator('#xp-trace-grid-columns').bounding_box()
            assert columns['y']+columns['height'] <= grid_panel.bounding_box()['y']
            main=grid_panel.locator('.xp-panel-main')
            assert main.locator('.controls').bounding_box()['x'] <= main.locator('h3').bounding_box()['x']+main.locator('h3').bounding_box()['width']+8
            original_trace_style=page.evaluate('window.pyMicrogliaExplorer.style("trace")')
            grid_panel.get_by_role('button',name='Edit plot',exact=True).click()
            for key,value in (('height','180'),('lineWidth','2.5'),('background','#fefefe')):
                control=page.locator(f'#xp-style-trace-grid-{key}')
                if key=='background':
                    control.evaluate('(el,value)=>el.value=value',value)
                else:
                    control.fill(value)
                control.dispatch_event('change')
            page.wait_for_function('document.querySelector("#xp-trace-grid canvas[data-logical-height=\\"180\\"]")')
            assert page.evaluate('window.pyMicrogliaExplorer.style("trace")') == original_trace_style
            assert page.evaluate('window.pyMicrogliaExplorer.style("trace-grid").lineWidth') == 2.5
            assert page.locator('#xp-trace-grid canvas').first.evaluate('(c)=>[...c.getContext("2d").getImageData(0,0,1,1).data]') == [254,254,254,255]
            page.reload()
            page.wait_for_function('window.pyMicrogliaExplorer?.gridRows?.size===94')
            assert page.evaluate('window.pyMicrogliaExplorer.styles["trace-grid"].height') == 180
            assert page.evaluate('window.pyMicrogliaExplorer.styles["trace-grid"].lineWidth') == 2.5
            grid_panel.get_by_role('button',name='Edit plot',exact=True).click()
            grid_panel.get_by_role('button',name='Export',exact=True).click()
            assert grid_panel.locator('.xp-export select').inner_text() == 'All measured cell traces · traces'
            with page.expect_response(lambda r: '/api/availability/all_cell_trace_grid?' in r.url, timeout=90000):
                grid_panel.locator('.xp-export').get_by_role('button',name='Configure export',exact=True).click()
            page.locator('#action-dialog').wait_for(state='visible')
            page.locator('#close-action').click()
            grid_panel.get_by_role('button',name='Export',exact=True).click()
            assert page.locator("#xp-trace-grid-settings").evaluate("el=>el.open")
            for control in ("scale","fit","overlay-scale","measurement-layout","measurement-scale"):
                assert page.locator(f"#xp-trace-grid-{control}").is_visible()
            assert page.locator("#xp-trace-time-guides").is_visible()
            assert page.locator("#xp-trace-time-guides-offset").is_visible()
            assert page.locator("#xp-trace-grid-measurements-picker").is_visible()
            assert page.locator("#xp-trace-cells-picker").is_visible()
            grid_panel.get_by_role('button',name='Settings',exact=True).click()
            assert not page.locator("#xp-trace-grid-settings").evaluate("el=>el.open")
            grid_panel.get_by_role('button',name='Settings',exact=True).click()
            assert page.locator('#xp-trace-grid-settings').evaluate('(el)=>el.open')
            page.locator("#xp-cell-select").select_option("6")
            page.wait_for_function("window.pyMicrogliaExplorer.exact?.cell.identity===6")
            page.locator('#xp-trace-panel .settings:not(.xp-plot-settings)').evaluate("e=>e.open=true")
            page.locator("#xp-trace-measurements-picker").evaluate("e=>e.open=true")
            page.locator('#xp-trace-measurements-list input[value="signal_mean"]').check()
            page.wait_for_function("window.pyMicrogliaExplorer.gridRows.get(6)?.values_by_metric?.signal_mean")
            page.locator("#xp-trace-measurements-picker").evaluate("e=>e.open=false")
            started = perf_counter()
            page.locator("#xp-trace-layout").select_option("overlay")
            overlay_seconds = perf_counter() - started
            assert page.locator("#xp-trace").get_attribute("data-trace-layout") == "overlay"
            assert page.locator("#xp-trace").get_attribute("data-trace-scale") == "each trace 0–1"
            series = json.loads(page.locator("#xp-trace").get_attribute("data-trace-series"))
            assert [row["metric"] for row in series] == ["integrated_density", "signal_mean"]
            assert len({row["colour"] for row in series}) == 2
            page.wait_for_function("JSON.parse(document.getElementById('xp-peer-one-trace').dataset.traceSeries||'[]').length===2")
            page.locator("#xp-trace-overlay-scale").select_option("native")
            assert page.locator("#xp-trace").get_attribute("data-trace-scale") == "original units"
            page.locator("#xp-trace-overlay-scale").select_option("normalized")
            page.locator("#xp-trace-fit").check()
            page.wait_for_function("JSON.parse(document.getElementById('xp-trace').dataset.traceSeries).every(s=>s.fit==='available')", timeout=45000)
            page.locator("#xp-trace-panel").scroll_into_view_if_needed()
            page.screenshot(path=str(output / "measurements-overlay.png"))
            page.locator("#xp-trace-layout").select_option("stack")
            assert page.locator("#xp-trace").get_attribute("data-trace-layout") == "stack"
            page.screenshot(path=str(output / "measurements-stack.png"))
            # More lanes grow vertically while the cell image stays square.
            page.locator("#xp-trace-measurements-picker").evaluate("e=>e.open=true")
            page.locator('#xp-trace-measurements-list input[value="signal_median"]').check()
            page.locator("#xp-trace-measurements-picker").evaluate("e=>e.open=false")
            assert float(page.locator("#xp-trace").get_attribute("data-logical-height")) >= 255
            dimensions = page.locator("#xp-selected-mini").bounding_box()
            assert abs(dimensions["width"] - dimensions["height"]) < 2
            assert json.loads(page.locator("#xp-trace").get_attribute("data-trace-series"))[-1]["fit"] == "unavailable"
            page.locator("#xp-trace-measurements-picker").evaluate("e=>e.open=true")
            page.locator('#xp-trace-measurements-list input[value="signal_median"]').uncheck()
            page.locator("#xp-trace-measurements-picker").evaluate("e=>e.open=false")
            # Comparing cells uses the existing trace view, with exact sparse samples.
            page.locator("#xp-trace-grid-panel").scroll_into_view_if_needed()
            page.locator("#xp-trace-cells-picker").evaluate("e=>e.open=true")
            page.locator("#xp-trace-cells-clear").click()
            for identity in (3, 6, 7):
                page.locator(f'#xp-trace-cells-list input[value="{identity}"]').check()
            page.locator("#xp-trace-cells-picker").evaluate("e=>e.open=false")
            page.evaluate("""() => {window.layoutStrokes=[];const stroke=CanvasRenderingContext2D.prototype.stroke;
              CanvasRenderingContext2D.prototype.stroke=function(...args){if(this.canvas.id==='xp-trace-comparison-canvas')window.layoutStrokes.push({colour:this.strokeStyle,dash:this.getLineDash()});return stroke.apply(this,args);};} """)
            page.get_by_role("switch",name="Overlay cells",exact=True).check()
            assert page.locator("#xp-trace-grid").is_hidden()
            comparison = page.locator("#xp-trace-comparison-canvas")
            series = json.loads(comparison.get_attribute("data-trace-series"))
            assert len(series) == 6 and {s["identity"] for s in series} == {3, 6, 7}
            strokes = page.evaluate("window.layoutStrokes")
            assert {s["colour"] for s in strokes if not s["dash"]} >= {s["colour"] for s in series}
            assert {s["colour"] for s in strokes if s["dash"]} >= {s["colour"] for s in series}
            page.locator("#xp-trace-grid-overlay-scale").select_option("normalized")
            assert comparison.get_attribute("data-trace-scale") == "each trace 0–1"
            frame = page.evaluate("window.pyMicrogliaExplorer.state.frame")
            comparison.click(position={"x": 40, "y": 8})
            assert page.evaluate("window.pyMicrogliaExplorer.state.identity") == 3
            assert page.evaluate("window.pyMicrogliaExplorer.state.frame") == frame
            hit = comparison.evaluate("""canvas=>{const e=window.pyMicrogliaExplorer,p=canvas.traceHitSeries.find(s=>s.identity===7),i=p.indices.indexOf(70),b=p.box,[lo,hi]=p.range;
              return {x:b.left+e.fraction(70)*(b.right-b.left),y:b.bottom-3-(p.values[i]-lo)/(hi-lo)*(b.bottom-b.top-6)};} """)
            comparison.click(position=hit)
            assert page.evaluate("window.pyMicrogliaExplorer.state.identity") == 7
            assert page.evaluate("window.pyMicrogliaExplorer.state.frame") == 70
            # One shared cursor moves the images, trace, and comparison together.
            before = page.evaluate("window.pyMicrogliaExplorer.state.frame")
            box = comparison.bounding_box()
            comparison.click(position={"x": box["width"] * .72, "y": box["height"] * .65})
            assert page.evaluate("window.pyMicrogliaExplorer.state.frame") != before
            page.screenshot(path=str(output / "cells-overlay.png"))
            page.get_by_role("switch",name="Overlay cells",exact=True).uncheck()
            page.locator("#xp-trace-grid-columns").select_option("1")
            assert comparison.is_hidden()
            ids = page.locator("#xp-trace-grid .xp-grid-cell:not([hidden])").evaluate_all("els=>els.map(e=>+e.dataset.identity)")
            assert ids == page.evaluate("window.pyMicrogliaExplorer.data.identities")
            starts=[]
            for identity in ids[:3]:
                tile=page.locator(f'#xp-trace-grid .xp-grid-cell[data-identity="{identity}"]')
                tile.scroll_into_view_if_needed()
                page.wait_for_function("id=>document.querySelector(`#xp-trace-grid .xp-grid-cell[data-identity=\"${id}\"] canvas`).dataset.logicalWidth===String(Math.round(document.querySelector(`#xp-trace-grid .xp-grid-cell[data-identity=\"${id}\"] canvas`).getBoundingClientRect().width))",arg=identity)
                starts.append(tile.locator('canvas').evaluate('el=>el.gridBox.left'))
            assert len(set(starts)) == 1
            assert page.locator('#xp-trace-grid-layout option[value="stack"]').count() == 0
            page.screenshot(path=str(output / "cells-one-column.png"))
            # Settings reload; old sessions and single-metric grids still work.
            page.reload()
            page.wait_for_function("window.pyMicrogliaExplorer?.exact && window.pyMicrogliaExplorer.gridRows.size===94")
            assert page.locator("#xp-trace-layout").input_value() == "stack"
            assert page.locator("#xp-trace-grid-layout").input_value() == "grid"
            assert page.locator("#xp-trace-grid-columns").input_value() == "1"
            assert page.evaluate("window.pyMicrogliaExplorer.tracePicks('cells')") == [3, 6, 7]
            assert page.evaluate("window.pyMicrogliaExplorer.tracePicks('measurements')") == ["signal_mean"]
            assert page.evaluate('window.pyMicrogliaExplorer.styles["trace-grid"].height') == 180
            page.get_by_role("switch",name="Overlay cells",exact=True).uncheck()
            assert page.locator("#xp-trace-grid .xp-grid-cell:not([hidden])").count() == 94
            for theme, width in (("dark", 1440), ("light", 800)):
                page.set_viewport_size({"width": width, "height": 1000})
                page.locator("#micro-theme").select_option(theme)
                page.get_by_role("switch",name="Overlay cells",exact=True).check()
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                page.locator("#xp-trace-grid-panel").scroll_into_view_if_needed()
                page.screenshot(path=str(output / f"comparison-{theme}-{width}.png"))
            # Upgrade a workspace saved before the multi-measurement picker.
            page.evaluate("""() => {const key='pm-exploration-session-v1',old=JSON.parse(localStorage.getItem(key));
              old.controls['xp-trace-grid-layout']='stack';old.controls['xp-trace-grid-columns']='6';delete old.traceMeasurements;delete old.traceGridControlsVersion;old.folds['xp-trace-grid-settings']=false;old.controls['xp-trace-secondary']='signal_median';localStorage.setItem(key,JSON.stringify(old));} """)
            page.reload()
            page.wait_for_function("window.pyMicrogliaExplorer?.exact")
            assert page.evaluate("window.pyMicrogliaExplorer.tracePicks('measurements')") == ["signal_median"]
            assert page.locator("#xp-trace-grid-settings").evaluate("el=>el.open")
            assert page.locator("#xp-trace-grid-layout").input_value() == "grid"
            assert page.locator("#xp-trace-grid-columns").input_value() == "1"
            assert not errors and not jobs, (errors, jobs)
        except Exception:
            page.screenshot(path=str(output / "failed.png"))
            print(json.dumps({"screenshots": str(output), "browser_errors": errors,
                              "analysis_jobs": jobs}))
            raise
        finally:
            browser.close()
    assert fingerprints == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    receipt = {"first_usable_seconds": first_usable, "overlay_seconds": overlay_seconds,
               "measurements_and_saved_results_unchanged": True, "browser_errors": errors,
               "analysis_jobs": jobs, "screenshots": str(output)}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
