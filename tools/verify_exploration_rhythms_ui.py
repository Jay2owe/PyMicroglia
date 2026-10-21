"""Exercise rhythm controls on a real run and the complete synthetic job flow."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
from urllib.parse import urlencode

import numpy as np
from playwright.sync_api import sync_playwright
from pymicroglia.measure.signal import centered_frame_mean


def fixture(root):
    movies = []
    for stem, identities in (("synthetic_one", (19, 26)), ("synthetic_two", (7, 11))):
        folder = root / "measure" / stem
        folder.mkdir(parents=True)
        path = folder / "cell_frame.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["identity", "frame_index", "hours", "integrated_density", "integrated_density_mean3", "signal_mean"])
            for index, identity in enumerate(identities):
                frames = np.arange(145)
                signal = np.full(145, 100.) if index else 100 + 20 * np.cos(2 * np.pi * (frames / 2 - 6) / 24)
                averaged = centered_frame_mean(signal, frames)
                for frame in frames:
                    mean_light = 10 + 2 * np.cos(2 * np.pi * (frame / 2 - 11 - 5 * index) / 20)
                    writer.writerow([identity, frame, frame / 2, signal[frame], averaged[frame], mean_light])
        movies.append({"stem": stem})
    (root / "manifest.json").write_text(json.dumps({"movies": movies}))


def load(page, url, run):
    page.goto(url + "/?" + urlencode({"run": str(run)}))
    page.wait_for_function("window.pyMicrogliaExplorer?.data && window.pyMicrogliaExplorer.rhythmSchema", timeout=60000)


def check(page, url, real_run, root):
    errors, jobs = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: jobs.append(r.url) if r.method == "POST" and "/rhythm-jobs" in r.url else None)
    load(page, url, real_run)
    assert page.locator("#xp-run-rhythms").is_enabled()
    assert page.locator("#xp-rhythm-method").input_value() == "fft_nlls"
    assert page.locator("#xp-rhythm-test").input_value() == "fft_component"
    assert page.locator("[data-plot-type]").count() == 9
    for width in (1440, 1100, 800):
        page.set_viewport_size({"width": width, "height": 1000})
        bounds = page.locator("#xp-run-rhythms").bounding_box()
        assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.locator("#xp-rhythm-settings-toggle").click()
    assert page.locator("#xp-rhythm-method").is_visible()
    page.locator("#xp-rhythm-settings-toggle").click()
    page.screenshot(path=str(root / "real_controls.png"))
    assert not jobs  # merely opening the real dataset does not run an analysis

    source = root / "measure/synthetic_one/cell_frame.csv"
    original = source.read_bytes()
    load(page, url, root)
    page.wait_for_function("window.pyMicrogliaExplorer.data.stem==='synthetic_one'")
    page.locator('#xp-rhythm-measurements-toggle').click()
    page.locator('#xp-rhythm-clear').click()
    assert page.locator('#xp-run-rhythms').is_disabled()
    page.locator('#xp-rhythm-select-all').click()
    assert page.locator('#xp-rhythm-metric-list input:checked').count() == 3
    page.locator('#xp-rhythm-clear').click()
    for metric in ('integrated_density_mean3', 'signal_mean'):
        page.locator(f'#xp-rhythm-metric-list input[value="{metric}"]').check()
    assert not jobs  # ticking measurements does not start computation
    page.reload()
    page.wait_for_function("window.pyMicrogliaExplorer?.rhythmSchema && window.pyMicrogliaExplorer.data.stem==='synthetic_one'")
    assert page.evaluate('window.pyMicrogliaExplorer.selectedRhythmMetrics()') == ['integrated_density_mean3', 'signal_mean']
    page.evaluate("""() => {
        window.rhythmProgressSamples=[];
        new MutationObserver(() => {
            const status=document.querySelector('#xp-rhythm-status').textContent;
            const bar=document.querySelector('#xp-rhythm-progress');
            window.rhythmProgressSamples.push({status,hidden:bar.hidden,value:bar.value,max:bar.max});
        }).observe(document.querySelector('#xp-rhythm-status'),{childList:true,subtree:true});
    }""")
    page.locator("#xp-run-rhythms").click()
    page.wait_for_function("!!window.pyMicrogliaExplorer.data.results.rhythms.analysis_id", timeout=120000)
    assert len(jobs) == 1
    progress_samples = page.evaluate("window.rhythmProgressSamples")
    assert any('elapsed' in sample['status'] and '/2 cells' in sample['status']
               and not sample['hidden'] for sample in progress_samples), progress_samples
    assert all(0 <= sample['value'] <= sample['max'] for sample in progress_samples)
    result = page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms")
    assert {r["identity"] for r in result["rows"]} == {19, 26}
    assert result['metrics'] == ['integrated_density_mean3', 'signal_mean']
    assert len(result['rows']) == 4
    assert not result["preview"] and result["settings"]["fft_component_test"]
    assert source.read_bytes() == original
    first = result["rows"][0]
    assert first["phase_fraction"] is not None
    palette = page.evaluate("window.pyMicrogliaExplorer.data.trace_style")
    assert palette == {"rhythmic": "#c0392b", "other": "#7A7A7A"}
    page.locator("#xp-cell-select").select_option("19")
    page.wait_for_function("window.pyMicrogliaExplorer.$('image-panel').querySelector('.xp-rhythm-note').textContent.includes('Cell 19')")
    for identity in (19, 26):
        note = page.locator(f".xp-grid-cell[data-identity='{identity}'] .xp-grid-rhythm")
        assert note.is_visible()
        row = next(r for r in result["rows"] if r["identity"] == identity and r['metric'] == 'integrated_density_mean3')
        if row["status"] == "rhythmic":
            assert "*" in note.inner_text()
            assert "Rhythmic" in note.get_attribute("title")
        else:
            assert "*" not in note.inner_text()
    assert page.locator("#xp-clock-metric").input_value() == "integrated_density_mean3"
    assert page.evaluate("window.pyMicrogliaExplorer.hits['phase-clock'].some(r=>r.identity===19)")
    page.locator("#xp-cell-select").select_option("26")
    page.wait_for_function("window.pyMicrogliaExplorer.$('barcode-panel').querySelector('.xp-rhythm-note').textContent.includes('Cell 26')")
    assert page.locator('#xp-trace').get_attribute('data-trace-colour') == palette['other']
    assert page.locator('#xp-peer-one-trace').get_attribute('data-trace-colour') == palette['rhythmic']
    page.locator('#xp-trace-grid-panel').scroll_into_view_if_needed()
    page.wait_for_function("[...window.pyMicrogliaExplorer.gridTiles.values()].every(t=>t.canvas.dataset.traceColour)")
    assert page.locator('.xp-grid-cell[data-identity="19"] canvas').get_attribute('data-trace-colour') == palette['rhythmic']
    assert page.locator('.xp-grid-cell[data-identity="26"] canvas').get_attribute('data-trace-colour') == palette['other']
    # The selected grid cell keeps its statistical colour; selection uses a border.
    assert page.locator('.xp-grid-cell[data-identity="26"]').get_attribute('class').endswith('is-selected')
    assert page.evaluate("""() => {
        const c=document.querySelector('.xp-grid-cell[data-identity="19"] canvas'),p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
        let red=0;for(let i=0;i<p.length;i+=4)if(p[i]===192&&p[i+1]===57&&p[i+2]===43)red++;
        return red>10 && c.width===Math.round(c.getBoundingClientRect().width*devicePixelRatio);
    }""")
    # Mean-per-pixel light must display its own tested period and phase.
    page.locator("#xp-trace-aggregation").select_option("mean")
    mean_row = next(r for r in result['rows'] if r['identity'] == 26 and r['metric'] == 'signal_mean')
    total_row = next(r for r in result['rows'] if r['identity'] == 26 and r['metric'] == 'integrated_density_mean3')
    assert mean_row['period_hours'] is not None and total_row['period_hours'] is None
    assert page.evaluate("window.pyMicrogliaExplorer.rhythmFor(26,'trace').metric") == 'signal_mean'
    assert page.locator('#xp-trace').get_attribute('data-trace-colour') == palette['rhythmic']
    assert page.locator('.xp-grid-cell[data-identity="26"] canvas').get_attribute('data-trace-colour') == palette['rhythmic']
    assert "not analysed" not in page.locator(".xp-selected-card .xp-rhythm-note").inner_text()
    page.locator("#xp-trace-aggregation").select_option("total")
    page.locator("#xp-rhythm-visible").uncheck()
    assert page.locator(".xp-rhythm-note:not([hidden])").count() == 0
    assert page.locator(".xp-grid-rhythm:not([hidden])").count() == 0
    page.locator("#xp-rhythm-visible").check()
    page.locator('#xp-trace-panel .xp-panel-action').filter(has_text='Settings').click()
    page.locator('#xp-trace-rhythm-colours').uncheck()
    assert page.locator('#xp-trace').get_attribute('data-trace-colour') == page.locator('#xp-peer-one-trace').get_attribute('data-trace-colour')
    page.locator('#xp-trace-rhythm-colours').check()
    assert page.locator('#xp-trace').get_attribute('data-trace-colour') == palette['other']
    page.reload()
    page.wait_for_function("window.pyMicrogliaExplorer?.rhythmSchema && window.pyMicrogliaExplorer.data.results.rhythms.analysis_id")
    assert page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms.analysis_id") == result["analysis_id"]
    page.locator("#xp-recording-select").select_option("synthetic_two")
    page.wait_for_function("window.pyMicrogliaExplorer.data.stem==='synthetic_two' && window.pyMicrogliaExplorer.rhythmSchema")
    assert not page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms.analysis_id")
    assert page.locator(".xp-rhythm-note:not([hidden])").count() == 0
    page.locator("#xp-recording-select").select_option("synthetic_one")
    page.wait_for_function("window.pyMicrogliaExplorer.data.stem==='synthetic_one' && window.pyMicrogliaExplorer.data.results.rhythms.analysis_id")
    assert not errors, errors
    # Repeating unchanged settings reuses the completed result; no fits rerun.
    page.wait_for_function("!window.pyMicrogliaExplorer.$('run-rhythms').disabled")
    previous_job = page.evaluate("window.pyMicrogliaExplorer.rhythmJobId")
    page.locator("#xp-run-rhythms").click()
    page.wait_for_function("id => window.pyMicrogliaExplorer.rhythmJobId!==id && !window.pyMicrogliaExplorer.$('run-rhythms').disabled", arg=previous_job, timeout=30000)
    reused_job = page.request.get(url + '/api/jobs/' + page.evaluate("window.pyMicrogliaExplorer.rhythmJobId")).json()
    assert reused_job['reused'], reused_job
    assert page.evaluate("window.pyMicrogliaExplorer.data.results.rhythms.analysis_id") == result['analysis_id']
    assert page.locator('#xp-rhythm-progress').is_hidden()
    assert len(jobs) == 2
    print(json.dumps({"errors": errors, "analysis_jobs": len(jobs), "tested_rows": len(result["rows"]), "phase_hours": first["phase_hours"], "status": first["status"],
                      "progress_updates":len(progress_samples),"reused":reused_job['reused']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:62848")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pymicroglia-rhythm-ui-") as folder:
        root = Path(folder)
        fixture(root)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
            page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
            check(page, args.url, Path(args.run).resolve(), root)
            browser.close()


if __name__ == "__main__":
    main()
