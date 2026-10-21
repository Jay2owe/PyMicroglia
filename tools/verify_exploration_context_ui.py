"""Check native context views and significance colours without running analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


def check_gantt_density(page):
    """Read rendered bars: a compact row must not have blank space above/below it."""
    density = page.evaluate("""() => {
        const e=window.pyMicrogliaExplorer,c=e.$('events'),p=e.hits.events,d=devicePixelRatio,
            swatch=document.createElement('canvas').getContext('2d');
        swatch.fillStyle=e.style('events').accent;swatch.fillRect(0,0,1,1);
        const colour=[...swatch.getImageData(0,0,1,1).data].slice(0,3),rows=[];
        for(const [index,cell] of p.cells.slice(0,6).entries()){
            const segment=cell.segments.reduce((a,b)=>b[1]-b[0]>a[1]-a[0]?b:a),
                frame=Math.round(segment[0]+.3*(segment[1]-segment[0])),
                x=Math.round((p.left+e.fraction(frame)*(p.right-p.left))*d),
                y=Math.ceil((p.top+index*p.rowHeight)*d)+1,
                height=Math.floor((p.top+(index+1)*p.rowHeight)*d)-y-1,
                pixels=c.getContext('2d').getImageData(x,y,1,height).data;
            let filled=0;for(let i=0;i<pixels.length;i+=4){if(colour.every((v,j)=>pixels[i+j]===v))filled++;}
            rows.push({identity:cell.identity,filled_fraction:filled/height});
        }
        return {row_height:p.rowHeight,label_size:Math.min(12,e.style('events').fontSize),rows};
    }""")
    assert density["row_height"] <= density["label_size"] + 1.01, density
    assert all(row["filled_fraction"] >= .99 for row in density["rows"]), density
    return density


def check(page, url, run, output):
    errors, writes, failed = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: writes.append(request.url)
            if request.method == "POST" and "/api/results/select" not in request.url else None)
    page.on("response", lambda response: failed.append([response.status, response.url]) if response.status >= 400 else None)
    page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
    page.goto(url + "/?" + urlencode({"run": str(run)}))
    page.wait_for_function("window.pyMicrogliaExplorer?.timeline && [...window.pyMicrogliaExplorer.tissueCache.values()].some(e=>e.image)", timeout=30000)
    assert page.locator("[data-plot-type]").count() == 9
    assert page.locator("#xp-event-mode").input_value() == "gantt"
    assert page.locator("#xp-map-layer").input_value() == "tectonics:first_coverage"
    counts = page.evaluate("({tracks:window.pyMicrogliaExplorer.timeline.cells.length,identities:window.pyMicrogliaExplorer.data.identities.length})")
    assert counts["tracks"] == counts["identities"]
    density = check_gantt_density(page)
    assert page.locator("#xp-timeline-axis").inner_text().startswith("1 h")
    reasons = page.evaluate("Object.keys(window.pyMicrogliaExplorer.timeline.gap_legend)")
    assert "same_host_merge_hiding" in reasons
    assert "local_raw_supported_dropout" in reasons
    assert page.locator('#xp-timeline-legend [data-reason="same_host_merge_hiding"]').is_visible()
    gap_selection = page.evaluate("""() => {
        const e=window.pyMicrogliaExplorer,cell=e.timeline.cells.find(c=>c.gaps.some(g=>g.reason==='same_host_merge_hiding')),
            gap=cell.gaps.find(g=>g.reason==='same_host_merge_hiding'),row=e.timeline.cells.indexOf(cell),plot=e.hits.events,
            c=e.$('events'),r=c.getBoundingClientRect(),frame=gap.start_frame,
            x=plot.left+e.fraction(frame)*(plot.right-plot.left),y=plot.top+(row+.5)*plot.rowHeight;
        c.dispatchEvent(new MouseEvent('mousemove',{clientX:r.left+x,clientY:r.top+y,bubbles:true}));
        return {identity:cell.identity,frame,reason:gap.reason};
    }""")
    detail = page.locator("#xp-timeline-detail")
    assert "cell hidden inside the same merged object" in detail.inner_text()
    assert int(detail.get_attribute("data-identity")) == gap_selection["identity"]
    assert int(detail.get_attribute("data-frame")) == gap_selection["frame"]
    assert detail.get_attribute("data-reason") == gap_selection["reason"]

    # Read the rendered row marker, rather than accepting only a style flag.
    def colours():
        return page.evaluate("""() => {
            const e=window.pyMicrogliaExplorer,c=e.$('barcode'),p=e.hits.barcode,
                key=window.PyMicrogliaExplorationCore.metricKey(e.val('barcode-metric'),e.val('barcode-aggregation'),e.data),
                d=window.devicePixelRatio;
            return p.cells.map((cell,row)=>({identity:cell.identity,status:e.rhythmForMetric(cell.identity,key)?.status,
                rgb:[...c.getContext('2d').getImageData(Math.round((p.left-4)*d),Math.round((p.top+(row+.5)*p.rowHeight)*d),1,1).data].slice(0,3)}));
        }""")

    page.locator("#xp-barcode-aggregation").select_option("total")
    total = colours()
    assert any(row["status"] == "rhythmic" for row in total)
    for row in total:
        assert row["rgb"] == ([192, 57, 43] if row["status"] == "rhythmic" else [122, 122, 122]), row
    page.locator("#xp-barcode-aggregation").select_option("mean")
    mean = colours()
    assert any(a["status"] != b["status"] for a, b in zip(total, mean))
    for row in mean:
        assert row["rgb"] == ([192, 57, 43] if row["status"] == "rhythmic" else [122, 122, 122]), row
    page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.$('barcode-rhythm-colours').checked=false;e.drawBarcode();e.saveSession();}""")
    assert page.locator("#xp-barcode").get_attribute("data-rhythm-colours") == "false"
    assert len({tuple(row["rgb"]) for row in colours()}) == 1
    page.reload()
    page.wait_for_function("window.pyMicrogliaExplorer?.timeline", timeout=30000)
    assert not page.locator("#xp-barcode-rhythm-colours").is_checked()
    page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.$('barcode-rhythm-colours').checked=true;e.$('rhythm-visible').checked=false;e.drawBarcode();}""")
    assert page.locator("#xp-barcode").get_attribute("data-rhythm-colours") == "false"
    page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.$('rhythm-visible').checked=true;e.draw();}""")
    assert page.locator("#xp-barcode").get_attribute("data-rhythm-colours") == "true"
    page.locator("#xp-barcode-aggregation").select_option("total")
    page.locator("#xp-barcode-panel").scroll_into_view_if_needed()
    page.screenshot(path=str(output / "barcode.png"))

    # The all-cell chart shares both cell and time selection with the follow view.
    page.evaluate("""() => {const e=window.pyMicrogliaExplorer,c=e.$('events'),p=e.hits.events,r=c.getBoundingClientRect();
        c.dispatchEvent(new MouseEvent('click',{clientX:r.left+p.left+.5*(p.right-p.left),clientY:r.top+p.top+1.5*p.rowHeight,bubbles:true}));} """)
    selected = page.evaluate("window.pyMicrogliaExplorer.state")
    assert selected["identity"] == page.evaluate("window.pyMicrogliaExplorer.timeline.cells[1].identity")
    assert abs(selected["frame"] - page.evaluate("window.pyMicrogliaExplorer.data.frames.length/2")) <= 1
    page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.$('timeline-order').value='identity';e.$('timeline-order').dispatchEvent(new Event('change'));}""")
    page.wait_for_function("JSON.parse(window.pyMicrogliaExplorer.timelineKey).at(-1)==='identity' && window.pyMicrogliaExplorer.timeline.cells.every((c,i,a)=>!i||a[i-1].identity<c.identity)")
    page.locator("#xp-event-mode").select_option("observations")
    assert page.evaluate("!window.pyMicrogliaExplorer.hits.events.cells")
    assert page.locator("#xp-timeline-axis").is_hidden()
    page.locator("#xp-event-mode").select_option("gantt")
    assert page.evaluate("window.pyMicrogliaExplorer.hits.events.cells.length") == counts["tracks"]

    layers = ["first_coverage", "cumulative_occupancy", "unique_cells", "speed", "significant_period", "splitting_events"]
    for layer in layers:
        page.locator("#xp-map-layer").select_option("tectonics:" + layer)
        page.wait_for_function("() => {const e=window.pyMicrogliaExplorer;return !!e.tissueCache.get(e.tissueKey())?.image;}", timeout=30000)
        payload = page.evaluate("() => {const e=window.pyMicrogliaExplorer;const p=e.tissueCache.get(e.tissueKey()).payload;return {layer:p.layer,message:p.message,limits:p.limits};}")
        assert payload["layer"] == layer
        if layer in ("speed", "splitting_events"):
            assert "No saved" in payload["message"]  # this fixture has no recorded events
        else:
            assert payload["message"] == "", payload
    page.locator("#xp-map-layer").select_option("tectonics:significant_period")
    page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.$('tectonics-metric').value='signal_mean';e.$('tectonics-metric').dispatchEvent(new Event('change'));}""")
    page.wait_for_function("() => {const e=window.pyMicrogliaExplorer;return e.tissueCache.get(e.tissueKey())?.image && e.tissueCache.get(e.tissueKey()).payload.metric==='signal_mean';}")
    assert page.evaluate("() => {const e=window.pyMicrogliaExplorer;return e.tissueCache.get(e.tissueKey()).payload.limits;}") == [2, 48]

    for theme in ("light", "dark"):
        page.locator("#micro-theme").select_option(theme)
        for width in (1440, 800):
            page.set_viewport_size({"width": width, "height": 1000})
            page.wait_for_function("window.pyMicrogliaExplorer.$('tissue-map').width===Math.round(window.pyMicrogliaExplorer.$('tissue-map').getBoundingClientRect().width*devicePixelRatio)")
            assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
            for name in ("barcode", "events", "tissue-map"):
                assert page.evaluate("id=>{const c=document.getElementById(id),r=c.getBoundingClientRect();return Math.abs(c.width-r.width*devicePixelRatio)<=1 && Math.abs(c.height-r.height*devicePixelRatio)<=1;}", "xp-" + name)
            check_gantt_density(page)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.locator("#micro-theme").select_option("light")
    page.locator("#xp-tissue").scroll_into_view_if_needed()
    page.screenshot(path=str(output / "tissue-and-gantt.png"))
    assert not errors, errors
    assert not failed, failed
    assert not writes, writes
    return {"tracks": counts["tracks"], "gantt_density": density, "gap_reasons": reasons, "gap_hover": gap_selection,
            "tissue_layers": layers, "significant_total": sum(r["status"] == "rhythmic" for r in total),
            "significant_mean": sum(r["status"] == "rhythmic" for r in mean), "errors": errors, "analysis_requests": writes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:62855")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    source = run / "measure/95_A3/cell_frame.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path(tempfile.mkdtemp(prefix="pymicroglia-context-ui-"))
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        try:
            receipt = check(page, args.url, run, output)
        finally:
            browser.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    receipt.update(source_sha256=digest, source_unchanged=True, screenshots=str(output))
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
