"""Acceptance checks against the real server in the dedicated browser profile."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from playwright.sync_api import sync_playwright


def canvas_label_overlaps(page):
    """Measure rendered canvas labels, not the size of their outer panels."""
    return page.evaluate("""() => {
      const drawn=[],proto=CanvasRenderingContext2D.prototype,original=proto.fillText;
      proto.fillText=function(value,x,y,...rest){
        const m=this.measureText(String(value)),width=m.width;
        const left=x-(this.textAlign==='center'?width/2:this.textAlign==='right'||this.textAlign==='end'?width:0);
        drawn.push({id:this.canvas.id,text:String(value),left,right:left+width,
          top:y-(m.actualBoundingBoxAscent||12),bottom:y+(m.actualBoundingBoxDescent||3)});
        return original.call(this,value,x,y,...rest);
      };
      try{window.pyMicrogliaExplorer.draw();}finally{proto.fillText=original;}
      const pairs=[];
      for(let i=0;i<drawn.length;i++)for(let j=i+1;j<drawn.length;j++){
        const a=drawn[i],b=drawn[j];if(a.id!==b.id)continue;
        if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>2&&
           Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>2)
          pairs.push([a.id,a.text,b.text]);
      }
      return pairs;
    }""")


def exercise(page, base_url, fixture, output=None):
    checks, errors, jobs = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: jobs.append(request.url) if request.method == "POST" and any(x in request.url for x in ("/api/jobs", "/api/pipeline-jobs", "/api/refinements")) else None)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(base_url + "/?" + urlencode({"run": fixture["run"]}))
    page.evaluate("run=>{localStorage.removeItem('pm-exploration-session-v1');localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');localStorage.setItem('pm-run-folder',run)}", fixture["run"])
    page.reload()
    page.wait_for_function("window.pyMicrogliaExplorer?.data && window.pyMicrogliaExplorer.exact && window.pyMicrogliaExplorer.frameData && window.pyMicrogliaExplorer.history", timeout=60000)
    shell_ms = page.evaluate("performance.getEntriesByType('navigation')[0].domContentLoadedEventEnd")
    assert shell_ms < 1000, shell_ms
    checks.append(f"Workbench shell becomes available in {shell_ms:.0f} ms")
    assert page.locator("#exploration-host [data-plot-type]").count() == 9
    assert len(set(page.locator("#exploration-host [data-plot-type]").evaluate_all("els=>els.map(e=>e.dataset.plotType)"))) == 9
    assert not page.locator("#xp-show-trails").is_checked()
    checks.append("Nine unique real-data views autofill; trails start off")
    first_view = page.evaluate("""() => Object.fromEntries(['image','outline','trace','barcode'].map(name => {
      const el = document.querySelector('#xp-' + name + '-panel');
      const box = el.getBoundingClientRect();
      return [name, {top: box.top, bottom: box.bottom, height: box.height}];
    }))""")
    assert first_view["image"]["top"] < first_view["trace"]["top"] < first_view["barcode"]["top"]
    assert first_view["outline"]["top"] == first_view["image"]["top"]
    assert first_view["barcode"]["bottom"] <= 1000, first_view
    checks.append("Image, moving outline, trace and barcode fit together in a 1440 × 1000 desktop viewport")
    page.wait_for_function("window.pyMicrogliaExplorer.imageCache.size === window.pyMicrogliaExplorer.data.frames.length", timeout=60000)
    scrub = page.evaluate("""() => {
      const explorer=window.pyMicrogliaExplorer,canvas=document.querySelector('#xp-outline-history');
      const imageCanvas=document.querySelector('#xp-cell-image');
      const original=explorer.strokeContours,frames=[];
      let selected,imageOutline;
      explorer.strokeContours=function(ctx,item,transform,colour,width){
        if(ctx.canvas.id==='xp-outline-history'&&width===3.5)selected=item;
        if(ctx.canvas.id==='xp-cell-image'&&width===2.5)imageOutline=item;
        return original.call(this,ctx,item,transform,colour,width);
      };
      try{
        for(const frame of [53,54,56]){
          selected=undefined;imageOutline=undefined;explorer.select({identity:19,frame});
          frames.push({frame,exact:selected===explorer.history.observations[frame]?.observation,
            imageExact:JSON.stringify(imageOutline?.contours)===JSON.stringify(explorer.history.observations[frame]?.observation?.contours),
            image:canvas.toDataURL(),sourceImage:imageCanvas.toDataURL(),
            sourceFrame:explorer.imageMeta?.frame_index,hasPixels:!!explorer.image?.naturalWidth});
        }
      }finally{explorer.strokeContours=original;}
      return {dense:explorer.history.exact,count:explorer.history.observations.length,frames};
    }""")
    assert scrub["dense"] and scrub["count"] == page.evaluate("window.pyMicrogliaExplorer.data.frames.length")
    assert all(row["exact"] and row["imageExact"] and row["hasPixels"] and row["sourceFrame"] == row["frame"] for row in scrub["frames"]), scrub
    assert len({row["image"] for row in scrub["frames"]}) == 3
    assert len({row["sourceImage"] for row in scrub["frames"]}) == 3
    checks.append("Rapid slider changes draw exact source pixels and saved outlines in both linked views")
    source = page.evaluate("JSON.stringify(window.pyMicrogliaExplorer.data)")

    def image(id):
        return page.locator("#xp-" + id).evaluate("e=>e.toDataURL()")

    def choose(id, value):
        el = page.locator("#xp-" + id)
        el.evaluate("e=>{const d=e.closest('details');if(d)d.open=true;}")
        el.select_option(value)

    trace = image("trace")
    barcode = image("barcode")
    choose("barcode-metric", "area_px")
    assert image("trace") == trace
    assert image("barcode") != barcode
    assert page.locator("#xp-trace-metric").input_value() == "light"
    checks.append("Matrix changes metric in place without changing the trace")
    choose("barcode-order", "phase:integrated_density")
    choose("map-layer", "phase:integrated_density")
    phase_barcode, phase_map = image("barcode"), image("tissue-map")
    choose("clock-metric", "area_px")
    assert image("barcode") == phase_barcode and image("tissue-map") == phase_map
    checks.append("Phase metric choices are independent in clock, matrix and tissue map")
    choose("clock-metric", "integrated_density")
    choose("barcode-order", "identity")
    choose("map-layer", "identity")
    panel = page.locator('#xp-barcode-panel .xp-export')
    panel.evaluate("e=>e.open=true")
    options = panel.locator("select option").all_text_contents()
    selected = next(i for i, text in enumerate(options) if "Saved cell measurements across native frames" in text)
    panel.locator("select").select_option(str(selected))
    panel.locator("button").click()
    page.wait_for_selector("#action-dialog[open]")
    assert page.locator('#action-fields [name=run]').input_value() == fixture["run"]
    assert page.locator('#action-fields [name=stem]').input_value() == "synthetic_1"
    assert page.locator('#action-fields [name=view]').input_value() == "barcode"
    page.locator("#close-action").click()
    checks.append("Saved figure configuration opens with the selected recording and view")
    page.locator("#xp-cell-select").select_option("26")
    page.wait_for_function("window.pyMicrogliaExplorer.exact?.cell.identity===26")
    page.locator("#xp-frame-slider").evaluate("e=>{e.value=90;e.dispatchEvent(new Event('input'))}")
    page.wait_for_function("window.pyMicrogliaExplorer.frameData?.frame_index===90")
    page.locator("#xp-trace").evaluate("e=>{const r=e.getBoundingClientRect();e.dispatchEvent(new MouseEvent('click',{clientX:r.x+r.width*.4,clientY:r.y+r.height/2}));}")
    page.wait_for_function("window.pyMicrogliaExplorer.frameData && window.pyMicrogliaExplorer.state.frame!==90")
    assert page.evaluate("window.pyMicrogliaExplorer.state.identity") == 26
    checks.append("Cell selector, slider and trace click share one selection")
    page.locator("#xp-cell-select").select_option("19")
    page.locator("#xp-frame-slider").evaluate("e=>{e.value=75;e.dispatchEvent(new Event('input'))}")
    page.wait_for_function("window.pyMicrogliaExplorer.frameData?.frame_index===75 && window.pyMicrogliaExplorer.exact?.cell.identity===19")
    assert page.evaluate("window.pyMicrogliaExplorer.observation()===undefined")
    assert page.evaluate("window.pyMicrogliaExplorer.exact.cell.observed[75]") is False
    assert "no observed mask" in page.locator("#xp-selection-status").inner_text()
    assert "no observed outline" in page.locator("#xp-history-current").inner_text()
    checks.append("A missing saved cell stays absent in image, geometry and trace")
    assert page.evaluate("JSON.stringify(window.pyMicrogliaExplorer.data)") == source
    checks.append("Local settings and selection leave source data unchanged")
    # Force response reordering to verify that a slow old frame cannot repaint.
    page.evaluate("""() => {const e=window.pyMicrogliaExplorer;const original=e.options.api;e.options.api=async(...args)=>{if(args[0].includes('frame=10'))await new Promise(r=>setTimeout(r,500));return original(...args);};e.select({frame:10});}""")
    page.wait_for_timeout(100)
    page.evaluate("window.pyMicrogliaExplorer.select({frame:110})")
    page.wait_for_function("window.pyMicrogliaExplorer.frameData?.frame_index===110")
    page.wait_for_timeout(650)
    assert page.evaluate("window.pyMicrogliaExplorer.frameData.frame_index") == 110
    checks.append("Late image responses cannot paint an obsolete frame")
    rotation = page.evaluate("JSON.stringify(window.pyMicrogliaExplorer.rotation)")
    outline_before_rotation = image("outline-history")
    canvas = page.locator("#xp-outline-history")
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down(); page.mouse.move(box["x"] + box["width"] / 2 + 70, box["y"] + box["height"] / 2 + 35, steps=5);page.mouse.up()
    assert page.evaluate("JSON.stringify(window.pyMicrogliaExplorer.rotation)") != rotation
    assert image("outline-history") != outline_before_rotation
    checks.append("Saved outline history rotates interactively")
    page.locator("[data-page=pipelines]").click()
    assert page.locator("#exploration-host").is_hidden()
    assert page.locator("#micro-content .card").count() > 0
    page.locator("[data-page=follow]").click()
    assert page.evaluate("window.pyMicrogliaExplorer.state.frame") == 110
    assert page.locator("#xp-barcode-metric").input_value() == "area_px"
    checks.append("On-demand pages preserve the mounted exploration and local settings")
    page.locator("[data-workspace=auto]").click()
    assert page.locator("#auto-frame").is_visible()
    assert page.locator("#micro-sidebar").is_hidden()
    assert page.locator("#main").evaluate("e=>getComputedStyle(e).gridColumnStart") == "1"
    page.locator("[data-workspace=motion]").click()
    assert page.locator("#motion-sidebar").is_visible()
    assert page.locator("[data-motion-page]").count() == 5
    page.locator("[data-workspace=micro]").click()
    assert page.evaluate("window.pyMicrogliaExplorer.state.frame") == 110
    checks.append("Accepted full-window Auto and second-tab Motion workspaces remain usable")
    page.locator("#xp-recording-select").select_option("synthetic_2")
    page.wait_for_function("window.pyMicrogliaExplorer.data?.stem==='synthetic_2' && window.pyMicrogliaExplorer.frameData?.stem==='synthetic_2'")
    assert page.evaluate("window.pyMicrogliaExplorer.state.stem") == "synthetic_2"
    checks.append("Recording switches replace data, sources and selection scope")
    page.locator("#xp-recording-select").select_option("synthetic_1")
    page.wait_for_function("window.pyMicrogliaExplorer.frameData?.stem==='synthetic_1'")
    page.evaluate("window.pyMicrogliaExplorer.select({identity:19,frame:48})")
    page.wait_for_function("window.pyMicrogliaExplorer.frameData?.frame_index===48 && window.pyMicrogliaExplorer.history?.identity===19 && window.pyMicrogliaExplorer.exact?.cell.identity===19")
    assert page.locator(".xp-edit-plot").count() == 9
    assert page.locator("#xp-control-help dt").count() >= 40
    assert "shared selected time" in page.locator("#xp-frame-slider").get_attribute("title")
    assert page.locator("#exploration-host .hint").count() == 0
    checks.append("Stable control help feeds tooltips and a collapsed manual")
    page.locator("#xp-trace-panel .xp-edit-plot").click()
    page.locator("#xp-edit-trace [data-style=height]").fill("320")
    page.locator("#xp-edit-trace [data-style=accent]").fill("#e6ad35")
    assert page.locator("#xp-trace").evaluate("e=>e.height") == 320
    assert page.locator("#xp-barcode").evaluate("e=>e.height") == 165
    page.locator("#xp-undo-trace").click()
    assert page.locator("#xp-edit-trace [data-style=accent]").input_value() == "#72ded3"
    page.locator("#xp-undo-trace").click()
    assert page.locator("#xp-trace").evaluate("e=>e.height") == 190
    page.locator("#xp-edit-trace [data-style=height]").fill("320")
    page.locator("#xp-edit-trace [data-style=accent]").fill("#e6ad35")
    checks.append("One shared edit control per plot supports independent edits and multi-level undo")
    source_image = image("cell-image")
    page.locator("#micro-theme").select_option("light")
    assert page.locator("html").get_attribute("data-pm-theme") == "light"
    assert image("cell-image") == source_image
    def contrast_pairs(pairs):
        return page.evaluate("""pairs => {const rgb=s=>[...s.matchAll(/[\\d.]+/g)].slice(0,3).map(m=>Number(m[0])/255);const lum=s=>rgb(s).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0);return pairs.map(([text,bg])=>{const a=lum(getComputedStyle(document.querySelector(text)).color),b=lum(getComputedStyle(document.querySelector(bg)).backgroundColor);return (Math.max(a,b)+.05)/(Math.min(a,b)+.05);});}""", pairs)
    contrast = contrast_pairs([("#micro-sidebar .eyebrow", "#micro-sidebar"), ("#micro-workspace .eyebrow", "html"), ("#xp-image-panel h3", "#xp-image-panel"), ("#xp-image-panel .eyebrow", "#xp-image-panel")])
    assert min(contrast) >= 4.5, contrast
    page.locator("[data-page=pipelines]").click()
    pipeline_contrast = contrast_pairs([("#micro-content .card .button.ghost", "#micro-content .card"), ("#micro-content .section-heading small", "html")])
    assert min(pipeline_contrast) >= 4.5, pipeline_contrast
    page.locator("#micro-content .card .button.ghost").first.click()
    page.wait_for_selector("#action-dialog[open]")
    dialog_contrast = contrast_pairs([("#action-dialog .eyebrow", "#action-dialog"), ("#action-dialog .field", "#action-dialog")])
    assert min(dialog_contrast) >= 4.5, dialog_contrast
    page.locator("#close-action").click()
    page.locator("[data-page=follow]").click()
    page.reload()
    page.wait_for_function("window.pyMicrogliaExplorer?.frameData?.frame_index===48 && window.pyMicrogliaExplorer.exact?.cell.identity===19", timeout=60000)
    assert page.locator("#xp-trace").evaluate("e=>e.height") == 320
    assert page.locator("#xp-edit-trace").get_attribute("open") is not None
    assert page.locator("#micro-theme").input_value() == "light"
    checks.append("Selected recording, cell, time, plot edit and appearance survive reload; source image stays unchanged")
    draw_ms = page.evaluate("() => {const e=window.pyMicrogliaExplorer,t=performance.now();for(let i=0;i<3;i++)e.draw();return (performance.now()-t)/3;}")
    assert draw_ms < 500, draw_ms
    checks.append(f"All nine linked plots redraw in {draw_ms:.0f} ms on average")
    assert not canvas_label_overlaps(page)
    checks.append("Measured canvas text boxes have no overlaps")
    page.locator("#micro-theme").select_option("dark")
    checks.append("Light theme text contrast passes 4.5:1 in linked views, Pipelines and its action dialog")
    if output:
        output.mkdir(parents=True, exist_ok=True)
        shot_page = page.context.new_page()
        shot_page.set_viewport_size({"width": 1440, "height": 1000})
        shot_page.goto(base_url + "/?" + urlencode({"run": fixture["run"]}))
        shot_page.evaluate("run=>{localStorage.removeItem('pm-exploration-session-v1');localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');localStorage.setItem('pm-run-folder',run)}", fixture["run"])
        shot_page.reload()
        shot_page.wait_for_function("window.pyMicrogliaExplorer?.frameData && window.pyMicrogliaExplorer.history", timeout=60000)
        shot_page.evaluate("window.pyMicrogliaExplorer.select({identity:19,frame:53})")
        shot_page.wait_for_function("window.pyMicrogliaExplorer.frameData?.frame_index===53 && window.pyMicrogliaExplorer.exact?.cell.identity===19")
        shot_page.locator("#micro-workspace").evaluate("e=>e.scrollTop=0")
        assert shot_page.locator("#xp-barcode-panel").evaluate("e=>e.getBoundingClientRect().bottom<=1000")
        shot_page.locator("#micro-theme").select_option("light")
        shot_page.screenshot(path=str(output / "live-light.png"), timeout=30000, animations="disabled")
        shot_page.locator("#micro-theme").select_option("dark")
        shot_page.screenshot(path=str(output / "live-desktop.png"), timeout=30000, animations="disabled")
        shot_page.set_viewport_size({"width": 1050, "height": 900})
        assert shot_page.locator("#micro-workspace").evaluate("e=>e.scrollWidth<=e.clientWidth+1")
        assert not canvas_label_overlaps(shot_page)
        shot_page.screenshot(path=str(output / "live-narrow.png"), timeout=30000, animations="disabled")
        shot_page.set_viewport_size({"width": 1440, "height": 1000})
        shot_page.locator("#micro-workspace").evaluate("e=>e.scrollTop=0")
        shot_page.bring_to_front()
    checks.append("Extras controls and panels fit desktop and narrower windows")
    assert not jobs, jobs
    assert not errors, errors
    for path, expected in fixture["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    checks.append("No analysis jobs submitted, JavaScript errors or source pixel/identity changes")
    result = {"passed": True, "checks": checks, "performance_ms": {"shell_dom_ready": round(shell_ms, 1), "nine_view_redraw": round(draw_ms, 1)}, "javascript_errors": errors, "implicit_jobs": jobs, "run": fixture["run"], "url": base_url}
    if output:
        (output / "browser-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify(base_url, folder):
    fixture = json.loads((folder / "fixture.json").read_text())
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://127.0.0.1:9333")
        page = next((page for page in browser.contexts[0].pages if page.url.startswith(base_url)), None)
        page = page or browser.contexts[0].new_page()
        result = exercise(page, base_url, fixture, folder)
        page.bring_to_front()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("url");parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.folder), indent=2))
