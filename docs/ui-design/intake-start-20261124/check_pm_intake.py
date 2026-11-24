"""Drive PyMicroglia's "Analyse recordings" page in headless Edge.

Usage: check_pm_intake.py OUT_DIR [before|after]

Synthetic stacks, isolated Auto-Organotypic stores. The workbench server runs
in this process with ``auto_microglia.run`` replaced by a stand-in that records
the keywords it was given and stops when asked, so no analysis actually runs.
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
phase = sys.argv[2] if len(sys.argv) > 2 else 'after'
scratch = Path(tempfile.mkdtemp(prefix='pm-intakecheck-'))
for name in ('AUTO_ORGANOTYPIC_STORE', 'AUTO_ORGANOTYPIC_INDEX', 'AUTO_ORGANOTYPIC_DECISIONS'):
    os.environ[name] = str(scratch / name.lower())
os.environ['AUTO_ORGANOTYPIC_WARM_TRACE_PREVIEW'] = '0'

import tifffile
from auto_organotypic import demo
demo.FRAMES = 10
data = scratch / 'Microglia 2026-11'
data.mkdir()
for i in range(3):
    pixels, _ = demo.generate_stack(i)
    meta = {'axes': 'TCYX', 'Channel': {'Name': list(demo.CHANNEL_NAMES)}}
    if i < 2:
        meta.update(TimeIncrement=1800., TimeIncrementUnit='s')
    tifffile.imwrite(data / f'well_{i + 1}.ome.tif', pixels, ome=True, metadata=meta)

from pymicroglia.pipelines import auto_microglia
calls = []


def stand_in(**kwargs):
    progress = kwargs.pop('on_progress', None)
    calls.append({k: v for k, v in kwargs.items()})
    for _ in range(240):
        time.sleep(0.25)
        progress({'stage': 'register', 'status': 'ok'})   # raises once Stop is pressed
    return {'folder': ''}


import functools
auto_microglia.run = functools.wraps(auto_microglia.run)(stand_in)   # keeps the real keyword defaults

import uvicorn
from playwright.sync_api import sync_playwright
from pymicroglia.ui.server import create_app

port = 8893
server = uvicorn.Server(uvicorn.Config(create_app(), host='127.0.0.1', port=port, log_level='warning'))
thread = threading.Thread(target=server.run, daemon=True); thread.start()
url = f'http://127.0.0.1:{port}'
for _ in range(100):
    if server.started: break
    time.sleep(0.1)

report = {'phase': phase}
errors = []
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        ctx = browser.new_context(viewport={'width': 1440, 'height': 900})
        ctx.add_init_script("try{localStorage.setItem('pm-extras-theme','light');localStorage.setItem('pm-workspace','micro')}catch(e){}")
        page = ctx.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('response', lambda r: errors.append(f'{r.status}: {r.url}') if r.status >= 400 else None)
        page.goto(url + '/'); page.wait_for_selector('#intake-host .intake-drop', timeout=30000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(out / '01-empty.png'))
        page.fill('#intake-path', str(data)); page.click('[data-intake-form] button[type=submit]')
        page.wait_for_selector('.intake-table', timeout=20000); page.wait_for_timeout(500)
        report['rows'] = page.locator('.intake-table tbody tr').count()
        report['button'] = page.locator('[data-intake=run]').inner_text()
        report['beside_button'] = page.locator('.intake-actions small').inner_text()
        report['signal_preselected'] = page.locator('[data-intake-role]').evaluate_all('els => els.map(e => e.value)')
        page.screenshot(path=str(out / '02-read.png'), full_page=True)
        if phase == 'after':
            report['run_disabled_before_signal'] = page.locator('[data-intake=run]').is_disabled()
            report['hint_before_signal'] = page.locator('.intake-actions small').inner_text()
            page.select_option('[data-intake-signal-all]', '2'); page.wait_for_timeout(200)
            report['signal_after_choice'] = page.locator('[data-intake-role]').evaluate_all('els => els.map(e => e.value)')
            page.select_option('[data-intake-outline]', '3'); page.wait_for_timeout(200)
            report['plan_with_outline'] = page.locator('.intake-actions small').inner_text()
            page.click('.intake-options summary'); page.wait_for_timeout(200)
            page.uncheck('[data-intake-setting=tracked_video]'); page.wait_for_timeout(200)
            page.uncheck('[data-intake-setting=motion]'); page.wait_for_timeout(200)
            report['plan_without_tracking'] = page.locator('.intake-actions small').inner_text()
            page.check('[data-intake-setting=motion]'); page.wait_for_timeout(200)
            report['plan_final'] = page.locator('.intake-actions small').inner_text()
            page.screenshot(path=str(out / '03-options.png'), full_page=True)
        # Interval for the third stack, then a label, then run.
        page.fill('.intake-interval', '1800'); page.locator('.intake-interval').dispatch_event('change')
        page.fill('[data-intake-label=condition][data-intake-key="well_1.ome.tif"]', 'WT')
        page.wait_for_timeout(900)
        page.set_viewport_size({'width': 800, 'height': 900}); page.wait_for_timeout(300)
        report['narrow_overflow'] = page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(out / '04-narrow.png'), full_page=True)
        page.set_viewport_size({'width': 1440, 'height': 900})
        page.evaluate("document.documentElement.dataset.pmTheme='dark'"); page.wait_for_timeout(300)
        page.screenshot(path=str(out / '05-dark.png'), full_page=True)
        page.evaluate("document.documentElement.dataset.pmTheme='light'")
        page.click('[data-intake=run]')
        page.wait_for_selector('.intake-status.running', timeout=30000); page.wait_for_timeout(800)
        page.screenshot(path=str(out / '06-running.png'))
        report['run_keywords'] = {k: v for k, v in (calls[-1] if calls else {}).items() if k != 'claim'}
        page.click('[data-intake=stop]')
        page.wait_for_selector('.intake-status.failed', timeout=30000)
        report['after_stop'] = page.locator('.intake-status').inner_text()
        manifest = data / 'pymicroglia_analysis'
        report['manifest_written'] = manifest.is_dir()
        report['manifest_signals'] = sorted({m for f in manifest.rglob('*.json') for m in __import__('re').findall(r'"signal": *(\w+)', f.read_text(encoding='utf-8'))})
        # Desktop window version, with stand-in native pickers (no real window).
        desk = ctx.new_page()
        desk.on('pageerror', lambda e: errors.append('desktop: ' + str(e)))
        picks = {'folder': str(data), 'files': [str(data / 'well_1.ome.tif')]}
        desk.add_init_script(f"window.__picks={json.dumps(picks)};window.pywebview={{api:{{choose_folder:async()=>window.__picks.folder,choose_recordings:async()=>window.__picks.files}}}};")
        desk.goto(url + '/'); desk.wait_for_selector('#intake-host .intake-drop'); desk.wait_for_timeout(500)
        desk.evaluate("window.dispatchEvent(new Event('pywebviewready'))"); desk.wait_for_timeout(300)
        report['desktop_buttons'] = desk.locator('.intake-drop button').all_inner_texts()
        desk.click('[data-intake=folder]'); desk.wait_for_selector('.intake-table', timeout=30000)
        report['desktop_folder_rows'] = desk.locator('.intake-table tbody tr').count()
        desk.screenshot(path=str(out / '07-desktop.png'))
        browser.close()
except Exception as exc:
    report['failed'] = repr(exc)[:600]
    try:
        report['failure_text'] = page.locator('#intake-host').inner_text()[:800]
        page.screenshot(path=str(out / 'failure.png'))
    except Exception as again: report['failure_capture'] = repr(again)[:200]
finally:
    server.should_exit = True
    thread.join(timeout=15)
report['page_errors'] = errors
(out / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, indent=2, default=str))
