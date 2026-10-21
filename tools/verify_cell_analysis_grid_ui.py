"""Exercise the real cell-grid action dialog and preview real vector plots in-browser.

Screenshots are UI checks; no scientific figure files or experimental data are
created. The server and browser are disposed on completion.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from playwright.sync_api import sync_playwright
import uvicorn

from pymicroglia.ui.server import create_app
from pymicroglia import workbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--fixture', type=Path, help='Existing synthetic exploration fixture for completed-plot timings')
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='cell-analysis-grid-ui-'))
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    errors = []
    jobs = []
    report = {}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, channel='chrome')
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: jobs.append(request.url) if request.method == 'POST' and '/api/jobs' in request.url else None)
            responses = []
            page.on('response', lambda response: responses.append(response.url) if '/api/actions/cell_grid' in response.url else None)
            page.add_init_script("localStorage.setItem('pm-workspace','micro'); localStorage.setItem('pm-page','all_methods');")
            started = perf_counter()
            # The request retries only while this newly started server binds.
            import urllib.request
            for _ in range(100):
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{port}/api/catalogue', timeout=.1)
                    break
                except Exception:
                    threading.Event().wait(.1)
            page.goto(f'http://127.0.0.1:{port}')
            card = page.locator('article.card').filter(has=page.get_by_role('heading', name='Cell grid', exact=True))
            assert card.count() == 1
            assert page.get_by_role('heading', name='Cell analysis grid', exact=True).count() == 0
            assert page.get_by_role('heading', name='All measured cell traces', exact=True).count() == 0
            report['all_methods_ready_seconds'] = round(perf_counter() - started, 3)
            started_dialog = perf_counter()
            card.get_by_role('button').click()
            selector = page.locator('#action-dialog [name="analysis"]')
            selector.wait_for(state='visible')
            report['first_dialog_seconds'] = round(perf_counter() - started, 3)
            report['ready_card_to_dialog_seconds'] = round(perf_counter() - started_dialog, 3)
            report['cell_grid_description_requests'] = len(responses)
            assert selector.input_value() == 'trace'
            assert page.locator('[data-grid-group="preprocessing"][data-grid-key="mean_frames"]').input_value() == '3'
            assert page.locator('[data-grid-group="preprocessing"][data-grid-key="detrend"]').input_value() == 'robust_linear'
            assert not page.locator('#action-dialog [name="reference_metric"]').is_visible()
            arrangement = page.locator('#action-dialog details').filter(has=page.locator('summary').filter(has_text='Grid arrangement and overlays'))
            arrangement.evaluate('(e)=>{e.open=true}')
            layout = page.locator('[data-grid-group="grid_presentation"][data-grid-key="layout"]')
            layout.select_option('stack')
            layers = page.locator('[data-grid-group="grid_presentation"][data-grid-key="trace_layers"]')
            components = page.locator('[data-grid-group="grid_presentation"][data-grid-key="fft_components"]')
            assert components.is_visible()
            components.check()
            layers.locator('..').locator('input[type="checkbox"][value="raw"]').check()
            layers.locator('..').locator('input[type="checkbox"][value="mean"]').check()
            assert json.loads(layers.input_value()) == ['processed', 'raw', 'mean']
            page.locator('[data-grid-group="grid_presentation"][data-grid-key="alpha"]').fill('0.55')
            page.locator('[data-grid-group="grid_presentation"][data-grid-key="line_width"]').fill('1.7')
            page.locator('#action-dialog [name="run"]').fill('unused')
            with page.expect_request('**/api/validate') as trace_request:
                page.locator('#validate-action').click()
            trace_payload = trace_request.value.post_data_json['params']['grid_presentation']
            assert trace_payload == {'fft_components': True, 'layout': 'stack', 'trace_layers': ['processed', 'raw', 'mean'], 'show_fit': True, 'legend': True, 'line_width': 1.7, 'alpha': 0.55}, trace_payload
            page.wait_for_function("document.querySelector('#action-message').textContent === 'Settings accepted.'")
            page.locator('#action-dialog').evaluate('(e)=>{e.scrollTop=0}')
            page.screenshot(path=str(output / 'trace-composition-controls.png'))
            page.locator('#action-dialog details.advanced').evaluate_all('(elements)=>elements.forEach(e=>e.open=true)')
            assert page.locator('[data-grid-key="period_testing"]').is_visible()
            selector.select_option('wavelet_spectrum/wavelet')
            assert components.is_disabled() and not components.is_visible()
            assert layers.is_disabled() and not layers.locator('..').is_visible()
            wavelet = page.locator('[data-grid-key="wavelet_options"]')
            assert wavelet.is_visible()
            wavelet.fill('{"n_sim":0,"seed":17}')
            selector.select_option('wavelet_spectrum/wavelet_phase')
            assert page.locator('[data-grid-key="wavelet_options"]').input_value() == '{"n_sim":0,"seed":17}'
            assert page.locator('[data-grid-key="period_testing"]').count() == 0
            started = perf_counter()
            selector.select_option('wavelet_coherence/wavelet_coherence')
            assert page.locator('#action-dialog [name="reference_metric"]').is_visible()
            assert page.locator('#action-dialog [name="reference_metric"]').get_attribute('required') is not None
            report['paired_selection_seconds'] = round(perf_counter() - started, 3)
            page.locator('#action-dialog details.advanced').evaluate_all('(elements)=>elements.forEach(e=>e.open=true)')
            help_text = page.locator('#action-dialog [name="analysis_parameters"]').locator('..').locator('small').inner_text()
            assert 'wavelet_options' in help_text
            selector.select_option('autocorrelation/autocorrelation')
            assert not page.locator('#action-dialog [name="reference_metric"]').is_visible()
            assert page.locator('[data-grid-key="wavelet_options"]').count() == 0
            selector.select_option('wavelet_spectrum/wavelet')
            assert page.locator('[data-grid-key="wavelet_options"]').input_value() == '{"n_sim":0,"seed":17}'
            page.locator('#action-dialog [name="run"]').fill('unused')
            with page.expect_request('**/api/validate') as request:
                page.locator('#validate-action').click()
            submitted = request.value.post_data_json['params']
            assert submitted['analysis_parameters'] == {'wavelet_options': {'n_sim': 0, 'seed': 17}}, submitted
            assert submitted['preprocessing'] == {'mean_frames': 3, 'detrend': 'robust_linear', 'detrend_options': {}}, submitted
            assert submitted['grid_presentation'] == {'layout': 'stack'}, submitted
            page.wait_for_function("document.querySelector('#action-message').textContent === 'Settings accepted.'")
            report['typed_payload_verified'] = True
            selector.select_option('trace')
            assert json.loads(layers.input_value()) == ['processed', 'raw', 'mean']
            assert components.is_checked()
            page.reload()
            card.get_by_role('button').click()
            page.locator('#action-dialog details').filter(has=page.locator('summary').filter(has_text='Grid arrangement and overlays')).evaluate('(e)=>{e.open=true}')
            assert layout.input_value() == 'stack'
            assert json.loads(layers.input_value()) == ['processed', 'raw', 'mean']
            assert page.locator('[data-grid-group="grid_presentation"][data-grid-key="alpha"]').input_value() == '0.55'
            assert components.is_checked()
            selector.select_option('compare_periods/fft_nlls_components')
            assert components.is_visible() and not components.is_disabled()
            page.locator('#action-dialog [name="run"]').fill('unused')
            with page.expect_request('**/api/validate') as fft_request:
                page.locator('#validate-action').click()
            assert fft_request.value.post_data_json['params']['grid_presentation'] == {'fft_components': True, 'layout': 'stack'}
            page.wait_for_function("document.querySelector('#action-message').textContent === 'Settings accepted.'")
            report['fft_components_typed_payload_and_persistence_verified'] = True
            selector.select_option('wavelet_spectrum/wavelet')
            page.locator('#action-dialog details.advanced').evaluate_all('(elements)=>elements.forEach(e=>e.open=true)')
            views = page.locator('#action-dialog [name="analysis_views"]')
            for key in ('wavelet_spectrum/wavelet', 'wavelet_spectrum/wavelet_phase', 'wavelet_spectrum/wavelet_amplitude'):
                views.locator('..').locator(f'input[type="checkbox"][value="{key}"]').check()
            assert len(json.loads(views.input_value())) == 3
            paired_view = views.locator('..').locator('input[type="checkbox"][value="wavelet_coherence/wavelet_coherence"]')
            paired_view.check()
            assert page.locator('#action-dialog [name="reference_metric"]').is_visible()
            paired_view.uncheck()
            assert not page.locator('#action-dialog [name="reference_metric"]').is_visible()
            report['composition_controls_and_persistence_verified'] = True
            assert not jobs, jobs
            report['unrequested_analysis_jobs'] = len(jobs)
            for width in (1440, 800):
                page.set_viewport_size({'width': width, 'height': 1000})
                assert selector.is_visible()
                assert page.locator('#action-dialog').evaluate('(e)=>e.scrollWidth <= e.clientWidth + 2')
                page.locator('#action-dialog').evaluate('(e)=>{e.scrollTop=0}')
                page.screenshot(path=str(output / f'dialog-{width}.png'))
            # Both interface themes use the existing application theme controls.
            for theme in ('light', 'dark'):
                page.evaluate("theme=>{const input=document.querySelector('#micro-theme');input.value=theme;input.dispatchEvent(new Event('change'));}", theme)
                assert page.locator('html').get_attribute('data-pm-theme') == theme
                assert page.locator('#action-dialog').evaluate('(e)=>e.scrollWidth <= e.clientWidth + 2')
                page.locator('#action-dialog').evaluate('(e)=>{e.scrollTop=0}')
                page.screenshot(path=str(output / f'dialog-{theme}.png'))
            report['themes_checked'] = ['light', 'dark']

            if args.fixture:
                fixture = json.loads(args.fixture.read_text(encoding='utf-8'))
                live = browser.new_page(viewport={'width': 1440, 'height': 1000})
                live.on('pageerror', lambda error: errors.append(str(error)))
                live.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');localStorage.removeItem('pm-exploration-session-v1');")
                from urllib.parse import urlencode
                started = perf_counter()
                live.goto(f'http://127.0.0.1:{port}/?' + urlencode({'run': fixture['run']}))
                ready = """stem=>{const e=window.pyMicrogliaExplorer;return e?.data?.stem===stem && e.exact && e.history && e.frameData && e.image?.naturalWidth>0 && document.querySelector('#xp-trace')?.width>0;}"""
                try:
                    live.wait_for_function(ready, arg='synthetic_1', timeout=30000)
                except Exception:
                    print(live.evaluate("()=>{const e=window.pyMicrogliaExplorer;return {stem:e?.data?.stem,exact:!!e?.exact,history:!!e?.history,frame:e?.frameData?.stem,image:e?.image?.naturalWidth,trace:document.querySelector('#xp-trace')?.width,errors:document.body.innerText.slice(-1600)};}"))
                    raise
                live.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                report['first_completed_headline_plots_seconds'] = round(perf_counter() - started, 3)
                started = perf_counter()
                live.locator('#xp-recording-select').evaluate("e=>{const option=[...e.options].find(o=>o.textContent==='synthetic_2');e.value=option.value;e.dispatchEvent(new Event('change'));}")
                live.wait_for_function(ready, arg='synthetic_2', timeout=60000)
                live.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                report['recording_switch_completed_plots_seconds'] = round(perf_counter() - started, 3)
                report['performance_fixture'] = str(args.fixture.resolve())
                live.close()

            # Real completed Workbench plots displayed in an HTML page. The
            # scientific SVG exists in memory only, not as an unaudited file.
            hours = np.arange(144.)
            values = 10 + np.sin(2 * np.pi * hours / 24)
            values[48:52] = np.nan
            definitions = []
            started = perf_counter()
            with workbench.analysis_session():
                for analysis, label in [('wavelet_spectrum/wavelet', 'Cell 3 | Wavelet power'),
                                        ('autocorrelation/autocorrelation', 'Cell 3 | Lag correlation')]:
                    completed = workbench.analyze_cell_view(hours, values, analysis=analysis,
                        parameters={'wavelet_options': {'n_sim': 0}} if analysis.startswith('wavelet') else {})
                    definitions.append((label, completed.definition))
            figure, _ = workbench.draw_cell_views(definitions, canvas=Figure(), rows=1, columns=2)
            stream = io.BytesIO()
            import matplotlib
            with matplotlib.rc_context({'svg.fonttype': 'none'}):
                figure.savefig(stream, format='svg')
            report['two_panel_analysis_and_draw_seconds'] = round(perf_counter() - started, 3)
            page.set_viewport_size({'width': 1440, 'height': 900})
            page.set_content('<html><body style="margin:0"><div>' + stream.getvalue().decode('utf-8') + '</div></body></html>')
            page.add_style_tag(content='svg {width:100%; height:auto;}')
            page.screenshot(path=str(output / 'vector-grid-browser.png'))
            figure, axes = workbench.draw_cell_views([definitions], canvas=Figure(), rows=1, columns=1, stacked=True)
            assert len(axes) == 2
            stream = io.BytesIO()
            with matplotlib.rc_context({'svg.fonttype': 'none'}):
                figure.savefig(stream, format='svg')
            page.set_content('<html><body style="margin:0">' + stream.getvalue().decode('utf-8') + '</body></html>')
            page.add_style_tag(content='svg {width:100%; height:auto;}')
            page.screenshot(path=str(output / 'native-stacked-lanes-browser.png'), full_page=True)
            from pymicroglia.figure_tables import cell_grid
            from pymicroglia.visualisation.figures import load
            from pymicroglia.visualisation.panels.cell_analysis_grid import analyses
            import pandas as pd
            from types import SimpleNamespace
            table = pd.DataFrame([{'identity': identity, 'hours': hour, 'frame_index': index,
                'integrated_density': value, 'area_px': 300 + 40*np.cos(2*np.pi*hour/24)}
                for identity in (3, 8) for index, (hour, value) in enumerate(zip(hours, values))])
            source = SimpleNamespace(table=lambda _: table, module_params=lambda _: {}, figure_theme='classic')
            settings = {row.name: row.default for row in load()['cell_grid'].options}
            settings.update(metrics=['integrated_density', 'area_px'], analysis_parameters={'period_testing': False},
                grid_presentation={'layout': 'stack', 'trace_layers': ['raw', 'mean', 'processed']})
            prepared, _ = cell_grid.prepare(source, settings)
            figure = Figure()
            result = analyses(figure, prepared['analyses'])
            assert len(result.axes) == 12
            stream = io.BytesIO()
            with matplotlib.rc_context({'svg.fonttype': 'none'}):
                figure.savefig(stream, format='svg')
            page.set_content('<html><body style="margin:0">' + stream.getvalue().decode('utf-8') + '</body></html>')
            page.add_style_tag(content='svg {width:100%; height:auto;}')
            page.screenshot(path=str(output / 'trace-stacked-lanes-browser.png'), full_page=True)
            report['native_and_trace_stacked_lanes_verified'] = True
            # The frozen display fixture tests colours and lane geometry;
            # its p-values are explicitly synthetic, not experimental results.
            import runpy
            from pymicroglia.figure_tables.grid_presentation import apply
            fixture_factory = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tests' / 'test_fft_component_presentation.py'))['prepared']
            for layout, count in (('overlay', 3), ('stack', 8)):
                component_fixture = fixture_factory()
                apply(component_fixture, {'grid_presentation': {'fft_components': True, 'layout': layout, 'show_fit': False}})
                figure = Figure()
                drawn = analyses(figure, component_fixture['analyses'])
                assert len(drawn.axes) == count
                stream = io.BytesIO()
                with matplotlib.rc_context({'svg.fonttype': 'none'}):
                    figure.savefig(stream, format='svg')
                page.set_content('<html><body style="margin:0"><p>Synthetic component presentation check</p>' + stream.getvalue().decode('utf-8') + '</body></html>')
                page.add_style_tag(content='svg {width:100%; height:auto;}')
                page.screenshot(path=str(output / f'fft-components-{layout}-browser.png'), full_page=True)
            report['fft_component_overlay_and_stack_verified'] = True
            report['page_errors'] = errors
            assert not errors, errors
            report['output'] = str(output.resolve())
            (output / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2))
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == '__main__':
    main()
