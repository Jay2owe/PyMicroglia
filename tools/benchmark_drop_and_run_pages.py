"""Time the drop-and-run pages in a real browser, one fresh server per trial.

    python tools/benchmark_drop_and_run_pages.py list
    python tools/benchmark_drop_and_run_pages.py trials WORKFLOW [...] --fixtures DIR --before SNAP --after SNAP --out DIR [--repeats 2] [--tag T]
    python tools/benchmark_drop_and_run_pages.py serve --source SNAP --work DIR   (used by trials)

The page twin of ``benchmark_drop_and_run.py``: same fixtures, snapshots,
private stores and receipt shape. Each trial starts the workbench from one
snapshot (``pymicroglia.ui.desktop.launch``, both servers in one process, the
optional assistants off on both sides), opens a clean browser context in the
automation Chrome on http://127.0.0.1:9333, and clicks the real controls. A
receipt keeps every timed segment, what the page showed (hashed), the files the
server imported and how long the server took to start, which is not part of a
workflow's time.

Runs that a workflow starts (the chain, Auto-Organotypic processing) are
stopped by ending the server once the measured point is reached; the trial's
folders are then deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from benchmark_drop_and_run import PM, _normal, _private_store, _setup, tree_digest  # noqa: E402

CDP = "http://127.0.0.1:9333"
#: workflow -> the drop-step workload it uses (benchmark_drop_and_run._setup).
WORKFLOWS = {
    "pm.page.drop-to-table": "local12",
    "pm.page.run-to-first-stage": "small",
    "pm.page.reopen-with-results": "r20",
    "ao.page.paste-to-inputs": "local12",
    "ao.page.create-and-process": "local12",
}


# ------------------------------------------------------------------ server
def serve(args) -> int:
    sys.path[:0] = [str(Path(args.source).resolve())]
    os.chdir(args.work)  # no Motion checkout beside it, so no Motion service
    import pymicroglia, auto_organotypic
    print("imported " + json.dumps({"pymicroglia": pymicroglia.__file__,
                                    "auto_organotypic": auto_organotypic.__file__}), flush=True)
    from pymicroglia.ui.desktop import launch
    return launch(serve_only=True)


def _start(source: Path, work: Path) -> tuple[subprocess.Popen, str, dict, float]:
    env = {**os.environ, "PYMICROGLIA_AI": "0", "AUTO_ORGANOTYPIC_AI": "0", "PYTHONUNBUFFERED": "1"}
    env.setdefault("PYMICROGLIA_MASK_WEIGHTS", str(
        PM / "development" / "single_frame_mask_unet" / "MCG_mask_model" / "runs"
        / "stage3_disguise" / "model.pt"))
    t0 = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-u", str(HERE), "serve", "--source", str(source),
                                "--work", str(work)], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    imported, lines = {}, []
    for line in process.stdout:
        lines.append(line)
        if line.startswith("imported "):
            imported = json.loads(line[9:])
        if line.startswith("PyMicroglia workbench: "):
            return process, line.split(": ", 1)[1].strip(), imported, time.perf_counter() - t0
        if time.perf_counter() - t0 > 300:
            break
    _stop(process)
    raise RuntimeError("workbench did not start:\n" + "".join(lines[-30:]))


def _stop(process: subprocess.Popen) -> None:
    import psutil
    try:
        parent = psutil.Process(process.pid)
        family = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for one in family:
        try:
            one.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(family, timeout=20)


# --------------------------------------------------------------- workflows
def _since(t0: float) -> float:
    return round(time.perf_counter() - t0, 4)


def _intake(page, url: str) -> None:
    page.goto(url)  # the pages poll, so the network never goes quiet: wait for the control
    page.wait_for_function("document.querySelector('#intake-path')?.offsetParent || [...document"
                           ".querySelectorAll('button')].some(b => b.textContent.trim() === "
                           "'Analyse recordings')", timeout=60_000)
    if not page.locator("#intake-path").is_visible():
        page.get_by_role("button", name="Analyse recordings").click()
    page.wait_for_selector("#intake-path", state="visible")


def _read(page, folder: str, rows: int) -> float:
    page.fill("#intake-path", folder)
    t0 = time.perf_counter()
    page.locator("[data-intake-form] button[type=submit]").click()
    page.wait_for_function(f"document.querySelectorAll('.intake-table tbody tr').length >= {rows}",
                           timeout=120_000)
    return _since(t0)


def _table(page) -> list:
    return page.eval_on_selector_all(".intake-table tbody tr", "rows => rows.map(r => "
                                     "[...r.cells].map(c => c.innerText.trim()))")


def _auto_inputs(page, auto: str, folder: str, count: int) -> float:
    page.goto(auto)
    page.get_by_role("button", name="Open Source").first.click(timeout=60_000)
    page.wait_for_selector("#drop-path", state="visible")
    page.fill("#drop-path", folder)
    t0 = time.perf_counter()
    page.locator("#drop-path-form button[type=submit]").click()
    page.wait_for_function("document.querySelectorAll('[data-intake-label=\"name\"]').length >= "
                           f"{count}", timeout=120_000)
    return _since(t0)


def _act(workflow: str, page, url: str, auto: str, ctx: dict) -> tuple[dict, object]:
    folder = str(ctx["folder"])
    if workflow == "pm.page.drop-to-table":
        _intake(page, url)
        seconds = _read(page, folder, 12)
        return {"drop_to_table": seconds}, _table(page)
    if workflow == "pm.page.run-to-first-stage":
        _intake(page, url)
        _read(page, folder, 1)
        t0 = time.perf_counter()
        page.locator("[data-intake=run]").click()
        page.wait_for_selector(".intake-status", timeout=60_000)
        shown = _since(t0)
        page.wait_for_function("(() => { const s = document.querySelector('.intake-status strong');"
                               " return s && /^Running · (?!starting|queued)/.test(s.innerText); })()",
                               timeout=600_000)
        stage = page.inner_text(".intake-status strong")
        return {"run_to_status": shown, "run_to_first_stage": _since(t0)}, stage.split(" · ")[1]
    if workflow == "pm.page.reopen-with-results":
        # The earlier visit left this folder in the box; a clean context has no storage.
        page.add_init_script(f"localStorage.setItem('pm-intake-folder', {json.dumps(folder)})")
        t0 = time.perf_counter()
        _intake(page, url)
        opened = _since(t0)
        t1 = time.perf_counter()
        page.locator("[data-intake-form] button[type=submit]").click()
        page.wait_for_selector(".intake-result", timeout=120_000)
        results = _since(t1)
        shown = page.eval_on_selector_all(".intake-result", "rows => rows.map(r => "
                                          "r.querySelectorAll('[data-intake-open]').length)")
        return {"open_page": opened, "read_to_results": results,
                "reopen_with_results": round(opened + results, 4)}, {"results": shown, "table": _table(page)}
    if workflow == "ao.page.paste-to-inputs":
        seconds = _auto_inputs(page, auto, folder, 12)
        names = page.eval_on_selector_all("[data-intake-label=\"name\"]", "els => els.map(e => e.value)")
        return {"paste_to_inputs": seconds}, names
    if workflow == "ao.page.create-and-process":
        _auto_inputs(page, auto, folder, 12)
        # Untimed person steps: Auto-Organotypic asks for the outline and signal
        # channel of every stack, then a verify click each, before it enables Create.
        for n in range(12):
            for role, channel in (("outline", "1"), ("signal", "2")):
                page.select_option(f'select[data-source-role="{role}"][data-source-key="new:{n}"]', channel)
            page.locator(f'[data-verify-input="{n}"]').click()
            page.wait_for_function(f"APP.inspected[{n}]?.valid && !APP.inspected[{n}]._edited",
                                   timeout=60_000)
        t0 = time.perf_counter()
        page.locator("[data-action=create-and-process]").click()
        page.wait_for_function("document.body.innerText.includes('Processing submitted')", timeout=120_000)
        seconds = _since(t0)
        project = page.evaluate("({recordings: (APP.project?.recordings || []).length,"
                                " name: APP.project?.name})")
        return {"create_and_process": seconds}, project
    raise SystemExit(f"unknown workflow {workflow}")


def run_trial(workflow: str, fixtures: Path, source: Path, label: str, out: Path,
              scratch: Path) -> int:
    receipt_path = out / f"{workflow}--{label}.json"
    if receipt_path.exists():
        raise SystemExit("Choose a fresh label; existing evidence is never overwritten")
    work = scratch / "work-pages" / workflow  # fixed: recording identities hash their path
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    _private_store(fixtures, work)
    ctx = _setup(f"{workflow}.{WORKFLOWS[workflow]}", fixtures, work)
    from playwright.sync_api import sync_playwright
    process, url, imported, started = _start(source, work)
    segments, result, error = {}, None, None
    try:
        import urllib.request
        auto = json.load(urllib.request.urlopen(url + "/api/config"))["auto_url"]
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(CDP)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            try:
                segments, result = _act(workflow, context.new_page(), url, auto, ctx)
            finally:
                context.close()
    except Exception:  # recorded, not hidden
        import traceback
        error = traceback.format_exc()
    finally:
        _stop(process)
    roots = [(str(work), "<WORK>"), (str(fixtures), "<FIXTURES>")]
    payload = json.dumps(_normal(result, roots), sort_keys=True, default=repr)
    receipt = {"case": workflow, "label": label, "source": str(source), "imported": imported,
               "seconds": segments.get(workflow.rsplit(".", 1)[1].replace("-", "_")),
               "segments": segments, "server_start_seconds": round(started, 3),
               "output_sha256": hashlib.sha256(payload.encode()).hexdigest(), "error": error,
               "browser": CDP}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{workflow}--{label}.output.json").write_text(payload, encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: receipt[k] for k in ("case", "label", "seconds", "segments", "output_sha256")}),
          flush=True)
    return 1 if error else 0


def _check_identity(snapshot: Path) -> None:
    want = json.loads((snapshot / "identity.json").read_text(encoding="utf-8"))
    for name, digest in want.items():
        if tree_digest(snapshot / name) != digest:
            raise SystemExit(f"{snapshot / name} no longer matches its identity.json")


def trials(args) -> int:
    before, after = Path(args.before).resolve(), Path(args.after).resolve()
    for snap in (before, after):
        _check_identity(snap)
    order = []
    for repeat in range(args.repeats):
        order += [("before", before), ("after", after)] if repeat % 2 == 0 else \
                 [("after", after), ("before", before)]
    if args.after_only:  # the page did not exist before: judged against the budget alone
        order = [side for side in order if side[0] == "after"]
    scratch = (args.scratch or Path(tempfile.gettempdir())).resolve()
    failed = 0
    for workflow in args.workflows:
        for n, (side, source) in enumerate(order):
            failed += run_trial(workflow, args.fixtures.resolve(), source, f"{args.tag}{side}{n}",
                                args.out.resolve(), scratch)
    for snap in (before, after):
        _check_identity(snap)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    one = sub.add_parser("serve")
    one.add_argument("--source", type=Path, required=True)
    one.add_argument("--work", type=Path, required=True)
    two = sub.add_parser("trials")
    two.add_argument("workflows", nargs="+", choices=list(WORKFLOWS))
    two.add_argument("--fixtures", type=Path, required=True)
    two.add_argument("--before", required=True)
    two.add_argument("--after", required=True)
    two.add_argument("--out", type=Path, required=True)
    two.add_argument("--scratch", type=Path)
    two.add_argument("--repeats", type=int, default=2)
    two.add_argument("--tag", default="")
    two.add_argument("--after-only", action="store_true",
                     help="time only the after snapshot (a page the baseline lacks)")
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(WORKFLOWS))
        return 0
    if args.command == "serve":
        return serve(args)
    return trials(args)


if __name__ == "__main__":
    raise SystemExit(main())
