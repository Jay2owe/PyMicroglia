# Stage 07: Verify and open the live workbench

<!-- plan-records: {"depends_on": ["06"], "exit_gate": ["All focused exploration tests and existing UI/server, catalogue, saved-review and handoff tests pass.", "tools/verify_exploration_ui.py records a passing real-browser receipt with no JavaScript errors.", "Protected Auto-Organotypic and Motion sections match the frozen baseline.", "The real updated workbench opens with a saved measurement run.", "The seven stages are marked completed and the split-plan folder is finalised."], "files_touched": [{"change": "NEW", "path": "tools/create_exploration_fixture.py", "reason": "Reproducible source-to-measurement browser fixture"}, {"change": "NEW", "path": "tools/verify_exploration_ui.py", "reason": "Real HTTP/browser acceptance runner"}, {"change": "NEW", "path": "tests/test_ui_exploration_browser.py", "reason": "Reusable focused browser gate"}, {"change": "NEW", "path": "docs/workbench-layout/live-exploration.md", "reason": "Usage, verified scope and source contract"}, {"change": "MODIFY", "path": "README.md", "reason": "Live exploration entry point and controls"}], "read_first": ["00_overview.md", "docs/workbench-layout/demo/verify_demo.py", "src/pymicroglia/ui/desktop.py", "tests/test_ui_end_to_end.py", "tests/test_ui_server.py", "README.md"]} -->

## Why this stage exists

The final result must demonstrate the approved interactions against the real server and show that the accepted upstream workspaces remain intact.

## Prerequisites

06

## Read first

- 00_overview.md
- docs/workbench-layout/demo/verify_demo.py
- src/pymicroglia/ui/desktop.py
- tests/test_ui_end_to_end.py
- tests/test_ui_server.py
- README.md

## Scope

- Create a small reproducible real measurement fixture from synthetic source pixels and fixed identities, with gaps and representative saved results.
- Exercise actual server endpoints and browser interactions, local controls, async selection, pipeline boundaries, persistence and resizing.
- Run focused likely-regression tests and verify protected workspace fingerprints.
- Document use and limits, update the existing Research Gallery record, mark all successful stages completed, finalise the plan and open the actual app.

## Out of scope

- No large biological reanalysis batch or deployment.
- Do not refresh Graphify or run unrelated broad audits.

## Files touched

| Path | Change | Reason |
|---|---|---|
| tools/create_exploration_fixture.py | NEW | Reproducible source-to-measurement browser fixture |
| tools/verify_exploration_ui.py | NEW | Real HTTP/browser acceptance runner |
| tests/test_ui_exploration_browser.py | NEW | Reusable focused browser gate |
| docs/workbench-layout/live-exploration.md | NEW | Usage, verified scope and source contract |
| README.md | MODIFY | Live exploration entry point and controls |

## Implementation sketch

Use the dedicated browser-use profile for real UI checks. Assert nine canonical plot instances, at least two selection sources, local metric independence, preserved input fingerprints and missing observations, correct recording switch and no implicit POST job requests. Check Auto and Motion baseline code and existing integration tests. Save the receipt and UI screenshots beside the small fixture. Finish with plan_records.py finalize after every stage's concrete gates pass.

## Exit gate

1. All focused exploration tests and existing UI/server, catalogue, saved-review and handoff tests pass.
2. tools/verify_exploration_ui.py records a passing real-browser receipt with no JavaScript errors.
3. Protected Auto-Organotypic and Motion sections match the frozen baseline.
4. The real updated workbench opens with a saved measurement run.
5. The seven stages are marked completed and the split-plan folder is finalised.

## Known risks

Some biological runs lack raw files or optional saved results. The UI must retain available trace exploration with an explicit per-view explanation. A synthetic fixture proves the interaction/data contract, not biological validity of an unrun analysis.
