# Architecture and interface checks

From this checkout:

```powershell
python -m pip install -r scripts/contracts-requirements.txt
python scripts/verify.py
python scripts/verify.py discover
python scripts/verify.py --group release --json
```

The default calls the existing capability-map validator and tests linked cell
selection, missing observations and current workspace routing. The release
group also tests saved exploration context, image routing and cell filters.
Node.js is required for tests of production browser helpers; required skips fail.

The all-cell comparison is explicitly scoped as `cell-grid`. It complements
the selected-cell inspector; the uniqueness check still rejects duplicate
canonical panels and any additional undeclared comparison home.

`architecture.yaml` is editable project configuration. The runner and schema are
versioned developer templates guarded by `architecture.lock.json`; they work
without an agent skill installed. Update canonical templates with the installer
`--update --reason`, preserving local configuration. Discovery returns actual
validated helper definitions and call sites; it does not execute tests.

Theme checking covers adopted cell-filter selectors. This is not certification
of all legacy styles, scientific images, renderer pixels or performance budgets.
Existing scientific registries remain the owners of methods and defaults.

The dedicated workflow runs on pushes/pull requests. Publication verifies the
release group before building. Remote branch protection must be set separately.
