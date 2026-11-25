# PyMicroglia deploy contract

<!-- deploy-required-channels: ["github-source", "pypi"] -->

## Scope and baseline

The distribution is `PyMicroglia`; the import package is `pymicroglia` and the
version is read from `pymicroglia.__version__`. Each release is one clean
commit on `main` of the public repository `Jay2owe/PyMicroglia`, targeted by an
annotated `vX.Y.Z` tag. PyPI already holds 0.1.0 to 0.4.0, while `main` still
reads 0.3.0: 0.4.0 was tagged on the unmerged branch `release/tracking-0.4.0`.
The next release is therefore at least 0.5.0, and its changelog must say what
0.4.0 contained that `main` does or does not carry.

Prepare every release in a fresh clone outside Dropbox (for example
`%LOCALAPPDATA%/PyMicrogliaReleases/<version>/source`). The Dropbox checkout's
git metadata is not reliable: on 2026-11-25 it had lost its commit history while
keeping the staged files, and fetching into it failed. Analysis outputs under
`development/`, `validation/` and `tmp/` never enter a release commit.

## Surfaces and artifacts

One wheel and one source distribution, built by the release workflow, carry
the package, the bundled usage guide, the frozen Motion engine and the web
assets. The release verification covers one shared synthetic fixture on every
user-facing surface:

- Python API (`pymicroglia.auto_microglia.run` and the registered actions);
- CLI (`pymicroglia`);
- browser workbench (`ui` extra), including the Analyse recordings start page
  and the integrated Auto-Organotypic workspace;
- desktop workbench (`desktop` extra, pywebview), including its file pickers.

Record the resolved versions of Auto-Organotypic, circadian-workbench,
ReproFig and analysis-kit in the release evidence. There is no standalone EXE,
Zenodo record, lab copy or documentation site for this package.

## Gates

1. Focused tests for the change, then the full owned suite locally (CI runs only
   the release group). Pre-existing failures are listed by name in the
   evidence, never skipped silently.
2. `python scripts/verify.py verify --group release --json`. If it stops on
   "generated tooling changed", the local canonical template is newer than the
   project's copies: refresh them from the template with a reason, or run the
   group's pytest commands from `architecture.yaml` directly and say so.
3. The pinned dependency ranges must include the versions the checks ran
   against. Release blocker as of 2026-11-25: `Auto-Organotypic>=0.7.2,<0.8`,
   while development and testing run against the Auto-Organotypic source,
   now 0.11.0 (its editable install still reports 0.8.0). circadian-workbench
   0.15.0 is in use and inside `>=0.15,<0.16`, but the newest release, 0.16.2,
   is outside it. Widen a range only after the surfaces pass against the newest
   release, or prove the release works inside the declared range.
4. Push-guard in push mode must be clean: the repository is public.

## Publication

1. Commit on `main` and push. The `Project architecture and interface
   contracts` workflow must pass on that commit
   (`deploy/scripts/ci_runs.py wait`).
2. Build a local wheel and source archive from the same commit; inspect them
   for private paths, secrets and research data; install the wheel in a fresh
   environment with published dependencies and run every surface above.
3. Create and push the annotated tag. `.github/workflows/release.yml` reruns the
   release group, builds, Twine-checks and publishes through PyPI trusted
   publishing (environment `pypi`). This workflow rebuilds the archives, so the
   bytes on PyPI are the workflow's, not the local build's.
4. Verify the PyPI version and file hashes, install from PyPI in a fresh
   environment and repeat the representative workflow on the API, CLI and
   browser surfaces.
5. Create the GitHub release for the tag with the changelog section as notes
   and the PyPI files as assets. A failed channel leaves the release
   incomplete; fix it and resume without retagging different source.

Keep the reviewed `DEPLOY_MANIFEST.json`, preflight receipts, surface results
and archive audit in `development/release-<version>/` beside the checkout.
