# PyMicroglia deploy contract

## Scope and baseline

The package is `PyMicroglia`; its version is read dynamically from the package.
The release version is `0.3.0` and the source identity is the commit targeted by
the annotated `v0.3.0` tag on `main`.

## Artifacts and destinations

The tag-triggered `.github/workflows/release.yml` builds and Twine-checks one
wheel and one source distribution, then publishes them to PyPI as
`PyMicroglia`. The release contract also requires a clean commit/tag pushed to
`Jay2owe/PyMicroglia` and verification at `https://pypi.org/project/PyMicroglia/`.

One combined clean-install run must cover the core package, CLI, optional
feature extras, fixture-backed pipeline actions, and the versioned bundled
documentation. Because the package integrates with Auto-Organotypic,
Circadian Workbench, ReproFig, and optional Fiji workflows, record the
resolved dependency versions in the release evidence.

The existing audit calls GitHub the intended release target, but the live
workflow clearly publishes PyPI. Both channels are therefore required.

No EXE or Zenodo route is documented. Do not create one implicitly.
