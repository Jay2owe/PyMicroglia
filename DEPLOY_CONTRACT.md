# PyMicroglia deploy contract

## Scope and baseline

The package is `PyMicroglia`; its version is read dynamically from the package.
The release version is `0.4.0` and the source identity is the commit targeted by
the annotated `v0.4.0` tag on the isolated tracking release branch.

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

<!-- deploy-required-channels: ["github", "pypi"] -->

Release 0.4.0 publishes the accepted tracking integration, its memory improvements, and supporting tests/documentation from an isolated release branch. Unrelated working-tree changes are excluded. The measured exact full-run claim covers ordinary execution. Disk-backed execution has unresolved native floating-point evidence differences and is not certified byte-identical. Publication requires a clean source commit, wheel/sdist checks, focused tracking tests, installed-wheel core/CLI/documentation smoke checks, and dependency resolution for supported extras.

Public 0.4.0 supports ordinary tracking only. The experimental bounded_memory=True option fails explicitly before reading inputs; its shared storage dependency is not available in the released Auto-Organotypic 0.7.2. No 30-percent memory claim applies to this release.
