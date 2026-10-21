# Native portion of the Motion port, 12 October 2026

This is the earlier, partial port record. The requested complete target is
[documented separately](motion-complete-accepted-base.md); see its validation status.

The default packaged tracker is `2026-10-12-native-speed59-recording-audit192`.
Jamie authorized the port with: "port this into pymicroglia as the new accepted base".

The package contains the current Motion native pipeline and production-local
correction operators, the accepted speed implementation from round 223 execution
revision 59, and attempt 192's recording-interval and offset fixes. The ordinary
and bounded-memory runners both enter Motion's optimized execution contexts.
The existing bounded-memory storage and deferred review behavior are preserved.
An optional callback on Motion's optimized entry point permits this embedding;
ordinary command-line behavior is unchanged. Cross-run raw-evidence caching is
disabled for these isolated packaged runs.

Prepared recordings default to zero excluded leading frames. Their acquisition
interval is supplied during preparation. A configured nonnegative frame exclusion
must exactly match the returned label/raw alignment; it is no longer capped at two.
The native engine still requires observed bodies to choose an anchor, but no longer
requires five cells. Current recording-specific exclusion settings remain explicit
in existing input configurations.

## Scope

The completed-feedback study's accepted labels (round 218, attempt 4) include a
separate later sequence replayed from stored upstream arrays and tracking evidence.
That study replay is **not** the native `run_base` engine. It has not been declared
portable by this update or silently substituted with saved study labels. The
package manifest states this boundary. Future recordings do not inherit a human
approval of their identities merely because the method is accepted.

The previous packaged archive is preserved in
`validation/native_base_20261012/previous_motion_engine.zip`. Existing tracking
folders with a different engine hash remain immutable and cannot be reused under
the new engine. The installed editable PyMicroglia checkout uses the new archive;
no public release or package upload is part of this change.

## Verification

`tests/test_motion_accepted_base.py` verifies every archived source checksum,
executes the real optimized context around both runner modes, checks import-state
restoration, and exercises result alignment at zero, two and three excluded frames.
The port tests pin the new scientific configuration hash. The only scientific
configuration difference from the previous package is removal of
`anchor.minimum_observations=5`; source-frame and acquisition settings are recording
metadata. The source-port manifest is in `validation/native_base_20261012`.

This is source and integration verification. Previous complete-movie speed and
equivalence measurements belong to Motion's recorded benchmark, not a newly timed
PyMicroglia movie. No new biological accuracy or full spatial/intensity invariance
claim is made. No existing accepted label files are rewritten.
