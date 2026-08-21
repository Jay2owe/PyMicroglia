/*
 * A synthetic ImageJ macro wrapper.
 * RUNNER_REQUIRES_ROI=0
 * RUNNER_OUTPUT_SUFFIX=_fixture
 */

// ============================ PROTOCOL PARAMETERS ============================
var DEFAULT_PYTHON_ENGINE = "fixture_engine.py";   // Sibling engine to call.
var DEFAULT_SATURATION = 0.35;                     // Display saturation.
var DEFAULT_THRESHOLD_SIGMA = 9.0;                 // Shared name; the Python
                                                   // engine wins on merge.
// ========================== END PROTOCOL PARAMETERS ==========================

requires("1.53d");
