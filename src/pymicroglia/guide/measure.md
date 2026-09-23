# Measure accepted identities

`pymicroglia.measure.measure(movies, output_dir=..., claim=...)` runs declared
modules over accepted identities. `enabled_modules` chooses modules;
`module_options` supplies validated settings by module name.
`frame_interval_min` and `microns_per_pixel` supply calibration where needed.

The registry declares required inputs, tables, columns and units.
`pymicroglia.measure.parameter_choices()` lists module choices.
Disabled optional evidence remains unavailable.

Movie tables retain identity and frame keys. Pooling retains recording origin.
Windows summarize declared observations. Contrasts aggregate to the declared
replication unit before calling Circadian Workbench. Input hashes and applied
module settings travel with the run.
