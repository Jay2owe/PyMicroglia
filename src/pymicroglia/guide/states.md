# Learn and replay cell states

The `states` action learns a shared vocabulary of cell-frame measurements.
Choose features, learning method, candidate state counts, held-out grouping and
seed. `state_options` supplies remaining validated StateOptions fields.

Training and validation remain distinct. Unsupported state structure is an
explicit result. `replay` applies a saved dictionary without refitting.
Occupancy, transitions, bouts and trajectories describe accepted assignments.
Optional state rhythms use separately configured estimators and significance
tests. Holding out cells is not biological-sample validation.

Install the `states` extra for learning dependencies. The `behaviour_states`
workflow adds declared validation, observed profiles, original-time timelines,
sample comparisons and a linked saved report.


Choose the public learning algorithm with `state_method`; `method` belongs
to the registration action. Detailed learning settings remain in
`state_options`.
