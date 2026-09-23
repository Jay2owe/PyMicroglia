# Six saved scientific workflows

Workflow actions read saved measurements; multiple movies require pooled tables.
Each takes `run`, `pipeline_request` (a request object or JSON path),
optional presentation, step selection through `only`, output policy and claim.
The separate run-envelope `request` is descriptive text.

| Action | Request pipeline name | Question |
|---|---|---|
| rhythm_discovery | rhythm-discovery | Cell detection, supported periods and compatible timing |
| method_audit | method-selection-audit | Method development and independent confirmation |
| measurement_relationships | measurement-relationships | Within-cell, between-cell and lag associations |
| behaviour_states | cell-behaviour-states | Supported shared states and observed dynamics |
| spatial_coordination | spatial-coordination | Spatial evidence, proximity and compatible timing |
| intervention_response | intervention-response | Declared responses, controls and sample effects |

Requests retain measurements, population, samples and scientific settings.
Selecting steps includes their prerequisites. Reuse requires matching scientific
identities, producer code and evidence hashes. Unavailable dependencies retain
their reasons. Opening the linked `index.html` performs no fresh science.

A prepared request can be submitted with
`run_action('rhythm_discovery', run=folder, pipeline_request=request_path,
if_exists='skip', claim=question)`. Audit profile export is an explicit
selection of saved decisions; rendering does not select a method.
