# Start from accepted tracks

Install optional learning, figure and video dependencies with
`python -m pip install "PyMicroglia[states,figure,video]"`.

1. Inspect `pymicroglia describe measure` and prepare movie declarations in the
   established analysis configuration format.
2. Read the configuration with `pymicroglia.measure.load_config(path)`.
   Relative paths resolve against the file. Pass the configuration to
   `pymicroglia.measure.run.run(config, output, claim=question)`.
3. Pool multiple recordings before a scientific workflow. Inspect
   `pymicroglia describe pool` for recording and condition inputs.
4. Run a workflow over saved measurements, then open its linked HTML report.
5. Use a figure action's named view and display settings to revise presentation.

Actions that conclude something require `claim`: the question being examined.
Public Python calls use `pymicroglia.run_action('ACTION', claim=question, **params)`.
The command-line form is `pymicroglia run ACTION key=value --claim "question"`.

The existing-output default is `version`, preserving the preceding run.
Other policies are `skip`, `error` and `overwrite`. A scientific workflow with
`skip` reuses verified completed steps in that same folder.
