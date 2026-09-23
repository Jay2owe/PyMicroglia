# PyMicroglia analysis guide

PyMicroglia measures tracked microglia, learns frame-level states and whole-cell
groups, and builds scientific workflows, figures and review films.
Auto-Organotypic owns image processing, shared file records and video encoding.
Circadian Workbench owns statistical and rhythm calculations.

```
accepted tracks -> measured tables -> scientific workflows
                         |                    |
                         +--- figures/films --+
```

Use `pymicroglia discover` for actions, `pymicroglia describe ACTION` for their
current parameters and `pymicroglia doctor` for installed capabilities.
`from pymicroglia import context; context.read('quickstart')` reads this guide.
`context.search('period search range')` searches it without opening data.
