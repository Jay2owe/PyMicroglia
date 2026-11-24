# 05 integrate

Inside `pymicroglia.measure.modules.lifecycle`: add `division_score` and `division_features` columns and a status that states a score exists but still is not a biological confirmation. Bump `METHOD_VERSION`; record thresholds in the manifest. Add an Events page filter for scored divisions, and let `follow` pick them. Tests: synthetic dividing, fragmenting and merging cells, plus the labelled gold set as a regression fixture. Update `guide/lifecycle.md`.

Done when: lifecycle tests pass, the Events page shows the new filter, and unchanged recordings give identical old columns.
