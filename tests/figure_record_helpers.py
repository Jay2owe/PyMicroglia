"""Read the exact figure entry from its shared folder ledger."""
from auto_organotypic.store.ledger import entry_for

def figure_record(result):
    entry=entry_for(result["figures"][0])
    assert entry is not None and entry["stage"]=="figure"
    extra=entry["extra"]
    return {"settings":entry["params"],"claim":extra["claim"],
            "drawn":extra["drawn"],"generated_from":extra["sources"],
            "plotted_table":extra["table"],"artefacts_drawn":extra["artefacts"]}
