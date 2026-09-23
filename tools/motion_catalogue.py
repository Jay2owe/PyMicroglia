"""Catalogue rows for the packaged tracked-cell analyses."""
import inspect


DESCRIPTIONS = {
    "pipeline_request": "Workflow request object or path to its JSON declaration.",
    "run": "Saved measurement run whose tables this action reads.",
    "state_options": "Additional validated StateOptions settings for learning and rhythm analysis.",
    "clustering_options": "Validated ClusteringOptions settings for whole-cell fingerprints.",
    "state_method": "Learning method: gmm, kmeans, agglomerative, hdbscan or hmm.",
    "states": "Candidate state counts; null uses the recorded automatic selection policy.",
    "features": "Snapshot measurement columns to use; null selects eligible measured features.",
    "holdout": "Unit kept out of training: group, subject, stem or cells.",
    "replay": "Saved state dictionary folder; applying it never refits the state model.",
    "rhythms": "Estimate periods and significance of state traces with separate configured methods.",
    "trajectory": "Group cell trajectories using the declared time-warping settings.",
    "seed": "Seed controlling reproducible learning and resampling.",
    "cell_count": "Maximum number of cell films when explicit identities are not supplied.",
    "claim": "Scientific question this analysis is intended to address.",
}


def rows(function):
    result = []
    for name, parameter in inspect.signature(function).parameters.items():
        if name in {"options", "key"}:
            continue  # Generic dispatch placeholders are not public parameters.
        required = parameter.default is inspect.Parameter.empty
        default = None if required else parameter.default
        kind = ("bool" if isinstance(default, bool) else "int" if isinstance(default, int)
                else "float" if isinstance(default, float) else "mapping" if name.endswith("_options") or name in {"pipeline_request", "presentation"}
                else "path" if name in {"run", "output_dir", "replay"} else "list" if name in {"features", "states", "events", "identities", "output_formats"}
                else "str")
        if name == "fps": kind = "float"
        result.append(dict(name=name, type=kind, units="fps" if name == "fps" else "-", required=required,
                           default=default, description=DESCRIPTIONS.get(name, name.replace("_", " ").capitalize()+".")))
    return result


def actions(figure_common=()):
    from pymicroglia.states import states
    from pymicroglia.clustering import cluster
    result = [dict(name="states", method="states.states", params=rows(states),
                 summary="Learn shared cell-frame states and describe their dynamics and separately tested rhythms."),
            dict(name="cluster", method="clustering.cluster", params=rows(cluster),
                 summary="Learn whole-cell fingerprints and describe cohort clusters with held-out reconstruction checks.")]

    from pymicroglia.pipelines._requests import FAMILIES
    from pymicroglia.figure_tables.film_action import follow
    result.append(dict(name='follow',method='figure_tables.film_action.follow',params=rows(follow),
        summary='Review accepted cell identities over the recording, lifespan or recorded event windows; display only.'))
    import importlib
    for name in FAMILIES:
        module = importlib.import_module("pymicroglia.pipelines." + name)
        result.append(dict(name=name, method=f"pipelines.{name}.{name}",
                           params=rows(getattr(module,name)), summary=module.__doc__))
    from pymicroglia.visualisation.figures import load
    from pymicroglia.visualisation.figures._vocabulary import meaning
    from pymicroglia.figure_tables.actions import draw
    for spec in load().values():
        params = rows(draw)
        common={row['name']:row for row in figure_common}
        params=[{**common[row['name']], 'default':row['default']} if row['name'] in common else row for row in params]
        present={row['name'] for row in params}
        params.extend(dict(row) for row in figure_common if row['name'] not in present)
        for row in params:
            if row["name"] == "view":
                row["choices"] = [v.key for v in spec.views]
                row["description"] = "Named view; null composes every view."
        params.append(dict(name="item",type="str",units="-",required=False,default=None,description="Saved figure-plan item identifying the exact scientific inputs."))
        for option in spec.options:
            default=option.default
            params.append(dict(name=option.name, type=meaning(option.name)["type"],
                units=meaning(option.name).get("units", "-"), required=False,
                default=default, description=option.description))
            if option.choices:
                params[-1]['choices'] = list(option.choices)
        result.append(dict(name=spec.key,method="figure_tables.actions."+spec.key,params=params,summary=spec.title,display_only=True))
    return result
