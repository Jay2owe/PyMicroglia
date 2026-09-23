"""Unconditional evidence overviews from saved coordination outcomes."""
from pymicroglia._sources import source_file
from pathlib import Path
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines._screening import file_hash
import pymicroglia.pipelines.coordination.display as display
import pymicroglia.pipelines.coordination.overview_data as data
SLUG = 'spatial-coordination-overview'
SAMPLE_SLUG = 'spatial-coordination-samples'

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def build(ctx):
    from pymicroglia.visualisation.panels import coordination_overview
    return display.build(ctx, coordination_overview)

def produce(context):
    from pymicroglia.visualisation.panels import coordination_overview
    settings, text = data.options(context.presentation)
    saved = data.collect(context)
    values, statistics, pages = data.pages(saved, settings)
    slugs = [SAMPLE_SLUG if page['view'] in {'samples', 'contrasts', 'sample_timing'} else SLUG for page in pages]
    metadata = {'settings': settings, 'scientific_population': 'Complete saved effects, including untestable and non-significant results; no support filtering', 'pair_inputs_id': context.saved('pair-inputs').outcome.scientific_id if 'pair-inputs' in context.dependencies else None, 'evidence_id': context.saved('coordination-evidence').outcome.scientific_id if 'coordination-evidence' in context.dependencies else None}
    return display.produce(context, values=values, statistics=statistics, metadata=metadata, pages=pages, slugs=slugs, panel=coordination_overview, sources=[__file__, data.__file__], text=text)
