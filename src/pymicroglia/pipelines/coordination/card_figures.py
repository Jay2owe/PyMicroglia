"""Selected connection cards with original observations and frozen diagnostics."""
from pymicroglia._sources import source_file
from pathlib import Path
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines._screening import file_hash
import pymicroglia.pipelines.coordination.display as display
import pymicroglia.pipelines.coordination.card_data as data
SLUG = 'spatial-coordination-pair-card'

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def build(ctx):
    from pymicroglia.visualisation.panels import coordination_cards
    return display.build(ctx, coordination_cards)

def produce(context):
    from pymicroglia.visualisation.panels import coordination_cards
    settings, text = data.options(context.presentation)
    saved = data.collect(context)
    values, statistics, pages, selection = data.pages(saved, context.selection.members, settings)
    metadata = {'settings': settings, 'selection': selection, 'branch_availability': saved['evidence']['branches'].to_dict('records'), 'evidence_id': context.saved('coordination-evidence').outcome.scientific_id, 'pair_inputs_id': context.saved('pair-inputs').outcome.scientific_id, 'original_time_preserved': True, 'traces_realigned': False, 'analysis_recomputed': False}
    return display.produce(context, values=values, statistics=statistics, metadata=metadata, pages=pages, slugs=[SLUG] * len(pages), panel=coordination_cards, sources=[__file__, data.__file__], text=text)
