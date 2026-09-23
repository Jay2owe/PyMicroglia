"""Review saved state histories through the shared figure action."""


def draw(run, max_cells=6, *, view=None, output_dir=None, **options):
    from ..figure_tables.actions import draw as figure
    return figure('state_histories',run,cells=max_cells,view=view,output_dir=output_dir,**options)
