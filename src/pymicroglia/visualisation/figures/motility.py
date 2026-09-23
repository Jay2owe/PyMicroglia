"""Views of recorded cell movement and its measured displacement curves."""
from ._declare import Figure, View, Table, Option, figure
from ..panels import motility

figure(Figure('step_size_distribution','Frame-to-frame step size distribution','motility',
    views=(View('distribution',motility.histogram),),reads=(Table('cell_frame'),),
    prepare='pymicroglia.figure_tables.motility:step_size',
    options=(Option('metrics','step_px_gapless'),Option('bins',45),Option('trace_luts',None)),
    claim='Positive steps measured between consecutive observations; gaps are excluded by the selected gapless metric.'))

figure(Figure('displacement_curves','How centroid displacement grows with elapsed time','motility',
    views=(View('curves',motility.curves),View('alpha',motility.histogram)),
    reads=(Table('msd_curves'),Table('cell_summary')),prepare='pymicroglia.figure_tables.motility:displacement',
    options=(Option('metrics','msd_alpha'),Option('max_lag',None),Option('bins',18)),layout='column',
    claim='Saved centroid displacement curves and their fitted growth exponents; the exponent does not establish a movement mechanism.'))

from ..panels import trajectories
figure(Figure('identity_trajectories','Centroid paths with independent measured encodings','motility',views=(View('map',trajectories.paths),),
 reads=(Table('cell_frame'),Table('cell_summary'),Table('rhythms',optional=True)),prepare='pymicroglia.figure_tables.trajectories:prepare',
 options=(Option('metrics','net_displacement_px'),Option('trace_luts',None),Option('line_width_metric',''),Option('line_width_range',[.55,2.8]),Option('line_dash_metric',''),Option('rhythm_metric','corrected_mean'),Option('period_bins',[2.,12.,20.,28.,48.]))))
