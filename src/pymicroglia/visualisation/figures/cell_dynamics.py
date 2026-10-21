"""Independent figures for saved mask changes, outline history, cards and barcode."""
from ._declare import Figure, View, Input, Option, figure
from ..panels import mask_changes, outline_history, cell_cards, recording_barcode


figure(Figure("mask_change_map", "Changes in one saved cell mask", "surveillance",
    views=(View("map", mask_changes.change_map, needs=("labels", "raw")),),
    reads=(Input("labels"), Input("raw")),
    prepare="pymicroglia.figure_tables.mask_change_map:prepare",
    options=(Option("identity", 1), Option("source_frame_index", None), Option("coordinates", "cell")),
    purpose="review", grammar="heatmap"))

figure(Figure("cell_time_sculpture", "Observed cell outlines through time", "morphology",
    views=(View("outlines", outline_history.sculpture, block=True, needs=("labels",)),),
    reads=(Input("labels"),), prepare="pymicroglia.figure_tables.outline_history:prepare",
    options=(Option("identity", 1), Option("coordinates", "cell"),
             Option("elevation", 25.), Option("azimuth", -55.)),
    purpose="review", grammar="line"))

figure(Figure("cell_trace_card", "Original cell image and its measured traces", "review",
    views=(View("card", cell_cards.cards, block=True, needs=("labels", "raw")),),
    reads=(Input("labels"), Input("raw")), prepare="pymicroglia.figure_tables.cell_cards:prepare",
    options=(Option("identity", 1), Option("source_frame_index", None)),
    purpose="review", grammar="small-multiples"))

figure(Figure("recording_barcode", "Saved cell measurements across native frames", "review",
    views=(View("barcode", recording_barcode.barcode, needs=("labels", "raw")),),
    reads=(Input("labels"), Input("raw")),
    prepare="pymicroglia.figure_tables.recording_barcode:prepare",
    options=(Option("reporter_metric", "integrated_density_mean3"), Option("barcode_scale", "cell")),
    purpose="review", grammar="heatmap"))
