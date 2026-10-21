<!-- aiify app map: app=pymicroglia sources=2adad410f5946e9f built=2026-10-21 by=claude -->
# App map: pymicroglia

## Overview
PyMicroglia is a lab analysis workbench for time-lapse recordings of microglia and SCN (suprachiasmatic nucleus) tissue slices, including bioluminescence (dLuc, Cry1-dLuc) and red/green fluorescence movies. It takes saved cell tracks, measures every cell in every frame, and lets a researcher look at one cell in context, test cell rhythms, review track events, compare conditions and export figures, videos and data. It is for research scientists working with these recordings. It opens as a desktop window (`pymicroglia ui`) or in a browser on the same computer (`pymicroglia serve`). Every analysis can also be run from the command line (`pymicroglia run ACTION key=value --claim "question"`).

## Screens

### Workbench header (top tabs)
Always visible at the top. It shows the brand "PyMicroglia WORKBENCH", a "LIVE ANALYSIS" badge and three workspace tabs: **Auto-Organotypic**, **Motion tracks** and **PyMicroglia extras**. The app remembers which tab you used last.

### PyMicroglia extras sidebar
Visible when the **PyMicroglia extras** tab is selected. It is headed "MICROGLIAL ANALYSIS / Explore together" and has these pages: **Follow a cell**, **Tissue & neighbours**, **Compare cells & groups**, **Events** and **Rhythm audit**. Under "RUN ON DEMAND" it has **Pipelines**, **Results and export** and **All methods**. It also has an **Appearance** menu (Use system setting / Light / Dark) and a readout of the selected recording, cell and source frame.

### Saved measurement run toolbar
Shown above most PyMicroglia extras pages. **Saved measurement run** is a text box for the run folder. **Browse folders** appears only in the desktop window. **Review saved files** opens the run's files on the Results and export page. A status line shows how many saved files and measured cells the run has, or "No run selected".

### Follow a cell (A cell in context)
Section 01, "Follow a cell through time". Every view here shares the same selected cell and time.
- **Shared transport bar:** **Recording**, **Cell**, previous/next frame buttons, **Play**/**Pause**, a frame slider, a time label and **Rate** (2/4/8 fps).
- **Grey outlines:** a toolbar button that opens a rule builder for greying out cells in images. It has **Export outlined image** and **Export outlined video**.
- **Rhythm analyses rows:** **+ Add analysis** (up to 16 rows). Each row has **Measurements** (with **Select all**/**Clear**), **Method**, **Detrending**, **Run this recording**, **Run all recordings**, **Advanced**, **Show this result in plots**, **Exclusions**, **Remove**, a completed-run menu ("Choose a completed run") and **Use shown settings**. **Advanced** opens "Rhythm analysis settings": Period estimator, Significance test, Minimum period (h), Maximum period (h), Minimum samples, Minimum cycles, Significance threshold, and Correction (None / False discovery rate / Bonferroni / Sidak).
- **Cell image:** **View** (Cell crop / Whole field) and **Colour**. Its Settings hold **Outline** (Saved masks / Base / Base + saved / Measurement regions / None), **Motion trail**, **Trail frames** and **Contrast**. It shows two pictures, "Original signal + current outline" and "Gained and lost mask pixels". Click a cell in the field to select it.
- **Outline history:** a rotatable 3-D stack of the cell's outlines, with time as height. It has **Reset view**, and its Settings hold **Position** (Cell centred / Include movement) and **Show** (Full recording / 48 frames around selection).
- **Measurement trace:** **Metric**, **Light** (Total in mask / Mean per pixel), **Time mean** (Raw / 3 frames / 7 frames) and **Audit this trace**. Its Settings hold Extra measurements, **Measurements** (Stack / Overlay), Overlay scale, **Colour by significance**, **Dashed rhythm fit** and **Trace values** (Shown analysis / Raw measurements). It also shows two **Nearby cell** comparison cards. Click a trace to move time.
- **Recording barcode:** a grid of cells by time. It has **Metric**, **Light**, **Time mean** and **Engine** (row order), and its Settings hold **Scale** (Within each cell / Shared recording scale) and **Colour by significance**. Click a row and time to update every view.
- **All cell traces:** a switch between **Separate cells** and **Overlay cells**, plus **Columns**, **Metric**, **Light** and **Time mean**. Its Settings hold **Y scale**, **Dashed rhythm fit**, **Guides every (h)**, **Guide offset (h)**, **Compare cells** (Selected & nearby / All / Clear) and **Overlay scale**.
- **Track timeline / Gantt:** **View** (All cell tracks (Gantt) / Selected cell events / Cell states / Observation gaps), **Order** in Settings, and a **Configure state analysis** button.
- **Phase clock:** **Cells** (All / Selected group) and a **Configure rhythm analysis** button. Click a marker to select its cell.
- **Buttons on every plot panel:** **Edit plot** opens "Plot appearance": Height, Label size, Line width, Data colour, Background, Y minimum/maximum and **Undo appearance**. Panels also have **Settings**, **Export figure** (a **Saved figure** menu plus **Configure export**) and **Exclusions** with **Use main exclusions**.
- **Bottom of the page:** **Control help** lists what each control does. **Data source and display provenance** shows where the data came from.

### Tissue & neighbours
Section 02, "See the tissue & neighbours".
- **Tissue tectonics / map:** **Layer**, plus **Period measurement** in Settings. Click to select a cell. **Add plot** and **Columns** make more maps side by side.
- **Cell relationships:** **Show** (Selected neighbourhood / All cells), **Links** in Settings and a **Configure contact measurements** button. Click a node to select its cell.

### Compare cells & groups
Section 03, "Compare cells & groups".
- **Set conditions:** **Load recordings together** (one folder per line) with **Add recordings**. **Select** chooses Cells by ID, Recordings or Recording name pattern, with **Recording**, **Cell IDs**, **Select all**, **Clear selection**, **Name pattern** and **Ignore case**. Then **Condition** and **Set condition**, a recordings table, and **Earlier**/**Later** to reorder conditions. Its **Advanced** section has **Save design**, **Load design**, **Extra factors**, **Sample identifiers** and **Saved rhythm runs**.
- **Cell & group comparison plot:** **Metric**, **Light**, **Plot**, **Group** (Condition / Recording / Subject / One population) and **Points** (Recording / Cell / Subject).
  - **Comparison settings:** Use (Time window / Selected frame), Cell summary, Within sample, From (h), To (h), Included groups. Its **Advanced** section has Normalisation, Detrending, Input filter, Window, Band, Bins, Density bandwidth and the JSON options.
  - **Statistical comparison:** Test, Test unit, Correction, Alpha, **Run comparison** and **Completed**.
  - **Export comparison:** SVG / PNG / Data and settings / Reproduce in Python, then **Download**.
  - **Add plot** and **Columns** make more comparison plots.

### Events
Reached from the sidebar's **Events**. It shows a grid of cells with recorded track events.
- **Event type tabs:** All events, Births, Divisions, Merges / hiding, Appeared, Entered field, Left field, Deaths, Lost identities, Vanished, Recording start, Recording end, Other gaps and State changes.
- **Left panel:** **Saved measurement run**, a recording menu and **Refresh**. A status filter offers All statuses, Exclude recording boundaries, Candidate, Unproven, Observed, Tracking assessment, Unexplained or Censored at boundary. There is also a **Cell or event** search and an event count.
- **Selected-event view:** **Back to grid**, **Preview**, **Saved video**, a saved-video version menu, **Save event video**, **Download** and **Reproduce**. Below the picture are frame back/forward buttons, **Play** and a slider.
- **Right panel:** "Recorded event" facts, **Open selected event** and **Inspect in PyMicroglia extras**. **Display and video settings** holds Hours each side, Playback fps, Crop pixels and Colour. **Recorded evidence** shows the saved record.

### Rhythm audit
Reached from the sidebar's **Rhythm audit**, or from **Audit this trace** on the Measurement trace. It compares many detrending and period-finding methods on one cell's trace.
- **Choosing what to audit:** **Recording**, **Cell**, **Measurement**, **Run audit** and **Cancel audit**.
- **Comparison settings:** tick boxes for **Detrending rows** and **Period-estimator columns**, plus **Rhythmicity test** and **Normalisation (display)**. **Additional input filters** has **+ Add filter**, each with a Window, samples box and **Remove**. **Advanced analysis settings** holds period limits, minimums, threshold and correction.
- **Earlier audits:** **Completed audit** and **Restore shown settings**.
- **Results:** a **Raw trace** panel, and a **Detrending and rhythm comparison** grid with Show input, Guides every (h), Guide offset (h), Time start and Time end. Clicking a result opens **Selected comparison**, which has **Use this setup**, "Components and spectral diagnostics" and "Recorded evidence and settings". Each panel's **Export** lists the saved files, **Reproduce in Python** and **Configure native audit figure**.

### Pipelines
Sidebar → **Pipelines** ("Complete workflows"). It shows one card per complete workflow, each with **Set up →**: auto microglia, Full recordings: original Red and Green, dluc single cell, cry1 dluc photon, bioluminescence, phase green red, rhythm discovery, method audit, measurement relationships, behaviour states, spatial coordination and intervention response.

### Results and export
Sidebar → **Results and export**, or **Review saved files**. **Saved evidence** shows previews of the run's images and videos, each with **Open**. **Recent recorded analyses** shows up to 20 past runs, each with **Inspect →**.

### All methods
Sidebar → **All methods**. It has a search box ("Search actions by name or description"). Every registered action appears as a card grouped by area, each with **Set up →**. A separate section, "All saved figures and composite reports", lists every exportable figure.

### Action settings dialog (LIVE PYTHON ACTION)
Opens from any **Set up →**, **Configure export** or **Configure …** button. It shows the action title and summary, the required fields, and the rest under **More settings**. A **claim** field ("What should this analysis show?") appears when the analysis draws a conclusion. Cell-grid actions add **Analysis settings** and **Grid arrangement and overlays**. Buttons: **Check settings** and **Run analysis**. For figures, it also lists which saved inputs are ready or unavailable.

### Jobs tray
The **Jobs N** button sits at the bottom corner. It opens "Analysis jobs", which lists each job's action, its state (queued / running / completed / failed) and its progress.

### Motion tracks workspace
Opened with the **Motion tracks** tab, and needs the Motion service to be connected. The toolbar has **Source** (a saved editing bundle, session or Motion package) and **Refresh sources**. Sidebar pages:
- **Review frames:** the original frame with saved identities, a frame slider with back/forward buttons, and a "Cell N" button for each cell in the frame.
- **Continuity and gaps:** **Frame** and **Show cells**. Click a cell to see its occupied frame runs and gaps.
- **Correct identities:** edit buttons swap identities, reassign identities, delete between frames, remove identities, apply plan and apply batch. Read-only inspect/preview buttons and feedback-note buttons sit below.
- **Outline handoff:** **Open mask change figure** and **Prepare measurement inputs**.
- **History and export:** a numbered list of saved correction steps, with **build review** and **render movie**.

### Motion session action dialog
Opens from a Motion action button and shows that action's fields. Its button reads **Save new session** for edits, **Save output**, or **Run read-only action**. The original source is never changed.

### Choose outline refinement dialog
Opens after **Prepare measurement inputs**. **Method** offers Single-frame gated U-Net (accepted default), CD68 foreground U-Net, Saved mask or outline, Custom Python provider, or Off: use Motion outlines. Other fields: **Custom method**, **Model weights, if needed** (with **Browse model** in the desktop window), **Method options as JSON**, and "Allow a missing cell to be removed for that frame…". The button is **Prepare refined masks**, or **Continue to measurements** when the method is Off.

### Auto-Organotypic workspace
Opened with the **Auto-Organotypic** tab. It shows the separately installed Auto-Organotypic interface inside the window, used for image processing, registration and outlining. If that interface is not installed, the tab says so.

## Tasks

### How to open a saved measurement run
1. Click the **PyMicroglia extras** tab.
2. Type or paste the run folder into **Saved measurement run**, or click **Browse folders** in the desktop window.
3. Press Enter or click outside the box. The status line shows the number of saved files and measured cells.
4. Pick the movie in the **Recording** menu.

### How to look at the saved output files of a run
1. Open the run (see above), then click **Review saved files**, or click **Results and export** in the sidebar.
2. Under **Saved evidence**, click **Open** on any image, video or file.

### How to reopen an earlier analysis
1. Sidebar → **Results and export**.
2. Under **Recent recorded analyses**, click **Inspect →** on a run to list its outputs.

### How to follow one cell through the recording
1. Sidebar → **Follow a cell**.
2. Choose the cell in **Cell**, or click it in **Cell image** (Whole field), on the tissue map, in the barcode or in the phase clock.
3. Use **Play**, the frame slider or the previous/next frame buttons. Set the speed with **Rate**.
4. Every linked view follows the same cell and frame.

### How to change which measurement is plotted
1. In **Measurement trace**, **Recording barcode** or **All cell traces**, choose **Metric**.
2. For light measurements, set **Light** to Total in mask or Mean per pixel.
3. Set **Time mean** to Raw, 3 frames or 7 frames.

### How to compare the selected cell with nearby cells
1. In **Measurement trace**, choose cells in the two **Nearby cell** menus.
2. Alternatively, in **All cell traces** → **Settings** → **Compare cells**, click **Selected & nearby** or **All**, then switch to **Overlay cells**.

### How to test whether cells are rhythmic
1. In the **Rhythm analyses** row, open **Measurements** and tick the measurements to test.
2. Choose **Method** and **Detrending**. Click **Advanced** for period limits, significance threshold and correction.
3. Click **Run this recording**. Progress shows in the row and in **Jobs**.
4. Tick **Show this result in plots** to colour traces and barcodes by significance.

### How to run the same rhythm analysis on every loaded recording
1. Set up a rhythm analysis row as above.
2. Click **Run all recordings**.

### How to try several rhythm settings side by side
1. Click **+ Add analysis** to add another independent row.
2. Pick a past result from that row's completed-run menu. Use **Use shown settings** to copy its settings back.

### How to find the best detrending and period method for a cell
1. On **Measurement trace**, click **Audit this trace**, or open **Rhythm audit** from the sidebar.
2. Choose **Cell** and **Measurement**. Tick **Detrending rows** and **Period-estimator columns**, and choose a **Rhythmicity test**.
3. Optionally click **+ Add filter** to compare smoothed inputs.
4. Click **Run audit**. Use **Cancel audit** to stop.
5. Click a result in the grid to see **Selected comparison**.
6. Click **Use this setup** to add it as a rhythm analysis row on Follow a cell, then run it there.

### How to export a rhythm audit
1. On a Rhythm audit panel, open **Export**.
2. Download the listed files, or **Reproduce in Python**. Click **Configure native audit figure** to make the figure through period_method_audit.

### How to exclude cells from the plots
1. Click **Exclusions** on a rhythm row, or **Cell filters**.
2. Choose **Action** (Exclude matching cells / Keep matching cells) and **Unknown evidence**.
3. Add rules with **Non-rhythmic**, **Gap**, **Missing frames**, **Merge** or **Near another cell**, or build your own with **+ Rule**, **+ Group** and **NOT**.
4. Click **Preview**, then **Apply**. **Turn off** restores all cells and **Clear draft** clears the rules. Saved data does not change.
5. To give one plot different exclusions, use that panel's **Exclusions**. **Use main exclusions** undoes this.

### How to save, load or reproduce a cell filter
1. In the filter panel, open **Evidence and export**.
2. Click **Save filter JSON** or **Reproduce in Python**. Use **Load filter** to bring a saved filter back.
3. To use a pass/fail list from a CSV, use **Import decisions**.

### How to grey out some cells and export an outlined image or video
1. Click **Grey outlines** under the transport bar and build rules as for exclusions. Pick a **Grey colour**.
2. Click **Apply**.
3. Click **Export outlined image** (uses the current frame) or **Export outlined video**. Then **Run analysis** in the dialog (tracked_cell_image / tracked_cell_video).

### How to change how a plot looks
1. Click **Edit plot** on the panel.
2. Change Height, Label size, Line width, Data colour, Background or Y minimum/maximum.
3. **Undo appearance** steps back.

### How to show more than one plot of the same kind
1. Above the barcode, tissue map or comparison plot, click **Add plot**.
2. Set **Columns**. Each copy has its own settings.

### How to export a figure from a view
1. On the panel, open **Export figure**.
2. Choose the figure in **Saved figure**, then click **Configure export**.
3. Adjust the settings in the dialog, enter a claim if asked, then click **Check settings** and **Run analysis**.
4. The output appears in **Results and export**.

### How to label conditions and compare groups
1. Sidebar → **Compare cells & groups** → **Set conditions**.
2. To add more runs, paste their folders into **Load recordings together** and click **Add recordings**.
3. Use **Select** to pick cells by ID, recordings, or a recording name pattern. Type a **Condition** and click **Set condition**.
4. In the comparison plot, choose **Metric**, **Plot**, **Group** and **Points**. Set the time window in **Comparison settings**.
5. Use **Save design** / **Load design** under **Advanced** to keep the setup.

### How to run a statistical test between groups
1. In **Cell & group comparison**, open **Comparison settings** → **Statistical comparison**.
2. Choose **Test**, **Test unit**, **Correction** and **Alpha**.
3. Click **Run comparison**. Reopen past results from **Completed**.

### How to export the comparison summary or data
1. Open **Export comparison** on the comparison plot.
2. Choose SVG, PNG, **Data and settings** (the data and settings as JSON) or **Reproduce in Python**.
3. Click **Download**.

### How to combine figures and statistics into one Excel workbook
1. Sidebar → **All methods** and search "publication workbook".
2. Click **Set up →** on publication_workbook, fill in the fields, then click **Run analysis**.

### How to review cell births, divisions, merges and gaps
1. Sidebar → **Events**. Check **Saved measurement run** and choose the recording.
2. Pick an event-type tab, and narrow the list with the status filter or the **Cell or event** search.
3. Click a cell in the grid, then click **Open selected event**.
4. Use **Preview** and the slider to step through the original frames with outlines.

### How to save or download an event video
1. Open an event in **Events**.
2. Under **Display and video settings**, set Hours each side, Playback fps, Crop pixels and Colour.
3. Click **Save event video**. When it is done, use **Saved video**, **Download**, or **Reproduce** for the Python script.

### How to jump from an event to the linked plots
1. In an opened event, click **Inspect in PyMicroglia extras**.
2. Follow a cell opens on that cell and frame.

### How to run a complete pipeline
1. Sidebar → **Pipelines**.
2. Click **Set up →** on the workflow, for example rhythm discovery or behaviour states.
3. Fill in the required fields and the **claim**, then click **Check settings** and **Run analysis**.
4. Watch it in **Jobs**.

### How to start a rhythm, state or contact analysis from a view
1. Click **Configure rhythm analysis** (Phase clock, rhythm_discovery), **Configure state analysis** (Track timeline, behaviour_states) or **Configure contact measurements** (Cell relationships, measure).
2. Complete the dialog and click **Run analysis**.

### How to find and run any analysis
1. Sidebar → **All methods** and type in the search box.
2. Click **Set up →**, complete the fields, open **More settings** if needed, and click **Check settings**, then **Run analysis**.

### How to check on running analyses
1. Click **Jobs** at the bottom of the window.
2. Each job shows its state and progress. **×** closes the tray.

### How to review and correct cell tracks
1. Click the **Motion tracks** tab and choose a **Source** (use **Refresh sources** if it is missing).
2. Use **Review frames** or **Continuity and gaps** to inspect identities.
3. Under **Correct identities**, click an edit such as swap identities or reassign identities, fill in the dialog, and click **Save new session**.
4. Check past steps under **History and export**.

### How to send corrected tracks for measurement
1. In **Motion tracks** → **Outline handoff**, click **Prepare measurement inputs**.
2. In **Choose outline refinement**, pick a **Method** and the model weights if needed, then click **Prepare refined masks**. Choose Off: use Motion outlines and **Continue to measurements** to skip refinement.
3. When it finishes, the measure dialog opens. Click **Run analysis** to make the measurement tables (route measure).

### How to see how one cell's mask changed
1. In **Motion tracks** → **Outline handoff**, click **Open mask change figure** (mask_change_map).
2. Alternatively, look at "Gained and lost mask pixels" in **Cell image**.

### How to make a review video of the tracks
1. In **Motion tracks** → **History and export**, click **render movie** or **build review**.
2. For outlined photon movies, use tracked_cell_video from **All methods**.

### How to pool several recordings before a workflow
1. Run route pool from **All methods** (search "pool"). It combines a run's per-movie tables into `<run>/pooled/`.

### How to export cell outlines to Fiji
1. Run route export_roi from **All methods**. It writes a RoiSet.zip that Fiji can open.

### How to turn a TIFF stack into an MP4
1. Run route stack_to_mp4 from **All methods**. Set the experimental hours per second and the colour map.

### How to register drift and remove cosmic rays
1. Run route register, then remove_cosmic_rays, from **All methods**.
2. To check the result, use registration_figure and cosmic_ray_preview.

### How to switch between light and dark appearance
1. In the PyMicroglia extras sidebar, set **Appearance** to Use system setting, Light or Dark.

## Terms

### Saved measurement run
A folder of measurement tables written by the measure step from tracked cells. Most pages need one.

### Recording
One movie within a run, chosen in the **Recording** menu.

### Cell / identity
A tracked cell with a number that lasts across frames. Motion assigns it, and it does not change during measurement.

### Source frame
The frame number in the original recording, shown in the sidebar readout.

### Claim
The question an analysis is examining. It is required for actions that draw a conclusion.

### Action vs pipeline
An action is one analysis or figure step. A pipeline is a complete multi-step workflow, run from **Pipelines**.

### Motion
The tracking engine. It links cell outlines across frames and saves sessions that can be corrected.

### Session
A saved state of Motion tracks. Every correction saves a new session and leaves the source unchanged.

### Outline refinement
An optional step after tracking. It changes only the mask pixels used for measurement, not cell identities.

### Mask / outline
The pixels assigned to a cell in a frame. "Gained" pixels are cyan and "lost" pixels are gold.

### Light: Total in mask / Mean per pixel
Whether a light measurement is the sum over the cell's pixels or the average per pixel.

### Time mean
A moving average over 3 or 7 frames, used for display. "Raw" means no smoothing.

### Detrending
Removing slow drift from a trace before testing it for rhythms.

### Period estimator
The method that estimates a rhythm's period length in hours.

### Significance test / Correction
The test that decides whether a rhythm is real, and the adjustment applied when many cells are tested at once (False discovery rate, Bonferroni, Sidak).

### Rhythm audit
A side-by-side grid of detrending × period-method results for one cell and measurement.

### Exclusions / cohort
Rules that hide cells from displays. Saved data and test populations do not change.

### Unknown evidence
Cells with no data for a rule, for example untested rhythms. You choose whether to keep them, exclude them, or require evidence.

### Recording barcode
A cells-by-time colour grid of one measurement. Blank cells mean no observation.

### Outline history
A rotatable 3-D stack of a cell's outlines, with time as height.

### Track timeline / Gantt
Bars showing when each cell was observed, with its gaps, saved events and states.

### Phase clock
A dial showing where each cell's fitted rhythm peaks within its cycle.

### Tissue tectonics
Maps of cell coverage and motion across the whole tissue field.

### Condition
A group label (such as control or treatment) given to cells or recordings for comparison.

### Test unit / Points
The independent unit for statistics and plot points: recording, cell or subject.

### Event status
How certain a recorded event is: Candidate, Unproven, Observed, Tracking assessment, Unexplained, or Censored at boundary (cut off by the start or end of the recording).

### Decoys / instrumental control
Area-matched regions placed on the tissue, plus checks off the tissue and in other channels. They test whether a rhythm comes from the instrument rather than the cells.

### Reproduce in Python
A downloadable script that repeats the shown filter, comparison, clip or audit.

### Existing-output policy
What happens when an output folder already exists. version (default) keeps the old one, skip reuses finished steps, error stops, and overwrite replaces it.
