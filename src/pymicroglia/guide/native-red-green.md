# Full recordings: original Red and Green

In **Pipelines**, select **Full recordings: original Red and Green**.
Install the `native` extra for this optional workflow (`PyMicroglia[native]`).
Supply the original Phase, Red and Green single-channel TIFFs, or reuse a
verified full-camera registration manifest. Motion detects and tracks cells
from registered Red minus Green, retaining all acquired frames. No U-Net mask
gates Motion's fluorescence input. Optional U-Net refinement runs afterward on
the same Red-minus-Green signal; select its checkpoint only when enabling it.
All acquired frames and the entire camera are retained. An optional acquisition
clock CSV preserves missing acquisitions. Several named input mappings can be
supplied in `recordings`; they run serially and share one saved workspace.

Choose **Original tracked outline** or **Expanded outline, exclusive cell
ownership**, set the expansion in native camera pixels, and choose whether the
preferred trace subtracts background. The workflow always retains these three
readouts for comparison: original-outline total, expanded-outline total and
expanded total minus local background. Expansion never changes cell identities,
invades another cell's original pixels or enters detected untracked foreground.
Background excludes every expanded cell and blocked foreground. Insufficient
background remains missing; negative corrected measurements remain negative.

**Relative-Intensity Pattern Registration** uses Auto-Organotypic's packaged
engine and requires an explicit recipe. **Structural features: rigid rotation
and translation** is a separate optional method. Both estimate a shared pose
from Phase. Registered Red minus Green is used for detection and tracking; labels are
projected back onto original Red and Green camera pixels before measurement.
No interpolated registration proxy is used as the quantitative signal.

The completed job opens its measurement workspace. Choose **Measurement regions**
under **Outline** to show the original outline, measured expanded outline and
local background. The three totals appear together below the selected cell.
The Recording dropdown includes every recording and both channels. Saved
comparison settings remain intact when switching recordings.

Enable **Comparison videos** to export the existing cropped-cell movie format
with three traces underneath. Select cell IDs to limit the export. Movies show
the same saved masks, background region and actual acquisition times.

The workflow runs in a separate process, waits for at least 4 GiB available RAM,
stops below that floor and applies a 4 GiB hard process-memory cap. It checks
scratch disk requirements before the large preparation step. Existing inputs
and completed runs are never overwritten.

For existing tracked movies, the **expanded_intensity** measurement module is
also available through `enabled_modules` and `module_options`. It is opt-in:

```python
from pymicroglia.measure import measure

measure(movies, output_dir="expanded-measurements",
        enabled_modules=("intensity", "expanded_intensity"),
        module_options={"expanded_intensity": {"margin_px": 3.0}})
```

This is an available option. The cancelled full-recording comparison was not
completed and does not establish accuracy on those recordings.
