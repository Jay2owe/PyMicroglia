"""Feedback marker: click a cell, press one key, done.

Opens a well's current accepted result (registered raw movie with the accepted
outlines and identity numbers drawn on top) and records what is wrong as
*marks*: the sentences Jamie already uses in review, but captured by clicking
the cells instead of typing numbers, so identities, frames and positions are
never mistyped or ambiguous.

Launch from the Motion root::

    python code/feedback_ui.py                  # pick a well from a list
    python code/feedback_ui.py --well 95_B4
    python code/feedback_ui.py --well 95_B4 --labels <candidate labels.tif>

Every mark is saved the moment it is made, to
``issues/NN_<well>/feedback/marks.jsonl``. Nothing else is ever written: no
label stack, ledger or review is touched. ``Ctrl+Z`` retracts the last mark.

Keys (the buttons on the right do the same):

    B  becomes    click the body with the WRONG number, type the right one
    S  swap       click the two cells that exchanged numbers (same frame)
    =  same cell  click one outline, go to any frame, click the other
    X  different  click two outlines that carry one number but are two cells
    P  persist    click a cell that must keep its number for the whole movie
    M  missing    click the middle of a visible cell that has no outline
    D  duplicate  click the REAL body of a number that appears twice
    /  split      click an outline that covers two cells, then drag a cut line
    J  join       click two outlines in one frame that are one cell
    K  ok         click a cell that is tracked correctly (a positive example)
    T  note       type a free-text note about the selected cells / frame
    Esc cancel    Ctrl+Z undo last mark    Left/Right frame (Shift: 5)
    Home/End first/last frame    Space play    G go to frame    +/- zoom
    R recentre    O outlines    I fill    N numbers    U unclaimed bodies
    Right-drag or left-drag pans; mouse wheel zooms around the cursor.
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import tifffile
from PIL import Image, ImageTk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adjudication_ui import (CONTRAST_PERCENTILES, WellStacks, colour_lut,  # noqa: E402
                             compose_frame, identity_colour, to_hex)
from feedback_marks import (Anchor, Mark, MarkBuilder, append_mark,  # noqa: E402
                            append_session, label_at, list_review_sets, list_wells,
                            mark_counts, next_mark_id, now_iso, read_marks,
                            read_review_set, retract_mark, well_paths)
from manual_editing_theme import DARK, apply_dark_theme  # noqa: E402
from well_workflow import find_motion_root  # noqa: E402

ZOOM_LEVELS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DEFAULT_ZOOM = 2.0
PLAY_MS = 160
SNAP_RADIUS = 3            # pixels: clicking just outside an outline still selects it
SELECT_COLOUR = "#ffe100"
PENDING_COLOUR = "#ff9d00"

KIND_COLOUR = {
    "becomes": "#ff4d4d", "swap": "#ff9d00", "same": "#3fd6ff", "join": "#3fd6ff",
    "different": "#ff5ce1", "persist": "#5cff8a", "missing": "#ffe100",
    "duplicate": "#c58bff", "split": "#ffffff", "ok": "#18c8b0", "note": "#9a9a9a",
}

# mode -> how many clicks, whether the clicks must share a frame, whether the
# click must land on an empty point, what happens after the clicks, and the
# banner text for each click.
MODES = {
    "becomes": dict(key="B", clicks=1, then="entry", banners=(
        "BECOMES: click the body that has the WRONG number",)),
    "swap": dict(key="S", clicks=2, same_frame=True, banners=(
        "SWAP: click the first of the two cells that exchanged numbers",
        "SWAP: now click the cell it swapped with (same frame)")),
    "same": dict(key="=", clicks=2, banners=(
        "SAME CELL: click the first outline (any frame)",
        "SAME CELL: move to any frame and click the other outline")),
    "different": dict(key="X", clicks=2, banners=(
        "DIFFERENT: click the first outline (any frame)",
        "DIFFERENT: move to any frame and click the outline that is NOT the same cell")),
    "persist": dict(key="P", clicks=1, banners=(
        "PERSIST: click the cell that must keep this number for the whole movie",)),
    "missing": dict(key="M", clicks=1, empty=True, then="entry", banners=(
        "MISSING: click the middle of the visible cell that has no outline",)),
    "duplicate": dict(key="D", clicks=1, banners=(
        "DUPLICATE: click the REAL body of the number that appears twice",)),
    "split": dict(key="/", clicks=1, then="line", banners=(
        "SPLIT: click the outline that covers two cells",)),
    "join": dict(key="J", clicks=2, same_frame=True, banners=(
        "JOIN: click the first outline",
        "JOIN: click the other outline in this frame that is the same cell")),
    "ok": dict(key="K", clicks=1, banners=(
        "OK: click a cell that is tracked correctly here",)),
    "note": dict(key="T", clicks=0, then="entry", banners=()),
}
KEY_TO_MODE = {spec["key"].lower(): mode for mode, spec in MODES.items()}
KEY_TO_MODE["equal"] = "same"
KEY_TO_MODE["slash"] = "split"

ENTRY_PROMPT = {
    "becomes": "Really which number?  type it, Enter",
    "missing": "Which number is it?  type it, or Enter to leave unknown",
    "note": "Note:  type, Enter",
}


def work_area(root: tk.Tk) -> tuple[int, int, int, int]:
    """Screen rectangle that excludes the Windows taskbar (left, top, right, bottom)."""
    try:
        import ctypes
        rect = ctypes.wintypes.RECT() if hasattr(ctypes, "wintypes") else None
        if rect is None:
            import ctypes.wintypes
            rect = ctypes.wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight() - 60


def load_well(paths, report=print) -> WellStacks:
    report(f"loading {paths.well}: raw {paths.raw.name}, labels {paths.labels}")
    raw = tifffile.imread(paths.raw)
    labels = tifffile.imread(paths.labels)
    unclaimed = tifffile.imread(paths.unclaimed) if paths.unclaimed else None
    if labels.shape != raw.shape:
        if raw.shape[0] == labels.shape[0] + paths.source_frame_offset \
                and raw.shape[1:] == labels.shape[1:]:
            raw = raw[paths.source_frame_offset:]
        else:
            raise SystemExit(f"{paths.well}: labels {labels.shape} do not match "
                             f"raw {raw.shape}")
    if unclaimed is not None and unclaimed.shape != labels.shape:
        unclaimed = None
    sample = raw[::5, ::3, ::3].ravel()
    low, high = np.percentile(sample, CONTRAST_PERCENTILES)
    return WellStacks(paths.well, raw, labels, unclaimed, float(low), float(high),
                      colour_lut(int(labels.max())))


class FeedbackApp:
    def __init__(self, root: tk.Tk, paths, well: WellStacks):
        self.root = root
        self.paths = paths
        self.well = well
        self.builder = MarkBuilder(paths.well, well.labels, paths.labels_sha256,
                                   paths.source_frame_offset)
        self.marks_path = paths.marks
        self.marks: list[Mark] = read_marks(self.marks_path)
        self.session_marks: list[str] = []
        self.session_started = now_iso()
        self.frames_viewed: set[int] = set()

        self.frame = 0
        self.zoom = DEFAULT_ZOOM
        self.centre = (well.shape[0] / 2.0, well.shape[1] / 2.0)
        self.selected: Anchor | None = None
        self.mode: str | None = None
        self.pending: list[Anchor] = []
        self.line: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.awaiting_line = False
        self.playing = False
        self._play_job = None
        self._press = None
        self._dragging = False
        self._slider_guard = False
        self._photo = None
        self._compose_cache: dict[tuple, np.ndarray] = {}
        self._centroid_cache: dict[int, dict[int, tuple[float, float]]] = {}
        self._view = (0, 0, 0.0, 0.0)   # pad_x, pad_y, x0, y0 of the drawn crop
        self._build()
        self._show_frame(0)
        self._fill_marks_list()
        self._set_status(f"marks are saved to {self._short(self.marks_path)}")

    # ------------------------------------------------------------------ ui
    def _build(self) -> None:
        root = self.root
        title = f"Motion feedback marks - {self.well.stem}"
        if self.paths.labels_source != "accepted":
            title += f"  [{self.paths.labels_source}]"
        root.title(title)
        apply_dark_theme(root)
        style = ttk.Style(root)
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Mono.TLabel", font=("Consolas", 10))
        style.configure("Muted.TLabel", foreground=DARK["muted"])
        style.configure("Mode.TButton", padding=(6, 6), font=("Segoe UI", 10, "bold"))
        left, top, right, bottom = work_area(root)
        root.geometry(f"{right - left - 16}x{bottom - top - 40}+{left}+{top}")
        root.minsize(1000, 650)

        self.var_outlines = tk.BooleanVar(value=True)
        self.var_fill = tk.BooleanVar(value=False)
        self.var_numbers = tk.BooleanVar(value=True)
        self.var_unclaimed = tk.BooleanVar(value=False)
        self.var_marks = tk.BooleanVar(value=True)
        self.var_brightness = tk.DoubleVar(value=1.0)
        self.var_frame_text = tk.StringVar()
        self.var_status = tk.StringVar(value="ready")
        self.var_entry = tk.StringVar()
        self.var_count = tk.StringVar()

        # banner: what to do next
        self.banner = tk.Label(root, text="", anchor="w", padx=12, pady=8,
                               font=("Segoe UI", 15, "bold"),
                               background=DARK["panel"], foreground=DARK["foreground"])
        self.banner.pack(side="top", fill="x")
        entry_row = ttk.Frame(root, padding=(12, 0, 12, 4))
        entry_row.pack(side="top", fill="x")
        self.entry_label = ttk.Label(entry_row, text="")
        self.entry_label.pack(side="left")
        self.entry = ttk.Entry(entry_row, textvariable=self.var_entry, width=40,
                               font=("Segoe UI", 12))
        self.entry.bind("<Return>", lambda e: self._entry_done())
        self.entry.bind("<Escape>", lambda e: self._cancel())
        # packed only when needed

        status = ttk.Frame(root, padding=(8, 3))
        status.pack(side="bottom", fill="x")
        ttk.Label(status, textvariable=self.var_status, style="Muted.TLabel").pack(side="left")
        ttk.Label(status, textvariable=self.var_count).pack(side="right")

        paned = ttk.Panedwindow(root, orient="horizontal")
        paned.pack(side="top", fill="both", expand=True)
        viewer = ttk.Frame(paned)
        sidebar = ttk.Frame(paned, width=440)
        paned.add(viewer, weight=4)
        paned.add(sidebar, weight=0)

        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(viewer, background="#000000", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._render())
        self.canvas.bind("<ButtonPress-1>", self._press_left)
        self.canvas.bind("<B1-Motion>", self._motion_left)
        self.canvas.bind("<ButtonRelease-1>", self._release_left)
        self.canvas.bind("<ButtonPress-3>", self._press_right)
        self.canvas.bind("<B3-Motion>", self._motion_right)
        self.canvas.bind("<MouseWheel>", self._wheel)

        controls = ttk.Frame(viewer, padding=(6, 4))
        controls.grid(row=1, column=0, sticky="ew")
        ttk.Button(controls, text="<", width=3, command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(controls, text=">", width=3, command=lambda: self._step(1)).pack(side="left", padx=(2, 6))
        self.play_button = ttk.Button(controls, text="Play", width=6, command=self._toggle_play)
        self.play_button.pack(side="left", padx=(0, 8))
        self.slider = ttk.Scale(controls, from_=0, to=self.well.n_frames - 1,
                                orient="horizontal", command=self._slider_changed)
        self.slider.pack(side="left", fill="x", expand=True)
        ttk.Label(controls, textvariable=self.var_frame_text,
                  style="Mono.TLabel").pack(side="left", padx=(8, 0))

        toggles = ttk.Frame(viewer, padding=(6, 2))
        toggles.grid(row=2, column=0, sticky="ew")
        for text, var in (("Outlines (O)", self.var_outlines), ("Fill (I)", self.var_fill),
                          ("Numbers (N)", self.var_numbers),
                          ("Unclaimed bodies (U)", self.var_unclaimed),
                          ("Marks", self.var_marks)):
            ttk.Checkbutton(toggles, text=text, variable=var,
                            command=self._overlay_changed).pack(side="left", padx=(0, 10))
        ttk.Label(toggles, text="Brightness").pack(side="left", padx=(12, 4))
        ttk.Scale(toggles, from_=0.3, to=3.0, variable=self.var_brightness,
                  orient="horizontal", length=140,
                  command=lambda v: self._overlay_changed()).pack(side="left")
        ttk.Label(toggles, text="Zoom").pack(side="left", padx=(12, 4))
        self.zoom_label = ttk.Label(toggles, text=f"{self.zoom:g}x", width=5)
        self.zoom_label.pack(side="left")
        ttk.Button(toggles, text="-", width=2, command=lambda: self._zoom_step(-1)).pack(side="left")
        ttk.Button(toggles, text="+", width=2, command=lambda: self._zoom_step(1)).pack(side="left", padx=(2, 6))
        ttk.Button(toggles, text="Recentre (R)", command=self._recentre).pack(side="left")

        # sidebar: mode buttons, selection, marks list
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=1)
        head = ttk.Frame(sidebar, padding=(10, 8, 10, 4))
        head.grid(row=0, column=0, sticky="ew")
        if self.paths.labels_source == "accepted":
            provenance = f"accepted result  {Path(self.paths.accepted_parent).name}"
        else:
            provenance = (f"review set '{self.paths.labels_source}'  "
                          f"{self.paths.labels.parent.parent.name}")
        ttk.Label(head, text=f"{self.well.stem}   outlines from {provenance}",
                  style="Muted.TLabel", wraplength=420).pack(anchor="w")
        self.lbl_selected = ttk.Label(head, text="nothing selected",
                                      font=("Segoe UI", 11, "bold"))
        self.lbl_selected.pack(anchor="w", pady=(4, 0))

        buttons = ttk.Labelframe(sidebar, text="What is wrong?  (click a cell first or after)",
                                 padding=6)
        buttons.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        buttons.columnconfigure((0, 1), weight=1)
        labels = {
            "becomes": "B  becomes (wrong number)", "swap": "S  swapped with",
            "same": "=  same cell as", "different": "X  different cell from",
            "persist": "P  keep number all movie", "missing": "M  missing outline",
            "duplicate": "D  duplicate number", "split": "/  covers two cells",
            "join": "J  join two outlines", "ok": "K  this one is fine",
            "note": "T  note",
        }
        for index, (mode, text) in enumerate(labels.items()):
            button = ttk.Button(buttons, text=text, style="Mode.TButton",
                                command=lambda m=mode: self._start_mode(m))
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        ttk.Button(buttons, text="Esc  cancel", command=self._cancel).grid(
            row=(len(labels) + 1) // 2, column=0, sticky="ew", padx=2, pady=(6, 2))
        ttk.Button(buttons, text="Ctrl+Z  undo last mark", command=self._undo).grid(
            row=(len(labels) + 1) // 2, column=1, sticky="ew", padx=2, pady=(6, 2))

        marks_box = ttk.Labelframe(sidebar, text="Marks in this well (click to jump)", padding=4)
        marks_box.grid(row=3, column=0, sticky="nsew", padx=10, pady=(4, 4))
        self.tree = ttk.Treeview(marks_box, columns=("id", "kind", "frames", "what"),
                                 show="headings", selectmode="browse", height=10)
        for column, text, width, stretch in (("id", "mark", 90, False), ("kind", "kind", 70, False),
                                             ("frames", "review fr.", 70, False),
                                             ("what", "identities", 150, True)):
            self.tree.heading(column, text=text)
            self.tree.column(column, width=width, anchor="w", stretch=stretch)
        for kind, colour in KIND_COLOUR.items():
            self.tree.tag_configure(kind, foreground=colour)
        tscroll = ttk.Scrollbar(marks_box, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tscroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        below = ttk.Frame(sidebar, padding=(10, 0, 10, 4))
        below.grid(row=4, column=0, sticky="ew")
        ttk.Button(below, text="Retract selected mark",
                   command=self._retract_selected).pack(side="left")
        self.lbl_sentence = ttk.Label(sidebar, text="", style="Muted.TLabel",
                                      wraplength=420, justify="left")
        self.lbl_sentence.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 4))
        ttk.Label(sidebar, style="Muted.TLabel", wraplength=420, font=("Segoe UI", 8), text=(
            "Click a cell to select it. Left/Right frame (Shift: 5), Home/End, Space play, "
            "G go to frame, wheel zooms, drag pans. Every mark saves itself; nothing else "
            "is written.")).grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 6))

        root.bind("<Key>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._set_banner("Click a cell, then press a key or button. Or press the key first.")

    # ------------------------------------------------------------ helpers
    def _short(self, path: Path) -> str:
        try:
            return str(path.relative_to(find_motion_root(path)))
        except Exception:
            return str(path)

    def _set_status(self, text: str) -> None:
        self.var_status.set(text)

    def _set_banner(self, text: str, colour: str | None = None) -> None:
        self.banner.configure(text=text, background=colour or DARK["panel"],
                              foreground="#000000" if colour else DARK["foreground"])

    def _frame_label(self, idx: int) -> str:
        return f"review {idx + 1} | ImageJ {idx + 1 + self.paths.source_frame_offset}"

    def _entry_active(self) -> bool:
        return self.entry.winfo_ismapped()

    # ---------------------------------------------------------- rendering
    def _compose(self) -> np.ndarray:
        focus = tuple(sorted({a.identity for a in self.pending if a.identity}
                             | ({self.selected.identity} if self.selected and self.selected.identity else set())))
        key = (self.frame, self.var_outlines.get(), self.var_fill.get(),
               self.var_unclaimed.get(), round(self.var_brightness.get(), 2), focus)
        image = self._compose_cache.get(key)
        if image is None:
            if len(self._compose_cache) > 40:
                self._compose_cache.clear()
            image = compose_frame(self.well, self.frame, key[4], key[1], key[2], key[3], focus)
            self._compose_cache[key] = image
        return image

    def _centroids(self, idx: int) -> dict[int, tuple[float, float]]:
        cached = self._centroid_cache.get(idx)
        if cached is None:
            plane = self.well.labels[idx]
            ids = [int(i) for i in np.unique(plane) if i]
            if ids:
                centres = ndi.center_of_mass(plane > 0, plane, ids)
                cached = {i: (float(c[0]), float(c[1])) for i, c in zip(ids, centres)}
            else:
                cached = {}
            self._centroid_cache[idx] = cached
        return cached

    def _render(self) -> None:
        canvas = self.canvas
        width = max(canvas.winfo_width(), 10)
        height = max(canvas.winfo_height(), 10)
        image = self._compose()
        h, w = image.shape[:2]
        z = self.zoom
        view_w, view_h = width / z, height / z
        cy, cx = self.centre
        x0 = min(max(cx - view_w / 2, 0), max(w - view_w, 0))
        y0 = min(max(cy - view_h / 2, 0), max(h - view_h, 0))
        x1 = min(x0 + view_w, w)
        y1 = min(y0 + view_h, h)
        ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
        ix1, iy1 = int(np.ceil(x1)), int(np.ceil(y1))
        crop = image[iy0:iy1, ix0:ix1]
        pil = Image.fromarray(crop).resize((max(int(crop.shape[1] * z), 1),
                                            max(int(crop.shape[0] * z), 1)), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(pil)
        pad_x = int((width - (x1 - x0) * z) / 2) if (x1 - x0) * z < width else 0
        pad_y = int((height - (y1 - y0) * z) / 2) if (y1 - y0) * z < height else 0
        self._view = (pad_x, pad_y, float(ix0), float(iy0))
        canvas.delete("all")
        canvas.create_image(pad_x, pad_y, anchor="nw", image=self._photo)
        if self.var_numbers.get() and z >= 1.0:
            self._draw_numbers(ix0, iy0, ix1, iy1)
        self._draw_overlays()
        self.var_frame_text.set(self._frame_label(self.frame))

    def _img_to_canvas(self, y: float, x: float) -> tuple[float, float]:
        pad_x, pad_y, x0, y0 = self._view
        return pad_x + (x - x0 + 0.5) * self.zoom, pad_y + (y - y0 + 0.5) * self.zoom

    def _canvas_to_img(self, cx: float, cy: float) -> tuple[int, int]:
        pad_x, pad_y, x0, y0 = self._view
        x = int(np.floor((cx - pad_x) / self.zoom + x0))
        y = int(np.floor((cy - pad_y) / self.zoom + y0))
        h, w = self.well.shape
        return int(min(max(y, 0), h - 1)), int(min(max(x, 0), w - 1))

    def _draw_numbers(self, x0: int, y0: int, x1: int, y1: int) -> None:
        font = ("Segoe UI", 9 if self.zoom < 3 else 11, "bold")
        for identity, (cy, cx) in self._centroids(self.frame).items():
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            px, py = self._img_to_canvas(cy, cx)
            colour = to_hex(identity_colour(identity))
            self.canvas.create_text(px + 1, py + 1, text=str(identity), fill="#000000", font=font)
            self.canvas.create_text(px, py, text=str(identity), fill=colour, font=font)

    def _draw_anchor(self, anchor: Anchor, colour: str, text: str, dashed: bool,
                     slot: int = 0) -> None:
        px, py = self._img_to_canvas(anchor.y, anchor.x)
        r = 7 + 2 * self.zoom + 3 * slot
        kwargs = dict(outline=colour, width=2)
        if dashed:
            kwargs["dash"] = (4, 3)
        self.canvas.create_oval(px - r, py - r, px + r, py + r, **kwargs)
        self.canvas.create_text(px + r + 2, py - r - 12 * slot, anchor="sw", text=text,
                                fill=colour, font=("Segoe UI", 9, "bold"))

    def _draw_overlays(self) -> None:
        if self.var_marks.get():
            slots: dict[tuple[int, int], int] = {}
            for mark in self.marks:
                for anchor in mark.anchors:
                    if anchor.frame_idx != self.frame or anchor.role == "control":
                        continue
                    if mark.kind in ("persist", "ok") and anchor.role != "body":
                        continue
                    key = (anchor.y // 6, anchor.x // 6)     # marks on one cell stack up
                    slot = slots.get(key, 0)
                    slots[key] = slot + 1
                    self._draw_anchor(anchor, KIND_COLOUR.get(mark.kind, "#ffffff"),
                                      mark.mark_id.rsplit("-", 1)[1], dashed=False, slot=slot)
                if mark.line and mark.anchors and mark.anchors[0].frame_idx == self.frame:
                    (y0, x0), (y1, x1) = mark.line
                    a = self._img_to_canvas(y0, x0)
                    b = self._img_to_canvas(y1, x1)
                    self.canvas.create_line(*a, *b, fill=KIND_COLOUR["split"], width=2)
        for anchor in self.pending:
            here = anchor.frame_idx == self.frame
            label = f"{anchor.identity or 'point'} @ review {anchor.review_frame}"
            self._draw_anchor(anchor, PENDING_COLOUR, label, dashed=not here)
        if self.selected and not self.pending and self.selected.frame_idx == self.frame:
            self._draw_anchor(self.selected, SELECT_COLOUR, str(self.selected.identity), False)
        if self.line:
            a = self._img_to_canvas(*self.line[0])
            b = self._img_to_canvas(*self.line[1])
            self.canvas.create_line(*a, *b, fill=KIND_COLOUR["split"], width=2)

    # ------------------------------------------------------------ frames
    def _show_frame(self, idx: int) -> None:
        idx = int(min(max(idx, 0), self.well.n_frames - 1))
        self.frame = idx
        self.frames_viewed.add(idx + 1)
        self._slider_guard = True
        self.slider.set(idx)
        self._slider_guard = False
        if self.selected and self.selected.frame_idx != idx:
            self.selected = None
            self.lbl_selected.configure(text="nothing selected")
        self._render()

    def _step(self, delta: int) -> None:
        self._show_frame(self.frame + delta)

    def _slider_changed(self, value) -> None:
        if not self._slider_guard:
            self._show_frame(int(float(value) + 0.5))

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Stop" if self.playing else "Play")
        if self.playing:
            self._tick()
        elif self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _tick(self) -> None:
        if not self.playing:
            return
        self._show_frame((self.frame + 1) % self.well.n_frames)
        self._play_job = self.root.after(PLAY_MS, self._tick)

    def _goto_prompt(self) -> None:
        self._show_entry("Go to review frame (or ij69 for an ImageJ frame):", "goto")

    # ------------------------------------------------------------ view
    def _overlay_changed(self) -> None:
        self._render()

    def _zoom_step(self, direction: int, about: tuple[int, int] | None = None) -> None:
        index = min(range(len(ZOOM_LEVELS)), key=lambda i: abs(ZOOM_LEVELS[i] - self.zoom))
        index = min(max(index + direction, 0), len(ZOOM_LEVELS) - 1)
        if about is not None:
            self.centre = (float(about[0]), float(about[1]))
        self.zoom = ZOOM_LEVELS[index]
        self.zoom_label.configure(text=f"{self.zoom:g}x")
        self._render()

    def _recentre(self) -> None:
        h, w = self.well.shape
        self.centre = (h / 2.0, w / 2.0)
        self.zoom = DEFAULT_ZOOM
        self.zoom_label.configure(text=f"{self.zoom:g}x")
        self._render()

    def _wheel(self, event) -> None:
        y, x = self._canvas_to_img(event.x, event.y)
        self._zoom_step(1 if event.delta > 0 else -1, about=(y, x))

    # ------------------------------------------------------------ mouse
    def _press_left(self, event) -> None:
        self._press = (event.x, event.y, self.centre)
        self._dragging = False
        self.canvas.focus_set()

    def _motion_left(self, event) -> None:
        if self._press is None:
            return
        x, y, centre = self._press
        if self.awaiting_line:
            self.line = (self._canvas_to_img(x, y), self._canvas_to_img(event.x, event.y))
            self._render()
            return
        if abs(event.x - x) > 3 or abs(event.y - y) > 3:
            self._dragging = True
            self.centre = (centre[0] - (event.y - y) / self.zoom,
                           centre[1] - (event.x - x) / self.zoom)
            self._render()

    def _release_left(self, event) -> None:
        if self._press is None:
            return
        press, self._press = self._press, None
        if self.awaiting_line and self.line:
            (y0, x0), (y1, x1) = self.line
            if abs(y1 - y0) + abs(x1 - x0) < 3:
                self.line = None
                self._render()
                return
            self._set_banner("SPLIT: line drawn. Enter to save, drag again to redo, Esc to cancel",
                             KIND_COLOUR["split"])
            self._set_status("press Enter to save the split with this cut line")
            return
        if self._dragging:
            return
        y, x = self._canvas_to_img(press[0], press[1])
        self._click(y, x)

    def _press_right(self, event) -> None:
        self._press = (event.x, event.y, self.centre)

    def _motion_right(self, event) -> None:
        if self._press is None:
            return
        x, y, centre = self._press
        self.centre = (centre[0] - (event.y - y) / self.zoom,
                       centre[1] - (event.x - x) / self.zoom)
        self._render()

    # ------------------------------------------------------------ clicks
    def _identity_under(self, y: int, x: int) -> int:
        return label_at(self.well.labels, self.frame, y, x, SNAP_RADIUS)

    def _click(self, y: int, x: int) -> None:
        if self._entry_active():
            return
        spec = MODES.get(self.mode) if self.mode else None
        if spec and spec.get("empty"):
            if int(self.well.labels[self.frame, y, x]):
                self._set_status(f"that point already has outline "
                                 f"{int(self.well.labels[self.frame, y, x])}; "
                                 f"click the middle of a cell with NO outline")
                return
            anchor = self.builder.point_anchor(self.frame, y, x)
        else:
            identity = self._identity_under(y, x)
            if not identity:
                self.selected = None
                self.lbl_selected.configure(text="nothing selected (no outline here)")
                self._set_status("no outline under the click; M marks a missing cell")
                self._render()
                return
            anchor = self.builder.cell_anchor(self.frame, identity, click=(y, x))
        if spec is None:
            self.selected = anchor
            self.lbl_selected.configure(
                text=f"selected {anchor.identity} at {self._frame_label(self.frame)}  "
                     f"({anchor.area} px)")
            self._set_status("now press a key or button to say what is wrong with it")
            self._render()
            return
        if spec.get("same_frame") and self.pending and \
                self.pending[0].frame_idx != anchor.frame_idx:
            self._set_status(f"{self.mode}: both cells must be clicked in the same frame "
                             f"(first was review {self.pending[0].review_frame})")
            return
        if self.pending and anchor.identity and anchor.identity == self.pending[0].identity \
                and anchor.frame_idx == self.pending[0].frame_idx and self.mode != "duplicate":
            self._set_status("that is the same outline you already clicked")
            return
        self.pending.append(anchor)
        self._advance_mode()

    # ------------------------------------------------------------ modes
    def _start_mode(self, mode: str) -> None:
        if self._entry_active():
            return
        self._cancel(quiet=True)
        self.mode = mode
        spec = MODES[mode]
        if spec["clicks"] == 0:
            self._show_entry(ENTRY_PROMPT[mode], mode)
            return
        if self.selected is not None and not spec.get("empty"):
            self.pending = [self.selected]
            self.selected = None
        self._advance_mode()

    def _advance_mode(self) -> None:
        spec = MODES[self.mode]
        done = len(self.pending)
        if done < spec["clicks"]:
            self._set_banner(spec["banners"][done], KIND_COLOUR[self.mode])
            self._set_status("Esc cancels")
            self._render()
            return
        follow = spec.get("then")
        if follow == "entry":
            self._render()
            self._show_entry(ENTRY_PROMPT[self.mode], self.mode)
        elif follow == "line":
            self.awaiting_line = True
            self.line = None
            self._set_banner("SPLIT: drag a line across the cell where it should be cut, "
                             "or press Enter to save without a line", KIND_COLOUR["split"])
            self._render()
        else:
            self._commit()

    def _show_entry(self, prompt: str, purpose: str) -> None:
        self.entry_purpose = purpose
        self.var_entry.set("")
        self.entry_label.configure(text=prompt + "   ")
        self.entry_label.pack(side="left")
        self.entry.pack(side="left")
        self.entry.focus_set()
        self._set_banner(prompt, KIND_COLOUR.get(purpose))

    def _hide_entry(self) -> None:
        self.entry.pack_forget()
        self.entry_label.configure(text="")
        self.canvas.focus_set()

    def _entry_done(self) -> None:
        text = self.var_entry.get().strip()
        purpose = self.entry_purpose
        if purpose == "goto":
            self._hide_entry()
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                frame = int(digits)
                if text.lower().startswith(("ij", "i")):
                    frame -= self.paths.source_frame_offset
                self._show_frame(frame - 1)
            self._reset_banner()
            return
        if purpose == "note":
            if not text:
                self._cancel()
                return
            self._hide_entry()
            about = [self.selected] if self.selected else []
            mark = self.builder.note(self._next_id(), self.frame, text, about)
            self._save(mark)
            return
        identity = None
        if text:
            digits = "".join(ch for ch in text if ch.isdigit())
            if not digits:
                self._set_status("type a number (the identity as shown on screen)")
                return
            identity = int(digits)
            if identity not in self.builder.presence.frames:
                self._set_status(f"identity {identity} does not exist in this stack; "
                                 f"try again or Esc")
                return
        if purpose == "becomes" and identity is None:
            self._set_status("type the number this body should have")
            return
        if purpose == "becomes" and self.pending and identity == self.pending[0].identity:
            self._set_status(f"that body already carries {identity}; type the number it "
                             f"SHOULD have, or Esc")
            return
        self._hide_entry()
        self.entry_value = identity
        self._commit()

    def _next_id(self) -> str:
        return next_mark_id(self.marks_path, self.well.stem)

    def _commit(self) -> None:
        mode = self.mode
        builder = self.builder
        mark_id = self._next_id()
        pending = self.pending
        try:
            if mode == "becomes":
                mark = builder.becomes(mark_id, pending[0], self.entry_value)
            elif mode == "swap":
                mark = builder.swap(mark_id, pending[0], pending[1])
            elif mode in ("same", "join"):
                mark = builder.same(mark_id, pending[0], pending[1])
            elif mode == "different":
                mark = builder.different(mark_id, pending[0], pending[1])
            elif mode == "persist":
                mark = builder.persist(mark_id, pending[0].identity, "persist", self.frame)
            elif mode == "ok":
                mark = builder.persist(mark_id, pending[0].identity, "ok", self.frame)
            elif mode == "missing":
                mark = builder.missing(mark_id, self.frame, pending[0].y, pending[0].x,
                                       getattr(self, "entry_value", None))
            elif mode == "duplicate":
                mark = builder.duplicate(mark_id, pending[0])
            elif mode == "split":
                mark = builder.split(mark_id, pending[0], self.line)
            else:
                raise ValueError(f"unknown mode {mode}")
        except Exception as error:      # never lose the session over one mark
            self._set_status(f"could not build the mark: {error}")
            self._cancel(quiet=True)
            return
        self._save(mark)

    def _save(self, mark: Mark) -> None:
        try:
            append_mark(self.marks_path, mark)
        except OSError as error:
            self._set_banner(f"NOT SAVED: {error}", KIND_COLOUR["becomes"])
            self._set_status("the marks file could not be written; try the mark again")
            self._cancel(quiet=True)
            return
        self.marks.append(mark)
        self.session_marks.append(mark.mark_id)
        self._fill_marks_list(select=mark.mark_id)
        self._cancel(quiet=True)
        short = mark.mark_id.rsplit("-", 1)[1]
        warn = f"   (unresolved: {'; '.join(mark.unresolved)})" if mark.unresolved else ""
        self._set_banner(f"Saved {short}: {mark.sentence}"[:160], None)
        self._set_status(f"{short} saved{warn}. Ctrl+Z undoes it.")
        self.lbl_sentence.configure(text=mark.sentence)

    def _cancel(self, quiet: bool = False) -> None:
        self.mode = None
        self.pending = []
        self.line = None
        self.awaiting_line = False
        self.entry_value = None
        if self._entry_active():
            self._hide_entry()
        if not quiet:
            self._reset_banner()
            self._set_status("cancelled")
        self._render()

    def _reset_banner(self) -> None:
        self._set_banner("Click a cell, then press a key or button. Or press the key first.")

    def _undo(self) -> None:
        if not self.session_marks:
            self._set_status("nothing from this session to undo; use 'Retract selected' "
                             "for older marks")
            return
        mark_id = self.session_marks.pop()
        self._retract(mark_id, "undo")

    def _retract_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            self._set_status("select a mark in the list first")
            return
        self._retract(selection[0], "retracted from list")

    def _retract(self, mark_id: str, reason: str) -> None:
        retract_mark(self.marks_path, mark_id, reason)
        self.marks = [m for m in self.marks if m.mark_id != mark_id]
        self.session_marks = [m for m in self.session_marks if m != mark_id]
        self._fill_marks_list()
        self._set_status(f"{mark_id} retracted (kept in the file as history)")
        self.lbl_sentence.configure(text="")
        self._render()

    # ------------------------------------------------------------ list
    def _fill_marks_list(self, select: str | None = None) -> None:
        self.tree.delete(*self.tree.get_children())
        for mark in self.marks:
            frames = mark.frames()
            body = [a.review_frame for a in mark.anchors if a.role in ("body", "point")]
            shown = f"{min(body)}" if body and min(body) == max(body) else \
                (f"{min(body)}-{max(body)}" if body else (f"{frames[0]}-{frames[-1]}" if frames else ""))
            what = " ".join(str(i) for i in mark.identities)
            if mark.kind == "becomes":
                what = f"{mark.expected_identity} -> {mark.anchors[0].identity}"
            elif mark.kind == "note":
                what = mark.note[:40]
            self.tree.insert("", "end", iid=mark.mark_id, tags=(mark.kind,),
                             values=(mark.mark_id.rsplit("-", 1)[1], mark.kind, shown, what))
        if select and self.tree.exists(select):
            self.tree.selection_set(select)
            self.tree.see(select)
        active = len(self.marks)
        self.var_count.set(f"{active} active mark{'s' if active != 1 else ''}, "
                           f"{len(self.session_marks)} this session")

    def _tree_selected(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        mark = next((m for m in self.marks if m.mark_id == selection[0]), None)
        if mark is None:
            return
        self.lbl_sentence.configure(text=mark.sentence + (f"  Note: {mark.note}" if mark.note and mark.kind != "note" else ""))
        body = [a for a in mark.anchors if a.role in ("body", "point", "side")] or mark.anchors
        if body:
            anchor = body[0]
            self.centre = (float(anchor.y), float(anchor.x))
            if self.zoom < 3.0:
                self.zoom = 4.0
                self.zoom_label.configure(text=f"{self.zoom:g}x")
            self._show_frame(anchor.frame_idx)

    # ------------------------------------------------------------ keys
    def _on_key(self, event) -> None:
        if self._entry_active():
            return
        key = event.keysym
        lower = key.lower()
        ctrl = bool(event.state & 0x4)
        shift = bool(event.state & 0x1)
        if ctrl and lower == "z":
            self._undo()
        elif key == "Escape":
            self._cancel()
        elif key == "Return":
            if self.awaiting_line:
                self._commit()
        elif key == "Left":
            self._step(-5 if shift else -1)
        elif key == "Right":
            self._step(5 if shift else 1)
        elif key == "Home":
            self._show_frame(0)
        elif key == "End":
            self._show_frame(self.well.n_frames - 1)
        elif key == "Prior":
            self._step(-10)
        elif key == "Next":
            self._step(10)
        elif key == "space":
            self._toggle_play()
        elif key in ("plus", "KP_Add"):
            self._zoom_step(1)
        elif key in ("minus", "KP_Subtract"):
            self._zoom_step(-1)
        elif lower == "r":
            self._recentre()
        elif lower == "g":
            self._goto_prompt()
        elif lower == "o":
            self.var_outlines.set(not self.var_outlines.get()); self._render()
        elif lower == "i":
            self.var_fill.set(not self.var_fill.get()); self._render()
        elif lower == "n":
            self.var_numbers.set(not self.var_numbers.get()); self._render()
        elif lower == "u":
            self.var_unclaimed.set(not self.var_unclaimed.get()); self._render()
        elif lower in KEY_TO_MODE and not ctrl:
            self._start_mode(KEY_TO_MODE[lower])

    # ------------------------------------------------------------ close
    def _close(self) -> None:
        self.playing = False
        try:
            append_session(self.marks_path, self.well.stem, self.paths.labels_sha256,
                           self.frames_viewed, self.session_marks,
                           self.session_started, now_iso())
        finally:
            self.root.destroy()


# ------------------------------------------------------------------ launch

def choose_well(root_dir: Path, review_set: str | None = None) -> str | None:
    wells = list_wells(root_dir)
    if not wells:
        raise SystemExit("no wells with WELL_BASELINE.json under issues/")
    counts = mark_counts(root_dir)
    in_set: set[str] = set()
    heading = "outlines: each well's accepted result"
    if review_set:
        data = read_review_set(root_dir, review_set)
        in_set = set(data["wells"])
        heading = f"outlines: review set '{review_set}' - {data.get('description') or data['source']}"
    picker = tk.Tk()
    picker.title("Motion feedback - choose a well")
    apply_dark_theme(picker)
    chosen: list[str] = []
    ttk.Label(picker, text="Which recording do you want to give feedback on?",
              font=("Segoe UI", 12, "bold"), padding=(12, 12, 12, 2)).pack(anchor="w")
    ttk.Label(picker, text=heading, foreground=DARK["muted"], wraplength=420,
              padding=(14, 0, 12, 8)).pack(anchor="w")
    for well_dir in wells:
        stem = well_dir.name.split("_", 1)[1]
        done = counts.get(stem, 0)
        note = f"{done} mark{'s' if done != 1 else ''}" if done else "not reviewed yet"
        missing = review_set and stem not in in_set
        text = f"{stem}      {note}" + ("   (not in this set)" if missing else "")
        button = ttk.Button(picker, text=text, width=42,
                            command=lambda s=stem: (chosen.append(s), picker.destroy()))
        button.pack(padx=16, pady=3)
        if missing:
            button.state(["disabled"])
    ttk.Label(picker, text="").pack(pady=4)
    picker.mainloop()
    return chosen[0] if chosen else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mark tracking mistakes by clicking cells.")
    parser.add_argument("--well", help="recording stem, e.g. 95_B4 (a list is offered if omitted)")
    parser.add_argument("--labels", type=Path, help="mark a candidate label stack instead of "
                        "the accepted one (the marks record its fingerprint)")
    parser.add_argument("--raw", type=Path, help="registered raw movie (default: registered inputs/<well>.tif)")
    parser.add_argument("--frame", type=int, default=1, help="review frame to open on")
    parser.add_argument("--review-set", help="open this well's stack from a named review "
                        "set instead of its accepted result")
    args = parser.parse_args(argv)
    root_dir = find_motion_root(Path(__file__).resolve().parent)
    if args.review_set and args.review_set not in list_review_sets(root_dir):
        raise SystemExit(f"no review set {args.review_set!r}; available: "
                         f"{', '.join(list_review_sets(root_dir)) or 'none'}")
    well = args.well or choose_well(root_dir, args.review_set)
    if not well:
        return 0
    paths = well_paths(root_dir, well, labels=args.labels, raw=args.raw,
                       verify=args.labels is None, review_set=args.review_set)
    stacks = load_well(paths)
    root = tk.Tk()
    app = FeedbackApp(root, paths, stacks)
    app._show_frame(args.frame - 1)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
