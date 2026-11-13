"""Feedback marks inside the manual track editor.

Adds a marks panel to ``ManualEditingApp``: the well's marks listed with their
status on the current labels, drawn on the image, click-to-jump, click-to-swap,
recording new marks by clicking, and fixing marks through ``manual_feedback``.
Tk is reached through the app (``app.tk``/``app.ttk``) so this module imports
without a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from feedback_marks import label_at
from manual_feedback import (KIND_COLOUR, SNAP_RADIUS, MarkSource, apply_plan,
                             build_mark, check_marks, describe_operation,
                             load_marks, mark_builder, mark_frame, mark_source,
                             next_id, plan_from_marks, retract, save_mark)

PENDING_COLOUR = "#ff9d00"
STATUS_COLOUR = {"violated": "#ff8a80", "satisfied": "#7ee08a",
                 "undecidable": "#d0c070", "not-testable": "#9a9a9a"}

# kind -> (clicks, same frame, first click may be empty, follow-up question)
CLICK_RULES = {
    "swap": (2, True, False, None),
    "becomes": (1, False, False, "number"),
    "same": (2, False, False, None),
    "different": (2, False, False, None),
    "join": (2, True, False, None),
    "persist": (1, False, False, None),
    "ok": (1, False, False, None),
    "missing": (1, False, True, "optional_number"),
    "duplicate": (1, False, False, None),
    "split": (1, False, False, None),
    "note": (0, False, False, "text"),
}
PROMPTS = {
    "swap": ("click the first of the two cells that exchanged numbers",
             "click the cell it swapped with (same frame)"),
    "becomes": ("click the body that carries the WRONG number",),
    "same": ("click the first outline (any frame)",
             "move to any frame and click the other outline of the same cell"),
    "different": ("click the first outline (any frame)",
                  "move to any frame and click the outline that is NOT the same cell"),
    "join": ("click the first outline", "click the other outline of the same cell here"),
    "persist": ("click the cell that must keep one number for the whole movie",),
    "ok": ("click a cell that is tracked correctly here",),
    "missing": ("click the middle of the visible cell that has no outline",),
    "duplicate": ("click the REAL body of the number that appears twice",),
    "split": ("click the outline that covers two cells",),
}
SWAP_EDIT = "swap_edit"


@dataclass
class ClickRecorder:
    """Collects the clicks one mark (or one click-to-swap) needs."""
    kind: str
    clicks: list[tuple[int, int, int]] = field(default_factory=list)
    identities: list[int] = field(default_factory=list)

    @property
    def rule(self):
        return CLICK_RULES["swap" if self.kind == SWAP_EDIT else self.kind]

    @property
    def done(self) -> bool:
        return len(self.clicks) >= self.rule[0]

    @property
    def prompt(self) -> str:
        texts = PROMPTS.get("swap" if self.kind == SWAP_EDIT else self.kind, ())
        index = min(len(self.clicks), max(len(texts) - 1, 0))
        title = "SWAP NOW" if self.kind == SWAP_EDIT else self.kind.upper()
        return f"{title}: {texts[index]}  (Esc cancels)" if texts else title

    def add(self, labels: np.ndarray, frame: int, y: int, x: int) -> str:
        """Record one click (one-based frame); returns an error or ``""``."""
        count, same_frame, empty_ok, _ask = self.rule
        # A missing cell sits on uncovered background, so no snapping there.
        radius = 0 if empty_ok and not self.clicks else SNAP_RADIUS
        identity = label_at(labels, frame - 1, y, x, radius=radius)
        if not identity and not (empty_ok and not self.clicks):
            return "That click is not on an outline; try again"
        if identity and empty_ok and not self.clicks and self.kind == "missing":
            return "That point already has an outline; click the uncovered cell"
        if same_frame and self.clicks and frame != self.clicks[0][0]:
            return f"Stay on frame {self.clicks[0][0]} for the second click"
        if identity and identity in self.identities and count > 1:
            return f"Cell {identity} is already chosen; click the other one"
        self.clicks.append((frame, y, x))
        self.identities.append(identity)
        return ""


class MarksPanel:
    """Sidebar section and canvas overlay for the well's feedback marks."""

    def __init__(self, app, parent):
        self.app = app
        tk, ttk = app.tk, app.ttk
        self.source: MarkSource = mark_source(app.controller.bundle)
        self.marks = []
        self.results: dict[str, dict] = {}
        self.recorder: ClickRecorder | None = None
        self.show = tk.BooleanVar(value=True)
        self.save_swaps = tk.BooleanVar(value=True)
        self.kind = tk.StringVar(value="swap")
        self.summary = tk.StringVar()
        self.banner = tk.StringVar()

        frame = ttk.LabelFrame(parent, text="Feedback marks", padding=6)
        frame.pack(fill="x", pady=(0, 10))
        self.frame = frame
        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.summary).pack(side="left")
        ttk.Checkbutton(top, text="Show on image", variable=self.show,
                        command=app._render).pack(side="right")
        self.tree = ttk.Treeview(frame, columns=("kind", "frame", "status"),
                                 show="tree headings", height=7, selectmode="browse")
        self.tree.heading("#0", text="Mark")
        for column, title, width in (("kind", "Kind", 80), ("frame", "Frame", 55),
                                     ("status", "Now", 90)):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, stretch=False)
        self.tree.column("#0", width=90, stretch=False)
        for status, colour in STATUS_COLOUR.items():
            self.tree.tag_configure(status, foreground=colour)
        self.tree.pack(fill="x", pady=(4, 4))
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._jump_to_selected())

        # REGRESSION GUARD: right-packed buttons fall off a narrow sidebar;
        # keep every control left-aligned.
        fixes = ttk.Frame(frame)
        fixes.pack(fill="x")
        ttk.Button(fixes, text="Fix selected mark",
                   command=self.fix_selected).pack(side="left")
        ttk.Button(fixes, text="Fix all fixable…",
                   command=self.fix_all).pack(side="left", padx=(6, 0))
        ttk.Button(fixes, text="Retract mark",
                   command=self.retract_selected).pack(side="left", padx=(6, 0))

        swap = ttk.Frame(frame)
        swap.pack(fill="x", pady=(6, 0))
        ttk.Button(swap, text="Click-to-swap (S)",
                   command=self.start_swap).pack(side="left")
        ttk.Checkbutton(swap, text="save swap as a mark",
                        variable=self.save_swaps).pack(side="left", padx=(6, 0))

        record = ttk.Frame(frame)
        record.pack(fill="x", pady=(6, 0))
        ttk.Combobox(record, textvariable=self.kind, values=tuple(CLICK_RULES),
                     state="readonly", width=10).pack(side="left")
        ttk.Button(record, text="Record mark",
                   command=self.start_record).pack(side="left", padx=(4, 0))
        self.choose_button = ttk.Button(record, text="Choose marks file…",
                                        command=self.choose_marks_file)
        self.choose_button.pack(side="left", padx=(6, 0))
        ttk.Label(frame, textvariable=self.banner, foreground=PENDING_COLOUR,
                  wraplength=560).pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------ state
    def refresh(self) -> None:
        """Reload marks and check each one on the editor's current labels."""
        self.source = MarkSource(
            self.source.stem, self.source.marks_path,
            mark_source(self.app.controller.bundle).source_frame_offset,
            self.source.well_dir)
        self.marks = load_marks(self.source)
        checked = check_marks(self.marks, self.app.labels, self.source)
        self.results = {r["mark_id"]: r for r in checked["results"]}
        summary = checked["summary"]
        if not self.source.available:
            self.summary.set(f"No feedback folder for {self.source.stem or 'this bundle'}")
        else:
            self.summary.set(
                f"{self.source.stem}: {len(self.marks)} marks · "
                f"{summary['satisfied']} hold · {summary['violated']} broken")
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for mark in sorted(self.marks, key=lambda m: (_frame(m), m.mark_id)):
            status = self.results[mark.mark_id]["status"]
            self.tree.insert("", "end", iid=mark.mark_id,
                             text=mark.mark_id.rsplit("-", 1)[-1],
                             values=(mark.kind, _frame(mark), status), tags=(status,))
        if selected and self.tree.exists(selected[0]):
            self.tree.selection_set(selected[0])

    def statuses(self) -> dict[str, str]:
        return {mark_id: r["status"] for mark_id, r in self.results.items()}

    def report_changes(self, before: dict[str, str]) -> str:
        """After a save: which marks started or stopped holding."""
        after = self.statuses()
        broken = sorted(m for m, s in before.items()
                        if s == "satisfied" and after.get(m) == "violated")
        fixed = sorted(m for m, s in before.items()
                       if s != "satisfied" and after.get(m) == "satisfied")
        if broken:
            from tkinter import messagebox
            messagebox.showwarning(
                "Marks broken by this edit",
                "These marks held before the edit and are now broken:\n\n"
                + "\n".join(f"{m}: {self._sentence(m)}" for m in broken)
                + "\n\nUndo (Ctrl+Z) returns to the previous checkpoint.",
                parent=self.app.root)
        parts = []
        if fixed:
            parts.append(f"{len(fixed)} mark(s) now hold")
        if broken:
            parts.append(f"{len(broken)} mark(s) broken")
        return "; ".join(parts)

    def _sentence(self, mark_id: str) -> str:
        return next((m.sentence for m in self.marks if m.mark_id == mark_id), "")

    def _selected_mark(self):
        selected = self.tree.selection()
        return next((m for m in self.marks if selected and m.mark_id == selected[0]), None)

    # ------------------------------------------------------------ navigation
    def _jump_to_selected(self) -> None:
        mark = self._selected_mark()
        if mark is None:
            return
        frame = _frame(mark)
        app = self.app
        app.frame_index = min(len(app.labels) - 1, max(0, frame - 1))
        app.frame_scale.set(app.frame_index + 1)
        found = self.results.get(mark.mark_id, {}).get("candidate_labels", [])
        here = [value for anchor, value in zip(mark.anchors, found)
                if value and anchor.review_frame == frame]
        app.selected = set(here)
        app._refresh_selection()
        app._render()
        anchor = next((a for a in mark.anchors if a.review_frame == frame), None)
        if anchor is not None:
            self._centre_on(anchor.y, anchor.x)
        app.status.set(f"{mark.mark_id}: {mark.sentence}")

    def _centre_on(self, y: int, x: int) -> None:
        app = self.app
        scale = app._zoom_factor()
        width = max(1, round(app.labels.shape[2] * scale))
        height = max(1, round(app.labels.shape[1] * scale))
        view_w = max(1, app.canvas.winfo_width())
        view_h = max(1, app.canvas.winfo_height())
        app.canvas.xview_moveto(max(0.0, x * scale - view_w / 2) / width)
        app.canvas.yview_moveto(max(0.0, y * scale - view_h / 2) / height)

    # ------------------------------------------------------------ drawing
    def draw(self, canvas, scale: float) -> None:
        frame = self.app.frame_index + 1
        radius = max(6.0, 3.0 * scale)
        if self.show.get():
            chosen = self.tree.selection()
            for mark in self.marks:
                status = self.results.get(mark.mark_id, {}).get("status", "")
                for anchor in mark.anchors:
                    if anchor.review_frame != frame or anchor.role == "control":
                        continue
                    if mark.kind in ("persist", "ok") and anchor.role != "body":
                        continue
                    cx, cy = (anchor.x + 0.5) * scale, (anchor.y + 0.5) * scale
                    wide = 3 if chosen and chosen[0] == mark.mark_id else (
                        2 if status == "violated" else 1)
                    r = radius + (3 if wide == 3 else 0)
                    canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                       outline=KIND_COLOUR.get(mark.kind, "#ffffff"),
                                       width=wide, dash=() if status == "violated" else (3, 2))
                    canvas.create_text(cx + r + 2, cy - r, anchor="sw",
                                       text=mark.mark_id.rsplit("-", 1)[-1],
                                       fill=KIND_COLOUR.get(mark.kind, "#ffffff"))
        if self.recorder is not None:
            for click_frame, y, x in self.recorder.clicks:
                cx, cy = (x + 0.5) * scale, (y + 0.5) * scale
                canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                   outline=PENDING_COLOUR, width=2,
                                   dash=() if click_frame == frame else (4, 3))

    # ------------------------------------------------------------ clicking
    def start_swap(self) -> None:
        self.recorder = ClickRecorder(SWAP_EDIT)
        self.banner.set(self.recorder.prompt)
        self.app._render()

    def start_record(self) -> None:
        if not self._require_source():
            return
        kind = self.kind.get()
        self.recorder = ClickRecorder(kind)
        if self.recorder.done:
            self._finish()
            return
        self.banner.set(self.recorder.prompt)
        self.app._render()

    def cancel(self) -> None:
        if self.recorder is not None:
            self.recorder = None
            self.banner.set("")
            self.app._render()
            self.app.status.set("Click mode cancelled")

    def handle_click(self, frame: int, y: int, x: int) -> bool:
        """Consume a canvas click while a click mode is active."""
        if self.recorder is None:
            return False
        error = self.recorder.add(self.app.labels, frame, y, x)
        if error:
            self.app.status.set(error)
            return True
        if self.recorder.done:
            self._finish()
        else:
            self.banner.set(self.recorder.prompt)
            self.app._render()
        return True

    def _finish(self) -> None:
        recorder, self.recorder = self.recorder, None
        self.banner.set("")
        self.app._render()
        if recorder.kind == SWAP_EDIT:
            self._swap_now(recorder)
        else:
            self._record(recorder)

    def _swap_now(self, recorder: ClickRecorder) -> None:
        app = self.app
        frame = recorder.clicks[0][0]
        first, second = recorder.identities
        answer = self._ask_swap(first, second, frame)
        if answer is None:
            app.status.set("Swap cancelled")
            return
        end, keep_mark = answer
        mark = None
        if keep_mark and self.source.available:
            try:
                builder = mark_builder(self.source, app.labels)
                mark = build_mark(builder, next_id(self.source), "swap", recorder.clicks)
            except ValueError as error:
                app.status.set(f"Swap not saved as a mark: {error}")
        before = app.controller
        app._run_saved_action(
            lambda target: before.swap_identities(
                first, second, frame, target, end),
            f"Swapped identities {first} and {second} from frame {frame} to {end}")
        if mark is not None and app.controller is not before:
            save_mark(self.source, mark)
            self.refresh()
            app._render()
            app.status.set(app.status.get() + f"; saved as {mark.mark_id}")

    def _ask_swap(self, first: int, second: int, frame: int):
        app = self.app
        tk, ttk = app.tk, app.ttk
        last = len(app.labels)
        window = tk.Toplevel(app.root)
        window.title("Swap two cells")
        window.transient(app.root)
        body = ttk.Frame(window, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=f"Swap identities {first} and {second}").grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(body, text=f"from frame {frame} through frame").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        end = tk.StringVar(value=str(last))
        entry = ttk.Entry(body, textvariable=end, width=6)
        entry.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(body, text=f"(last frame is {last})").grid(
            row=1, column=2, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(body, text="Also save as a feedback mark",
                        variable=self.save_swaps).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        result: list = []

        def accept(_event=None) -> None:
            try:
                value = int(end.get())
            except ValueError:
                return
            if frame <= value <= last:
                result.append((value, bool(self.save_swaps.get())))
                window.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=3, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Swap", command=accept).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(
            side="left", padx=(6, 0))
        window.bind("<Return>", accept)
        window.bind("<Escape>", lambda _e: window.destroy())
        entry.focus_set()
        entry.select_range(0, "end")
        window.grab_set()
        app.root.wait_window(window)
        return result[0] if result else None

    def _record(self, recorder: ClickRecorder) -> None:
        from tkinter import messagebox, simpledialog

        app = self.app
        ask = recorder.rule[3]
        number = None
        note = ""
        if ask in ("number", "optional_number"):
            number = simpledialog.askinteger(
                "Which number?",
                "Which identity is this really?" if ask == "number"
                else "Which identity is it? (leave empty if unknown)",
                parent=app.root, minvalue=1)
            if ask == "number" and number is None:
                app.status.set("Mark cancelled")
                return
        if ask == "text":
            note = simpledialog.askstring("Note", "Note:", parent=app.root) or ""
            if not note.strip():
                app.status.set("Mark cancelled")
                return
            recorder.clicks = recorder.clicks or [(app.frame_index + 1, 0, 0)]
        try:
            builder = mark_builder(self.source, app.labels)
            mark = build_mark(builder, next_id(self.source), recorder.kind,
                              recorder.clicks, number, note)
        except ValueError as error:
            messagebox.showerror("Cannot make this mark", str(error), parent=app.root)
            return
        text = mark.sentence + (
            f"\n\nUnresolved: {'; '.join(mark.unresolved)}" if mark.unresolved else "")
        if not messagebox.askyesno("Save mark", text, parent=app.root):
            app.status.set("Mark not saved")
            return
        save_mark(self.source, mark)
        self.refresh()
        self.tree.selection_set(mark.mark_id)
        app._render()
        app.status.set(f"Saved {mark.mark_id}: {mark.sentence}")

    # ------------------------------------------------------------ fixing
    def _plan(self, mark_ids=None):
        app = self.app
        return plan_from_marks(app.controller.bundle, app.labels, self.marks,
                               self.source, mark_ids)

    def fix_selected(self) -> None:
        from tkinter import messagebox

        app = self.app
        mark = self._selected_mark()
        if mark is None:
            app.status.set("Select a mark first")
            return
        plan = self._plan([mark.mark_id])
        step = plan["steps"][0]
        if step["action"] == "none":
            app.status.set(f"{mark.mark_id}: {step['reason'] or 'nothing to change'}")
            return
        if step["action"] == "manual":
            tool = step["workspace"]
            if tool and messagebox.askyesno(
                    "Needs drawing", f"{mark.sentence}\n\n{step['reason']}.\n\n"
                    "Open that tool on this cell now?", parent=app.root):
                self._open_workspace(mark, tool)
            elif not tool:
                messagebox.showinfo("Not fixable automatically",
                                    f"{mark.sentence}\n\n{step['reason']}.",
                                    parent=app.root)
            return
        words = describe_operation(step["operation"], len(app.labels))
        caution = ("\n\nCheck this one: " + step["reason"] + "."
                   if step["confidence"] == "check" else "")
        if not app._confirm("Fix from mark", f"{mark.sentence}\n\nEdit: {words}.{caution}"):
            return
        app._run_saved_action(
            lambda target: apply_plan(app.controller, plan, target),
            f"Fixed {mark.mark_id} ({words})")

    def _open_workspace(self, mark, tool: str) -> None:
        app = self.app
        frame = _frame(mark)
        found = self.results.get(mark.mark_id, {}).get("candidate_labels", [])
        app.selected = {value for value in found[:1] if value}
        app.range_start.set(str(frame))
        app.range_end.set(str(frame))
        app.frame_index = frame - 1
        app.frame_scale.set(frame)
        app._refresh_selection()
        app._render()
        if tool == "force_split":
            app._force_split_selected()
        else:
            app._add_new_cells()

    def fix_all(self) -> None:
        app = self.app
        if not self.marks:
            app.status.set("No marks to fix")
            return
        app.root.configure(cursor="watch")
        app.root.update_idletasks()
        try:
            plan = self._plan()
        finally:
            app.root.configure(cursor="")
        self._plan_window(plan)

    def _plan_window(self, plan: dict) -> None:
        app = self.app
        tk, ttk = app.tk, app.ttk
        window = tk.Toplevel(app.root)
        window.title("Fix marks")
        window.geometry("1250x480")
        window.transient(app.root)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)
        summary = plan["summary"]
        ttk.Label(frame, wraplength=960, text=(
            f"{summary['edits']} edit(s) proposed, {summary['manual']} mark(s) need "
            f"drawing or a decision. Marks holding: {summary['before']['satisfied']} now, "
            f"{summary['after']['satisfied']} after these edits. No mark that holds now "
            "is broken by them. Deselect any edit you do not want; double-click a row "
            "to look at it.")).pack(anchor="w")
        tree = ttk.Treeview(frame, columns=("frame", "edit", "confidence", "why"),
                            show="tree headings", selectmode="extended")
        tree.heading("#0", text="Mark")
        for column, title, width in (("frame", "Frame", 55), ("edit", "Edit", 260),
                                     ("confidence", "Confidence", 80),
                                     ("why", "Why", 480)):
            tree.heading(column, text=title)
            tree.column(column, width=width, stretch=column == "why")
        tree.column("#0", width=90, stretch=False)
        tree.tag_configure("check", foreground="#ffd27f")
        edits = [s for s in plan["steps"] if s["action"] == "edit"]
        for step in edits:
            tree.insert("", "end", iid=step["mark_id"],
                        text=step["mark_id"].rsplit("-", 1)[-1],
                        values=(step["frame"],
                                describe_operation(step["operation"], len(app.labels)),
                                step["confidence"], step["reason"]),
                        tags=(step["confidence"],))
        tree.selection_set([s["mark_id"] for s in edits])
        tree.pack(fill="both", expand=True, pady=(8, 8))

        def look(_event=None) -> None:
            chosen = tree.focus()
            if chosen and self.tree.exists(chosen):
                self.tree.selection_set(chosen)

        def apply() -> None:
            chosen = list(tree.selection())
            if not chosen:
                return
            window.destroy()
            app._run_saved_action(
                lambda target: apply_plan(app.controller, plan, target, chosen),
                f"Applied {len(chosen)} edit(s) from feedback marks")

        tree.bind("<Double-1>", look)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Apply selected edits", command=apply).pack(side="left")
        ttk.Button(buttons, text="Close", command=window.destroy).pack(side="right")
        if not edits:
            ttk.Label(buttons, text="Nothing can be fixed automatically.").pack(
                side="left", padx=(10, 0))

    def retract_selected(self) -> None:
        from tkinter import simpledialog

        app = self.app
        mark = self._selected_mark()
        if mark is None:
            app.status.set("Select a mark first")
            return
        reason = simpledialog.askstring(
            "Retract mark", f"{mark.sentence}\n\nWhy retract it?", parent=app.root)
        if reason is None:
            return
        retract(self.source, mark.mark_id, reason.strip() or "retracted in editor")
        self.refresh()
        app._render()
        app.status.set(f"Retracted {mark.mark_id}")

    def choose_marks_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            title="Marks file for this bundle (existing or new)",
            initialfile="marks.jsonl", defaultextension=".jsonl",
            confirmoverwrite=False,
            filetypes=(("Feedback marks", "*.jsonl"), ("All files", "*.*")),
            parent=self.app.root)
        if path:
            self.source = MarkSource(self.source.stem or "bundle", Path(path),
                                     self.source.source_frame_offset)
            self.refresh()
            self.app._render()

    def _require_source(self) -> bool:
        if self.source.available:
            return True
        self.app.status.set("Choose a marks file first (no well folder matches this bundle)")
        return False


_frame = mark_frame
