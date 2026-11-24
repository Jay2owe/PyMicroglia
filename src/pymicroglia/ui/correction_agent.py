"""The Review agent changes page's own assistant: Jamie's manual-correction workflow as actions.

Jamie's workflow (Motion trajectory_review README, October 2026), which this agent follows:
Jamie's join rule makes the joins, Jamie approves or denies them on the page and leaves notes, the
agent turns each note into the smallest edit that does what it says, asks about anything unclear
as a question pinned to a frame and cell (a picture, not prose), and exports only when asked.
The agent never makes approve/deny decisions, never invents numbers and never changes masks except
for a delete or piece edit a note asks for. It runs with access="app": only these actions, no shell.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import correction_review as cr

QUESTION = "Question"

INSTRUCTIONS = """You are the assistant on PyMicroglia's "Review agent changes" page. Jamie reviews the name
joins made on one tracked recording and leaves notes; you turn the notes into edits.

Jamie's rules (keep them exactly):
- Cells generally do not appear or disappear, so one cell keeps one number for the whole video.
- The join rule already ran: when a number ends, the nearest number starting up to 4 frames later
  (within 2 body radii, areas within 3x, pieces under 60 px ignored) carries its name, and each
  chain keeps the number it held longest. "Applied" joins stay unless Jamie denies them; "unsure"
  joins (up to 6 frames, 3 radii) apply only if Jamie approves them.
- Approve and deny are Jamie's alone. Never decide a join, and never treat silence as approval.
- Never create new numbers: give a cell a number that already exists in the recording. Splitting
  cells into new numbers was rejected ("most changes are horribly wrong").
- Masks stay as they are, except a delete or piece edit that a note asks for.

Workflow:
1. read_notes (unhandled first). Jamie's wording: "N here", "this is N", "should be N", "was N",
   "keep N" give the clicked cell N; "delete" or "not a cell" removes it (piece_only when it is a
   stray piece in one frame, otherwise the whole cell over frames); "swap" swaps two
   numbers; "split" or "same cell" around a merge is keep (the cell is cut back out of merges);
   "changes to N" or "swaps to N" describe the mistake, not the fix - ask.
   The one-click labels are Swap, Same cell, Split, Not a cell, Missed join, Wrong number.
2. Before each edit, check the cell with cell_at at the note's frame and position, and where the
   number goes with number_frames. Make the smallest edit_cell that does what the note says, give
   the note index, and say why in Jamie's words.
3. If a note is unclear, or two readings are possible, or a hand-drawn split is needed (Jamie draws
   those with C on the page),
   call ask_jamie at that frame and cell with lettered options instead of guessing.
4. Report what you changed as a short list (note -> edit), then what you asked.
5. export_to_motion only when Jamie asks; it writes a new Motion session and leaves the source alone.
6. Scan the timeline (Jamie's Gantt chart) when asked to look for problems, and after the notes are done:
   timeline_suspects lists, most suspicious first, numbers ending or starting mid-recording (often a
   missed join: look for another number starting or ending nearby with number_frames), jumps with no
   overlap (often a swap), a number on separate pieces, sudden size changes and gaps. Cells Jamie ticked
   as good and places already asked about are left out. Check each with cell_at / number_frames; pin a
   question with ask_jamie at that frame and cell, with lettered options, for each real-looking problem.
   Never edit from a timeline event alone: only Jamie's notes lead to edits. Then give a short list of
   what you checked, what you asked and what looked fine.
The page state tells you which recording, join, frame and cell Jamie is looking at."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _names(root, stem) -> set[int]:
    _, data, _ = cr._state(root, stem)
    return {int(n) for n, _, _ in data["tracks"]}


def install(app):
    """Mount the page agent on /correction-review only (the workbench assistant stays elsewhere)."""
    from aiify import Agent
    from aiify.actions import from_functions

    focus = app.state.correction_focus = {}

    def where(root: str = "", stem: str = "") -> tuple[str, str]:
        root, stem = root or focus.get("root", ""), stem or focus.get("stem", "")
        if not root or not stem:
            raise ValueError("Open a recording on the page first, or give root and stem")
        return root, stem

    def list_review_sets(root: str = "") -> dict:
        """List the review sets (one per recording) under a Motion editing folder."""
        root = root or focus.get("root", "")
        return {"root": root, "reviews": cr.list_reviews(root)}

    def open_review(root: str = "", stem: str = "") -> dict:
        """Summary of one recording's review: join counts, decisions, notes, edits and exports."""
        root, stem = where(root, stem)
        state = cr.load(root, stem)
        undecided = [it["key"] for it in state["items"] if it["key"] not in state["decisions"]]
        return {**cr.summary(root, stem), "rules": state["rules"], "undecided": len(undecided),
                "conflicts": state["conflicts"], "edits": state["edits"]}

    def list_joins(root: str = "", stem: str = "", kind: str = "", undecided_only: bool = False) -> dict:
        """The joins with Jamie's decisions. kind is 'applied', 'unsure' or '' for both."""
        root, stem = where(root, stem)
        state = cr.load(root, stem)
        rows = [{**it, "decision": state["decisions"].get(it["key"])} for it in state["items"]
                if (not kind or it["kind"] == kind) and not (undecided_only and it["key"] in state["decisions"])]
        return {"joins": rows}

    def read_notes(root: str = "", stem: str = "", unhandled_only: bool = True) -> dict:
        """Jamie's notes with their index; an edit or question naming a note's index handles it."""
        root, stem = where(root, stem)
        state = cr.load(root, stem)
        handled = ({e.get("note") for e in state["edits"]}
                   | {n.get("about") for n in state["notes"] if n.get("tag") == QUESTION})
        rows = [{"index": i, **n, "handled": i in handled} for i, n in enumerate(state["notes"])
                if n.get("tag") != QUESTION]
        return {"notes": [r for r in rows if not (unhandled_only and r["handled"])]}

    def cell_at(frame: int, y: int, x: int, root: str = "", stem: str = "") -> dict:
        """The number under a full-frame pixel in the source (before) and now (decisions and edits
        applied), and the last frame of its run (to)."""
        root, stem = where(root, stem)
        return cr.cell_now(root, stem, int(frame), int(y), int(x))

    def number_frames(number: int, root: str = "", stem: str = "") -> dict:
        """Frames (as runs) where a number appears now, with its centre in each run's first frame."""
        root, stem = where(root, stem)
        _, after, _ = cr.composed(root, stem)
        present = [(after[f] == number) for f in range(after.shape[0])]
        runs, start = [], None
        for f, mask in enumerate(present + [None]):
            on = mask is not None and mask.any()
            if on and start is None:
                start = f
            if not on and start is not None:
                ys, xs = present[start].nonzero()
                runs.append({"first": start, "last": f - 1,
                             "centre": [round(float(ys.mean())), round(float(xs.mean()))]})
                start = None
        return {"number": number, "runs": runs}

    def _picked(root, stem, frame, y, x):
        cell = cr.cell_now(root, stem, int(frame), int(y), int(x))
        if not cell["now"]:
            raise ValueError(f"No cell at y={y}, x={x} in frame {frame}")
        return {"frame": int(frame), "y": int(y), "x": int(x), "now": cell["now"], "cell": cell["now"]}, cell["to"]

    def _check(result):
        if not result.get("ok"):
            raise ValueError(result.get("message") or "Nothing changed")
        return result

    def edit_cell(action: str, frame: int, y: int, x: int, last: int = -1, number: int = 0, other_y: int = -1,
                  other_x: int = -1, piece_only: bool = False, note: int = -1, why: str = "",
                  root: str = "", stem: str = "") -> dict:
        """Jamie's edit buttons on the cell at (frame, y, x), from that frame to `last` (-1: to the end
        of the cell's run). action: 'give' (number, an existing number), 'delete', 'swap' (with the cell
        at other_y, other_x in the same frame), 'keep' (cut the cell back out of merges around the join
        on screen) or 'keep_whole' (over the whole recording). piece_only changes only the clicked
        connected part, in that frame. note is the note index it answers."""
        root, stem = where(root, stem)
        picked, to = _picked(root, stem, frame, y, x)
        if action == "give" and int(number) not in _names(root, stem):
            raise ValueError(f"{number} is not a number in this recording; Jamie's rule allows no new numbers")
        other = _picked(root, stem, frame, other_y, other_x)[0]["now"] if action == "swap" else None
        key = focus.get("key") if focus.get("stem") == stem else None
        result = cr.edit(root, stem, action="keep" if action == "keep_whole" else action, picked=picked,
                         frame=int(frame), to_frame=to if int(last) < 0 else int(last), key=key, other=other,
                         number=int(number) or None, piece_only=piece_only, whole=action == "keep_whole",
                         by="agent", why=why, note=note if note >= 0 else None)
        return _check(result)

    def remove_agent_edit(root: str = "", stem: str = "") -> dict:
        """Undo the last edit if the agent made it (never Jamie's)."""
        root, stem = where(root, stem)
        edits = cr.load(root, stem)["edits"]
        if not edits or edits[-1].get("by") != "agent":
            raise ValueError("The last edit is not the agent's; Jamie undoes edits on the page")
        return _check(cr.undo_edit(root, stem, len(edits)))

    def ask_jamie(frame: int, question: str, options: str = "", y: int = -1, x: int = -1, about: int = -1,
                  join: str = "", root: str = "", stem: str = "") -> dict:
        """Pin a question to a frame (and cell) so Jamie sees it on the picture. options: lettered
        choices such as 'A) keep 52  B) give it 67'. about: the note index it asks about."""
        root, stem = where(root, stem)
        picked = None
        if y >= 0 and x >= 0:
            cell = cr.cell_now(root, stem, int(frame), int(y), int(x))
            picked = {"frame": int(frame), "y": int(y), "x": int(x), "cell": cell["now"] or None, "panel": "After"}
        result = cr.add_note(root, stem, key=join or focus.get("key"), frame=int(frame), picked=picked,
                             text=question + (f"  {options}" if options else ""), tag=QUESTION, by="agent",
                             about=about if about >= 0 else None)
        notes = cr.load(root, stem)["notes"]
        return {**result, "questions": sum(n.get("tag") == QUESTION for n in notes)}

    def timeline_suspects(kind: str = "", limit: int = 30, include_ticked: bool = False,
                          root: str = "", stem: str = "") -> dict:
        """The timeline (Gantt chart) events worth a look, most suspicious first: kind 'end', 'start', 'jump',
        'pieces', 'size', 'gap' or '' for all. Leaves out cells Jamie ticked as good and places with an open
        question; says where a Jamie edit already covers the cell. Also returns Jamie's own flags and notes."""
        from .correction_truth import suspects
        root, stem = where(root, stem)
        return suspects(root, stem, kind=kind, limit=limit, include_ticked=include_ticked)

    def export_to_motion(root: str = "", stem: str = "") -> dict:
        """Write the reviewed labels as a new Motion editor session (only when Jamie asks)."""
        root, stem = where(root, stem)
        return cr.export(root, stem, app.state.motion_url, user="agent for jamie")

    def create_review_set(root: str, source: str, agent_session: str = "") -> dict:
        """Run Jamie's join rule on an untouched Motion session or bundle and make its review set."""
        return cr.create_review(root, source, agent_session=agent_session or None)

    functions = {f.__name__: f for f in (list_review_sets, open_review, list_joins, read_notes, cell_at,
                                         number_frames, timeline_suspects, edit_cell, remove_agent_edit, ask_jamie,
                                         export_to_motion, create_review_set)}
    actions = from_functions(functions, mutating={"edit_cell", "remove_agent_edit", "ask_jamie",
                                                  "create_review_set"},
                             destructive={"export_to_motion"})

    def state():
        out = dict(focus)
        if focus.get("root") and focus.get("stem"):
            try:
                s = cr.summary(focus["root"], focus["stem"])
                out["review"] = {k: s[k] for k in ("frames", "applied", "unsure", "decided", "notes", "edits")}
            except (ValueError, OSError):
                pass
        return out

    agent = Agent("pymicroglia-corrections", actions=actions, guide=INSTRUCTIONS, state=state,
                  instructions=INSTRUCTIONS, prewarm=False, limit_check_every=None, access="app",
                  suggestions=("Turn my notes into edits", "Which joins are still undecided?",
                               "Ask me about anything unclear in my notes",
                               "Scan the timeline for suspicious cells"))
    agent.mount(app, prefix="/aiify-corrections", inject=lambda path: path.startswith("/correction-review"),
                panel={"title": "Correction assistant", "layout": "float"})
    app.state.correction_agent = agent
    return agent
