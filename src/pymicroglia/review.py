"""What the pipeline decided on its own, and what it wants a person to check.

Some of this analysis cannot be automated. Which channel is the
bioluminescence, where the usable window starts, where the tissue outline goes,
whether a faint object is a cell — each is a judgement call, and a pipeline that
stopped to ask would never finish a batch. ``dluc_pipeline.py`` solved this by
deciding anyway and writing down what it was unsure about; nothing else in the
lab does. This module is that mechanism, generalised, plus the part that engine
never had: an answer, once given, is stored and the question is not asked again.

**A review is not a log.** ``open_questions()`` returning forty items is the
same as returning none, so the bar is deliberately high: a judgement call opens
a question only when it was made with less than full confidence, or when
choosing differently would change a number somebody reports. Everything else is
a ``note`` — said out loud once and then left alone.

The three severities are the engine's, unchanged, because agents and people
already read them:

``blocker``
    Do not report a result from this run until it is resolved.
``check``
    A judgement call the defaults may have got wrong. Show the evidence, ask
    the question, re-run with the answer.
``note``
    Worth saying out loud. No action, no question.

``WORKFLOW.md`` in ``dLuc_single_cell_analysis`` is the specification for how an
agent drives the loop this feeds: run once with defaults, read the review, bring
only the open questions back to the user, re-run with their answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SEVERITIES",
    "SEVERITY_ORDER",
    "CONFIDENCE_ORDER",
    "UnknownQuestion",
    "Note",
    "Review",
]

#: Ordered worst first, which is the order a person should read them in.
SEVERITIES: tuple[str, ...] = ("blocker", "check", "note")
SEVERITY_ORDER = {name: index for index, name in enumerate(SEVERITIES)}

#: How sure the pipeline was. Anything below ``high`` opens a question.
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


class UnknownQuestion(KeyError):
    """An answer was offered to a question this run never asked.

    Refused rather than stored. A decision keyed on a question the pipeline
    does not raise is a silent override: it sits in the store looking
    authoritative and nothing ever reads it, or worse, something does and
    nobody can see why the run behaved differently.
    """


@dataclass(frozen=True)
class Note:
    """One judgement call: what was chosen, how sure, and why.

    ``gate`` is the stable name — ``"channels"``, ``"window"``, ``"roi"``,
    ``"admissibility"`` — and is what an answer is keyed on. The headline and
    the detail are prose that may be reworded between versions; the gate is
    not, because a stored answer has to keep matching it.
    """

    gate: str
    severity: str = "check"
    headline: str = ""
    detail: str = ""
    evidence: tuple[str, ...] = ()
    question: str = ""
    remedy: str = ""
    chosen: Any = None
    options: tuple[Any, ...] = ()
    confidence: str = "high"
    changes_result: bool = False
    answer: Any = None
    answered: bool = False

    @property
    def open(self) -> bool:
        """Whether a person should actually look at this one.

        The bar, stated once: a blocker always; a plain note never; a check
        only while nobody has answered it. Confidence and ``changes_result``
        are what turned a judgement call into a check in the first place — see
        :meth:`Review.note`.
        """
        if self.answered:
            return False
        if self.severity == "blocker":
            return True
        return self.severity == "check" and bool(self.question)

    def as_dict(self) -> dict[str, Any]:
        """The engine's ``summary.json["review"]`` shape, with the new fields.

        The first six keys are ``dluc_pipeline.py``'s exactly, so anything that
        reads that file's review keeps working against this one.
        """
        return {
            "severity": self.severity,
            "gate": self.gate,
            "headline": self.headline,
            "detail": self.detail,
            "evidence": list(self.evidence),
            "question": self.question,
            "remedy": self.remedy,
            "chosen": _plain(self.chosen),
            "options": [_plain(option) for option in self.options],
            "confidence": self.confidence,
            "changes_result": self.changes_result,
            "answered": self.answered,
            "answer": _plain(self.answer),
        }


def _plain(value: Any) -> Any:
    """Something ``json.dump`` will take, without importing numpy to find out."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


class Review:
    """The judgement calls one run made, and the answers it already had.

    Constructed with the source it is about, so an answer can be stored against
    that source and found again by the next run. Constructed without one it
    still works and simply cannot remember anything, which is what the tests
    and the synthetic pipelines want.
    """

    def __init__(self, source: Any = None, *, decisions: Mapping[str, Any] | None = None):
        self.source = source
        self._notes: list[Note] = []
        self._decisions: dict[str, Any] = dict(decisions or {})
        if decisions is None and source is not None:
            self._decisions = dict(self._stored_decisions())

    # ------------------------------------------------------------ recording
    def note(self, gate: str, *, chosen: Any = None, confidence: str = "high",
             why: str = "", options: Sequence[Any] = (), question: str = "",
             evidence: Sequence[str] = (), remedy: str = "",
             changes_result: bool = False, headline: str = "") -> Note:
        """Record a judgement call the pipeline made rather than stopping to ask.

        The severity is *derived*, not passed, and that is the whole point of
        this door: a call made with full confidence that changes nothing
        downstream is a note, and anything else is a check. Deciding severity by
        hand at each call site is how a review turns into a log.
        """
        confidence = str(confidence).lower()
        if confidence not in CONFIDENCE_ORDER:
            raise ValueError(
                f"confidence must be one of {sorted(CONFIDENCE_ORDER)}; "
                f"got {confidence!r}")
        uncertain = CONFIDENCE_ORDER[confidence] > 0
        severity = "check" if (uncertain or changes_result) else "note"
        if severity == "check" and not question:
            question = _default_question(gate, chosen)
        return self._add(Note(
            gate=str(gate), severity=severity,
            headline=headline or _headline(gate, chosen),
            detail=why, evidence=tuple(evidence), question=question,
            remedy=remedy, chosen=chosen, options=tuple(options),
            confidence=confidence, changes_result=bool(changes_result)))

    def flag(self, severity: str, gate: str, headline: str, detail: str = "",
             evidence: Sequence[str] = (), question: str = "",
             remedy: str = "") -> Note:
        """``dluc_pipeline.py``'s ``flag``, argument for argument.

        Kept because every call site ported from that engine reads the same on
        both sides, which is what makes a port checkable by eye. New code
        should prefer :meth:`note`, where the severity is derived rather than
        asserted.
        """
        if severity not in SEVERITY_ORDER:
            raise ValueError(f"severity must be one of {list(SEVERITIES)}; "
                             f"got {severity!r}")
        return self._add(Note(
            gate=str(gate), severity=str(severity), headline=headline,
            detail=detail, evidence=tuple(evidence), question=question,
            remedy=remedy,
            confidence="high" if severity == "note" else "medium",
            changes_result=severity != "note"))

    def _add(self, item: Note) -> Note:
        if item.gate in self._decisions:
            item = replace(item, answered=True,
                           answer=self._decisions[item.gate])
        self._notes.append(item)
        return item

    # -------------------------------------------------------------- reading
    def __len__(self) -> int:
        return len(self._notes)

    def __iter__(self):
        return iter(self.sorted())

    def sorted(self) -> list[Note]:
        """Worst first, and stable within a severity so the order is readable."""
        return sorted(self._notes,
                      key=lambda item: SEVERITY_ORDER.get(item.severity, 9))

    def open_questions(self) -> list[Note]:
        """Only the ones a person should actually look at.

        Answered questions are gone, notes were never here, and what is left is
        exactly what an agent should batch into one message. See the module
        docstring for why the bar is where it is.
        """
        return [item for item in self.sorted() if item.open]

    def counts(self) -> dict[str, int]:
        counted = {name: 0 for name in SEVERITIES}
        for item in self._notes:
            counted[item.severity] = counted.get(item.severity, 0) + 1
        counted["open"] = len(self.open_questions())
        return counted

    def gates(self) -> list[str]:
        seen: list[str] = []
        for item in self._notes:
            if item.gate not in seen:
                seen.append(item.gate)
        return seen

    def as_records(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self.sorted()]

    def blocked(self) -> bool:
        """Whether any result from this run should be reported at all."""
        return any(item.severity == "blocker" and not item.answered
                   for item in self._notes)

    # ------------------------------------------------------------- answering
    def decided(self, gate: str, default: Any = None) -> Any:
        """A stored answer to this question, or ``default``.

        This is the read side of the loop: a pipeline calls it *before*
        deciding, so a question a person has already settled is never re-opened
        — not by a parameter change and not by a ``METHOD_VERSION`` bump.
        """
        return self._decisions.get(str(gate), default)

    def answer(self, gate: str, value: Any, *, note: str = "") -> Any:
        """A person's answer becomes a decision artefact keyed on the source.

        From here on the pipeline reads it from the store instead of deciding,
        and the question drops out of the review. That is why the same question
        is never asked twice, and why re-running after a parameter change does
        not re-open settled ground.

        An answer to a question this run never asked is refused. See
        :class:`UnknownQuestion` for why a silent override is worse than an
        error.
        """
        gate = str(gate)
        known = {item.gate for item in self._notes}
        if gate not in known:
            raise UnknownQuestion(
                f"this run never asked about {gate!r}, so an answer to it "
                f"cannot be stored. Questions raised: "
                f"{sorted(known) or 'none'}. Storing an answer to a question "
                f"nobody asked is a silent override: nothing would show why a "
                f"later run behaved differently.")
        self._decisions[gate] = value
        self._notes = [replace(item, answered=True, answer=value)
                       if item.gate == gate else item
                       for item in self._notes]
        if self.source is not None:
            from . import store

            store.decision(gate, self.source, value=value,
                           note=note or f"answered {gate}")
        return value

    def _stored_decisions(self) -> dict[str, Any]:
        try:
            from . import store

            return dict(store.decisions_for(self.source))
        except Exception:
            # A store that cannot be read costs this run its memory of past
            # answers, and nothing else. It must not stop the analysis.
            return {}

    # -------------------------------------------------------------- output
    def render(self, path=None, *, title: str = "", source_name: str = "",
               extra: Mapping[str, Any] | None = None) -> str:
        """The review as Markdown; written to ``path`` when one is given.

        The section is called ``## What needs your eye`` because that is the
        heading ``WORKFLOW.md`` tells an agent to open first, and an agent
        looking for it in a renamed section finds nothing.
        """
        counted = self.counts()
        lines = [f"# {title or 'Review'}", ""]
        if source_name:
            lines += [f"Source: `{source_name}`", ""]
        for key, value in (extra or {}).items():
            lines.append(f"- **{key}**: {value}")
        if extra:
            lines.append("")
        lines += [
            "## What needs your eye",
            "",
            f"{counted['blocker']} blocker(s), {counted['check']} check(s), "
            f"{counted['note']} note(s); {counted['open']} still open. An "
            "agent driving this pipeline should bring the open ones back to "
            "the user, and nothing else.",
            "",
        ]
        for index, item in enumerate(self.sorted(), 1):
            mark = " *(answered)*" if item.answered else ""
            lines.append(f"### {index}. [{item.severity.upper()}] "
                         f"{item.headline or item.gate}{mark}")
            lines.append("")
            if item.detail:
                lines += [item.detail, ""]
            if item.chosen is not None:
                lines.append(f"- Chosen: `{_plain(item.chosen)}` "
                             f"(confidence: {item.confidence})")
            if item.options:
                lines.append("- Options: "
                             + ", ".join(f"`{_plain(o)}`" for o in item.options))
            if item.evidence:
                lines.append("- Look at: " + ", ".join(item.evidence))
            if item.question and not item.answered:
                lines.append(f"- **Ask:** {item.question}")
            if item.answered:
                lines.append(f"- Answered: `{_plain(item.answer)}`")
            if item.remedy:
                lines.append(f"- Then re-run with: {item.remedy}")
            lines.append("")
        if counted["blocker"]:
            lines += [f"**Do not report a result from this run until the "
                      f"{counted['blocker']} blocker(s) above are resolved.**",
                      ""]
        elif not counted["open"]:
            lines += ["Nothing needs a decision. The result stands as it is.",
                      ""]
        text = "\n".join(lines)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def to_terminal(self) -> str:
        """The plain-text form the engine prints at the end of a run."""
        counted = self.counts()
        lines = [
            "REVIEW - what needs your eye",
            f"  {counted['blocker']} blocker(s), {counted['check']} check(s), "
            f"{counted['note']} note(s), {counted['open']} open.",
        ]
        for index, item in enumerate(self.sorted(), 1):
            lines.append(f"\n  [{item.severity.upper()}] {index}. "
                         f"{item.headline or item.gate}"
                         + ("  (answered)" if item.answered else ""))
            for line in _wrapped(item.detail, 68):
                lines.append("      " + line)
            if item.evidence:
                lines.append("      look at: " + ", ".join(item.evidence))
            if item.question and not item.answered:
                lines.append("      ASK: " + item.question)
            if item.remedy:
                lines.append("      then re-run with: " + item.remedy)
        return "\n".join(lines)

    def write_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_records(), indent=1),
                        encoding="utf-8")
        return path


# ------------------------------------------------------------------ helpers
_GATE_WORDS = {
    "channels": "channel assignment",
    "window": "analysis window",
    "roi": "region outline",
    "admissibility": "object admissibility",
    "tissue_mask": "tissue mask",
    "registration": "registration",
    "cosmic_rays": "cosmic-ray removal",
    "movement": "object movement",
    "control": "instrumental control",
    "learned_mask": "learned single-frame mask",
}


def _headline(gate: str, chosen: Any) -> str:
    words = _GATE_WORDS.get(gate, gate.replace("_", " "))
    return (f"The {words} was chosen automatically" if chosen is None
            else f"The {words} was chosen automatically: {_plain(chosen)}")


def _default_question(gate: str, chosen: Any) -> str:
    words = _GATE_WORDS.get(gate, gate.replace("_", " "))
    return f"Is the {words} right?" if chosen is None else \
        f"Is the {words} right ({_plain(chosen)})?"


def _wrapped(text: str, width: int) -> Iterable[str]:
    if not text:
        return []
    out: list[str] = []
    line = ""
    for word in str(text).split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out
