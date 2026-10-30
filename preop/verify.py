"""Mechanical check that every extracted fact is grounded in the source text.

An LLM that invents a quote or a number must not be able to move a patient's
status. Facts that fail are dropped and reported; rules.py then raises a
MISSING_REQUIRED_DATA "needs human review" issue so nothing silently passes.

Quotes that are *almost* right (the model garbles non-ASCII characters such as the
degree sign, or trims/pads a line) are "snapped" to the closest real line of the
source, so the quote a reviewer sees is always verbatim. Numbers are still checked
against the snapped line, so snapping can never invent a value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .facts import Extraction

_WS = re.compile(r"\s+")
SNAP_THRESHOLD = 0.85  # share of the quote's characters found, in order, inside one source line


def norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _num_str(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def snap_quote(quote: str, raw_text: str) -> str | None:
    """Return a verbatim span of raw_text matching `quote`, or None if nothing close enough."""
    nq, nt = norm(quote), norm(raw_text)
    if not nq:
        return None
    if nq in nt:
        return quote
    first = next((ln for ln in quote.split("\n") if ln.strip()), "")
    if first.strip() and norm(first) in nt:  # long multi-line quote: keep its verbatim first line
        return first.strip()
    best, best_ratio = None, 0.0
    for ln in raw_text.split("\n"):
        if not ln.strip():
            continue
        blocks = SequenceMatcher(None, nq, norm(ln), autojunk=False).get_matching_blocks()
        r = sum(b.size for b in blocks) / len(nq)
        if r > best_ratio:
            best, best_ratio = ln.strip(), r
    return best if best_ratio >= SNAP_THRESHOLD else None


@dataclass
class Verification:
    extraction: Extraction
    dropped: list[str] = field(default_factory=list)  # human-readable reasons

    @property
    def clean(self) -> bool:
        return not self.dropped


def verify(extraction: Extraction, doc_text_by_id: dict[str, str]) -> Verification:
    dropped: list[str] = []

    def grounded(kind: str, fact, extra_check=None):
        """Return the fact with a verbatim quote, or None (and record why)."""
        raw = doc_text_by_id.get(fact.doc_id)
        if raw is None:
            dropped.append(f"{kind}: unknown doc_id {fact.doc_id}")
            return None
        q = snap_quote(fact.quote, raw)
        if q is None:
            dropped.append(f"{kind}: quote not found in {fact.doc_id}: {fact.quote[:80]!r}")
            return None
        fact = fact.model_copy(update={"quote": q})
        if extra_check and not extra_check(norm(q)):
            dropped.append(f"{kind}: values not present in quote for {fact.doc_id}: {q[:80]!r}")
            return None
        return fact

    def vital_numbers_ok(v):
        def check(q: str) -> bool:
            if v.kind == "blood_pressure":
                return (
                    v.systolic is not None
                    and v.diastolic is not None
                    and re.search(rf"\b{_num_str(v.systolic)}\s*/\s*{_num_str(v.diastolic)}\b", q) is not None
                )
            return v.temp_value is not None and _num_str(v.temp_value) in q

        return check

    def keep(kind, facts, check_for=None):
        out = []
        for f in facts:
            g = grounded(kind, f, check_for(f) if check_for else None)
            if g is not None:
                out.append(g)
        return out

    ex = extraction
    out = Extraction(
        hp_documents=keep("hp", ex.hp_documents),
        consents=keep("consent", ex.consents),
        vitals=keep("vital", ex.vitals, vital_numbers_ok),
        med_claims=keep("med_claim", ex.med_claims),
        anticoag_plans=keep("anticoag_plan", ex.anticoag_plans),
        procedure_dates=keep("procedure_date", ex.procedure_dates),
    )
    return Verification(out, dropped)