"""Mechanical check that every extracted fact is grounded in the source text.

An LLM that invents a quote or a number must not be able to move a patient's
status. Facts that fail are dropped and reported; rules.py then raises a
MISSING_REQUIRED_DATA "needs human review" issue so nothing silently passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .facts import Extraction

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _num_str(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


@dataclass
class Verification:
    extraction: Extraction
    dropped: list[str] = field(default_factory=list)  # human-readable reasons

    @property
    def clean(self) -> bool:
        return not self.dropped


def verify(extraction: Extraction, doc_text_by_id: dict[str, str]) -> Verification:
    texts = {k: norm(v or "") for k, v in doc_text_by_id.items()}
    dropped: list[str] = []

    def grounded(kind: str, fact, extra_check=None) -> bool:
        text = texts.get(fact.doc_id)
        if text is None:
            dropped.append(f"{kind}: unknown doc_id {fact.doc_id}")
            return False
        q = norm(fact.quote)
        if not q or q not in text:
            dropped.append(f"{kind}: quote not found verbatim in {fact.doc_id}: {fact.quote[:80]!r}")
            return False
        if extra_check and not extra_check(q):
            dropped.append(f"{kind}: values not present in quote for {fact.doc_id}: {fact.quote[:80]!r}")
            return False
        return True

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

    ex = extraction
    out = Extraction(
        hp_documents=[f for f in ex.hp_documents if grounded("hp", f)],
        consents=[f for f in ex.consents if grounded("consent", f)],
        vitals=[f for f in ex.vitals if grounded("vital", f, vital_numbers_ok(f))],
        med_claims=[f for f in ex.med_claims if grounded("med_claim", f)],
        anticoag_plans=[f for f in ex.anticoag_plans if grounded("anticoag_plan", f)],
        procedure_dates=[f for f in ex.procedure_dates if grounded("procedure_date", f)],
    )
    return Verification(out, dropped)
