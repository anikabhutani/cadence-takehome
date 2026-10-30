"""triage_submission pipeline: extract -> verify -> rules -> explain."""

from __future__ import annotations

import os

from .rules import ASSUMPTIONS, evaluate
from .verify import verify


def get_extractor(name: str | None = None):
    name = name or os.environ.get("PREOP_EXTRACTOR", "llm")
    if name == "heuristic":
        from .heuristic_extract import extract
        return lambda sub, model: extract(sub)
    from .llm_extract import extract
    return extract


def run(sub, model: str) -> dict:
    extractor = get_extractor()
    ex = extractor(sub, model)
    ver = verify(ex, {d.doc_id: d.text or "" for d in sub.documents})
    res = evaluate(sub, ver.extraction, ver.dropped)
    n = len(res.issues)
    if res.decision == "READY":
        explanation = "All required documentation, testing, anticoagulation and safety criteria are satisfied."
    else:
        lead = {"NOT_CLEARED": "Not cleared: acute safety exclusion present.",
                "NEEDS_FOLLOW_UP": "Needs follow-up before scheduling."}[res.decision]
        explanation = lead + " " + " ".join(f"[{i['category']}] {i['finding']}" for i in res.issues)
    return {
        "decision": res.decision,
        "issues": res.issues,
        "explanation": explanation,
        "rules_passed": res.rules_passed,
        "assumptions": ASSUMPTIONS + res.notes,
        "procedure_date_used": res.procedure_date,
        "needs_human_review": bool(ver.dropped) or any(i["confidence"] == "low" for i in res.issues),
    }
