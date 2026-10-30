"""Deterministic implementation of the Cadence pre-op policy (Rules 1-4 + precedence).

Pure functions over (structured submission, verified Extraction). No LLM, no I/O.
Every issue carries: policy rule, the finding, the values compared, the source
evidence (ids + verbatim quotes), and what the reviewer should do next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any

from .drugs import canonical_anticoagulant
from .facts import Extraction

# Assumptions are surfaced in every output so reviewers know how edge cases were read.
ASSUMPTIONS = [
    "Window checks use calendar dates; day N inclusive (30 days before = still valid). Results dated after the procedure do not count.",
    "Procedure date: structured field first; else a single definite date written in a document; else MISSING_REQUIRED_DATA (window checks then skipped).",
    "A required H&P is any completed History & Physical note; notes that only reference an H&P do not count.",
    "Consent counts only with the PATIENT's signature evidenced and for the planned procedure; a physician attestation alone does not.",
    "Labs with status other than final/amended/corrected (e.g. entered-in-error) are ignored; BMP is not CMP.",
    "Vitals: latest reading per type across structured data and documents; times default to 00:00 when unknown and ties go to the worse value. A within-range recheck supersedes an earlier high reading.",
    "A required vital (BP or temperature) never recorded anywhere is MISSING_REQUIRED_DATA, not assumed normal.",
    "Anticoagulants only (antiplatelets such as aspirin/clopidogrel need no plan). Current use is the most recent statement; a plan must name that drug and state both pre-op and post-op handling explicitly.",
    "Unknown procedure risk is MISSING_REQUIRED_DATA; a CBC within 30 days is still demanded as the minimum.",
]


def _d(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _dt(date_s: str | None, time_s: str | None = None) -> datetime | None:
    d = _d(date_s)
    if d is None:
        return None
    t = time(0, 0)
    if time_s is None and date_s and len(date_s) >= 16 and "T" in date_s:
        time_s = date_s[11:16]
    if time_s:
        try:
            hh, mm = time_s.split(":")[:2]
            t = time(int(hh), int(mm))
        except ValueError:
            pass
    return datetime.combine(d, t)


def _issue(category, rule, finding, *, sources=None, values=None, action, confidence="high"):
    return {
        "category": category,
        "rule": rule,
        "finding": finding,
        "evidence": finding,
        "sources": sources or [],
        "values_checked": values or {},
        "reviewer_action": action,
        "confidence": confidence,
    }


@dataclass
class Result:
    decision: str
    issues: list[dict[str, Any]]
    rules_passed: list[str]
    procedure_date: str | None
    notes: list[str] = field(default_factory=list)


def evaluate(sub, ex: Extraction, dropped: list[str]) -> Result:
    docs = {d.doc_id: d for d in sub.documents}
    issues: list[dict] = []
    passed: list[str] = []
    notes: list[str] = []

    def doc_src(doc_id, quote=None, date_=None):
        d = docs.get(doc_id)
        return {"source": "document", "id": doc_id, "type": d.type if d else None,
                "date": date_ or (d.date if d else None), "quote": quote}

    # ---------------- procedure date / risk ----------------
    proc = sub.procedure
    pdate = _d(proc.procedure_date) if proc else None
    if pdate is None:
        defs = {f.date for f in ex.procedure_dates if f.date}
        if len(defs) == 1:
            pdate = _d(next(iter(defs)))
            notes.append(f"Procedure date taken from document text: {pdate}.")
        else:
            tbd = [f for f in ex.procedure_dates if not f.date]
            issues.append(_issue(
                "MISSING_REQUIRED_DATA", "Rules 1-2 (window checks)",
                "Procedure date is not set, so H&P and lab recency windows cannot be evaluated."
                + (" Documents say the date is still pending." if tbd else ""),
                sources=[doc_src(f.doc_id, f.quote) for f in ex.procedure_dates],
                values={"procedure.procedure_date": None},
                action="Obtain a scheduled procedure date, then re-run triage."))
    risk = proc.procedure_risk if proc else None
    if risk is None:
        issues.append(_issue(
            "MISSING_REQUIRED_DATA", "Rule 2 (risk level)",
            "Procedure risk level (LOW/MODERATE/HIGH) is blank, so the required lab set cannot be determined.",
            values={"procedure.procedure_risk": None},
            action="Have the surgical team assign a risk category."))

    # ---------------- Rule 1: H&P ----------------
    hps = []
    for f in ex.hp_documents:
        if f.is_history_and_physical:
            hd = _d(f.date_of_service) or _d(docs[f.doc_id].date if f.doc_id in docs else None)
            if hd:
                hps.append((hd, f))
    hps.sort(key=lambda x: x[0])
    if pdate:
        ok = [(hd, f) for hd, f in hps if 0 <= (pdate - hd).days <= 30]
    else:
        ok = hps
    if ok:
        hd, f = ok[-1]
        passed.append(f"Rule 1 H&P: {hd} ({(pdate - hd).days if pdate else '?'} days before procedure, limit 30) [{f.doc_id}]")
    elif pdate is None and not hps:
        issues.append(_issue("REQUIRED_DOCUMENTATION", "Rule 1", "No History & Physical found in the package.",
                             action="Obtain H&P within 30 days of the procedure."))
    elif pdate is not None:
        if hps:
            hd, f = hps[-1]
            age = (pdate - hd).days
            why = f"most recent H&P is {age} days before the procedure (limit 30)" if age >= 0 else "most recent H&P is dated after the procedure"
            issues.append(_issue(
                "REQUIRED_DOCUMENTATION", "Rule 1", f"H&P outside required window: {why}.",
                sources=[doc_src(h.doc_id, h.quote, str(d_)) for d_, h in hps],
                values={"procedure_date": str(pdate), "latest_hp": str(hd), "days_before": age, "limit_days": 30},
                action="Obtain a new H&P completed within 30 days of the procedure."))
        else:
            hp_refs = [doc_src(f.doc_id, f.quote) for f in ex.hp_documents]
            issues.append(_issue(
                "REQUIRED_DOCUMENTATION", "Rule 1", "No History & Physical found in the package.",
                sources=hp_refs, action="Obtain H&P within 30 days of the procedure."))

    # ---------------- Rule 1: consent ----------------
    signed = [c for c in ex.consents if c.patient_signature == "signed" and c.matches_planned_procedure is not False]
    if signed:
        c = sorted(signed, key=lambda c: c.signed_date or docs[c.doc_id].date or "")[-1]
        passed.append(f"Rule 1 consent: patient-signed for planned procedure [{c.doc_id}]")
    else:
        if not ex.consents:
            msg = "No surgical consent document found in the package."
        elif all(c.matches_planned_procedure is False for c in ex.consents):
            msg = "Only consent(s) on file are for a different procedure."
        else:
            msg = "Consent on file is not patient-signed (blank/pending/not completed)."
        issues.append(_issue(
            "REQUIRED_DOCUMENTATION", "Rule 1", msg,
            sources=[doc_src(c.doc_id, c.quote) for c in ex.consents],
            values={"consents": [{"doc_id": c.doc_id, "signature": c.patient_signature,
                                  "matches_procedure": c.matches_planned_procedure} for c in ex.consents]},
            action="Obtain a patient-signed consent for the planned procedure."))

    # ---------------- Rule 2: testing ----------------
    def latest_lab(kind):
        out = []
        for lab in sub.labs:
            name = f"{lab.code or ''} {lab.display or ''}".lower()
            is_cbc = "cbc" in name or "complete blood count" in name or "hemogram" in name
            is_cmp = ("cmp" in name or "comprehensive metabolic" in name)
            if (kind == "CBC" and is_cbc) or (kind == "CMP" and is_cmp):
                if (lab.status or "").lower() not in ("final", "amended", "corrected"):
                    notes.append(f"Ignored {kind} {lab.id} ({lab.effective_at}) with status '{lab.status}'.")
                    continue
                ld = _d(lab.effective_at)
                if ld is None or (pdate and ld > pdate):
                    continue
                out.append((ld, lab))
        return max(out, key=lambda x: x[0]) if out else None

    need = [("CBC", 30)]
    if risk == "HIGH":
        need = [("CBC", 14), ("CMP", 14)]
    elif risk is None:
        need = [("CBC", 30)]
    for kind, limit in need:
        res = latest_lab(kind)
        if res is None:
            issues.append(_issue("REQUIRED_TESTING", "Rule 2", f"No valid {kind} on file.",
                                 values={"required": kind, "limit_days": limit, "found": None},
                                 action=f"Order {kind} within {limit} days of the procedure."))
        elif pdate is None:
            continue
        else:
            ld, lab = res
            age = (pdate - ld).days
            src = [{"source": "lab", "id": lab.id, "type": lab.display, "date": lab.effective_at, "quote": None}]
            if age > limit:
                issues.append(_issue("REQUIRED_TESTING", "Rule 2",
                                     f"Most recent {kind} is {age} days before the procedure (limit {limit}).",
                                     sources=src, values={"required": kind, "limit_days": limit, "age_days": age},
                                     action=f"Repeat {kind} within {limit} days of the procedure."))
            else:
                passed.append(f"Rule 2 {kind}: {ld} ({age} days, limit {limit}) [{lab.id}]")

    # ---------------- Rule 3: anticoagulation ----------------
    status_by_drug: dict[str, dict[str, Any]] = {}

    def rec(drug):
        return status_by_drug.setdefault(drug, {"struct": None, "struct_src": None, "claims": []})

    for m in sub.medications:
        drug = canonical_anticoagulant(m.name)
        if drug:
            r = rec(drug)
            r["struct"] = "unclear" if m.active is None else ("taking" if m.active else "not_taking")
            r["struct_src"] = {"source": "medication_list", "id": None, "type": "medications", "date": None, "quote": m.name}
    for c in ex.med_claims:
        drug = canonical_anticoagulant(c.drug_as_written)
        if drug:
            st = {"taking": "taking", "not_taking": "not_taking", "discontinued_or_historical": "not_taking"}.get(c.status, "unclear")
            rec(drug)["claims"].append((docs[c.doc_id].date if c.doc_id in docs else "", st, c))
    for p in ex.anticoag_plans:  # a plan implies the clinician considers the drug in use
        drug = canonical_anticoagulant(p.drug_as_written)
        if drug:
            rec(drug)

    taking: list[str] = []
    for drug, r in sorted(status_by_drug.items()):
        claims = sorted(r["claims"], key=lambda x: x[0] or "")
        doc_status = claims[-1][1] if claims else None
        struct = r["struct"]
        if doc_status in (None, "unclear") and struct in (None, "unclear"):
            final = "unclear" if (struct == "unclear" or doc_status == "unclear") else None
        elif doc_status in (None, "unclear"):
            final = struct
        elif struct in (None, "unclear"):
            final = doc_status
        else:
            final = doc_status if doc_status == struct else "conflict"
        srcs = ([r["struct_src"]] if r["struct_src"] else []) + [doc_src(c.doc_id, c.quote) for _, _, c in claims]
        if final == "taking":
            taking.append(drug)
        elif final in ("unclear", "conflict") or (final is None and (r["struct_src"] or claims)):
            if final is not None:
                issues.append(_issue(
                    "MISSING_REQUIRED_DATA", "Rule 3",
                    f"Cannot determine whether the patient currently takes {drug} "
                    f"({'sources disagree' if final == 'conflict' else 'status unknown'}).",
                    sources=srcs, values={"drug": drug, "structured": struct, "latest_document": doc_status},
                    action=f"Confirm current {drug} use (pharmacy fill history / patient / prescriber)."))

    for drug in taking:
        plans = [p for p in ex.anticoag_plans if canonical_anticoagulant(p.drug_as_written) == drug]
        other = [p for p in ex.anticoag_plans if canonical_anticoagulant(p.drug_as_written) not in (None, drug)]
        srcs = [doc_src(p.doc_id, p.quote) for p in plans or other]
        full = [p for p in plans if p.pre_op == "explicit" and p.post_op == "explicit"]
        if full:
            passed.append(f"Rule 3 {drug}: complete plan [{full[-1].doc_id}]")
        elif plans:
            p = plans[-1]
            gaps = [n for n, v in (("pre-op", p.pre_op), ("post-op", p.post_op)) if v != "explicit"]
            issues.append(_issue(
                "ANTICOAGULATION_MANAGEMENT", "Rule 3",
                f"Plan for {drug} is incomplete/ambiguous: {' and '.join(gaps)} handling not explicitly stated.",
                sources=srcs, values={"drug": drug, "pre_op": p.pre_op, "post_op": p.post_op},
                action=f"Obtain explicit hold and resume instructions for {drug} from the managing clinician."))
        else:
            why = (f"The only plan on file addresses {', '.join(sorted({canonical_anticoagulant(p.drug_as_written) or p.drug_as_written for p in other}))}, not {drug}."
                   if other else f"No perioperative plan documented for {drug}.")
            issues.append(_issue(
                "ANTICOAGULATION_MANAGEMENT", "Rule 3", why, sources=srcs, values={"drug": drug},
                action=f"Obtain a perioperative plan covering {drug} (when to stop, when to restart)."))
    if not taking and not any(i["rule"] == "Rule 3" for i in issues):
        passed.append("Rule 3: no current anticoagulant identified (antiplatelets do not require a plan)")

    # ---------------- Rule 4: acute safety ----------------
    bps, temps = [], []
    for v in sub.vitals:
        dt = _dt(v.date)
        if dt is None:
            continue
        src = {"source": "structured_vital", "id": None, "type": v.type, "date": v.date, "quote": None}
        if getattr(v, "systolic", None) is not None and getattr(v, "diastolic", None) is not None:
            bps.append((dt, v.systolic, v.diastolic, src))
        elif getattr(v, "value_f", None) is not None:
            temps.append((dt, float(v.value_f), src))
    for f in ex.vitals:
        dt = _dt(f.measured_date, f.measured_time)
        if dt is None:
            continue
        src = doc_src(f.doc_id, f.quote, f"{f.measured_date} {f.measured_time or ''}".strip())
        if f.kind == "blood_pressure" and f.systolic is not None and f.diastolic is not None:
            bps.append((dt, f.systolic, f.diastolic, src))
        elif f.kind == "temperature" and f.temp_value is not None:
            tf = f.temp_value * 9 / 5 + 32 if f.temp_unit == "C" else f.temp_value
            src = dict(src, converted_from=f"{f.temp_value} {f.temp_unit}" if f.temp_unit == "C" else None)
            temps.append((dt, tf, src))

    if bps:
        dt, s, d_, src = max(bps, key=lambda x: (x[0], x[1] >= 180 or x[2] >= 110))
        hits = [t for t, v in (("systolic >= 180", s >= 180), ("diastolic >= 110", d_ >= 110)) if v]
        if hits:
            issues.append(_issue("ACUTE_SAFETY_EXCLUSION", "Rule 4", f"Most recent BP {s:g}/{d_:g} meets exclusion ({', '.join(hits)}).",
                                 sources=[src], values={"latest_bp": f"{s:g}/{d_:g}", "measured": str(dt), "thresholds": "SBP>=180 or DBP>=110"},
                                 action="Do not proceed; recheck/clinical review. A documented in-range recheck would supersede."))
        else:
            passed.append(f"Rule 4 BP: latest {s:g}/{d_:g} at {dt}")
    else:
        issues.append(_issue("MISSING_REQUIRED_DATA", "Rule 4", "No blood pressure recorded anywhere in the package.",
                             action="Obtain a pre-op blood pressure."))
    if temps:
        dt, tf, src = max(temps, key=lambda x: (x[0], x[1] > 100.4))
        if tf > 100.4 + 1e-9:
            issues.append(_issue("ACUTE_SAFETY_EXCLUSION", "Rule 4", f"Most recent temperature {tf:.1f} F exceeds 100.4 F.",
                                 sources=[src], values={"latest_temp_f": round(tf, 2), "measured": str(dt), "threshold_f": 100.4},
                                 action="Do not proceed; clinical review for fever. A documented in-range recheck would supersede."))
        else:
            passed.append(f"Rule 4 temp: latest {tf:.1f} F at {dt}")
    else:
        issues.append(_issue("MISSING_REQUIRED_DATA", "Rule 4", "No temperature recorded anywhere in the package (structured vitals or documents).",
                             action="Obtain a pre-op temperature."))

    # ---------------- extraction integrity ----------------
    if dropped:
        issues.append(_issue(
            "MISSING_REQUIRED_DATA", "Extraction integrity",
            f"{len(dropped)} extracted fact(s) could not be verified against the source text and were discarded.",
            values={"discarded": dropped[:10]}, confidence="low",
            action="Review the source documents manually; automated extraction was unreliable for this case."))

    cats = {i["category"] for i in issues}
    decision = "NOT_CLEARED" if "ACUTE_SAFETY_EXCLUSION" in cats else ("NEEDS_FOLLOW_UP" if issues else "READY")
    return Result(decision, issues, passed, str(pdate) if pdate else None, notes)