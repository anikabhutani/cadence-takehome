"""Offline regex extractor producing the same ``Extraction`` as the LLM extractor.

Purpose: (1) run the full pipeline with no API key, (2) give the rules engine a
regression harness against the labelled sample, (3) a deterministic baseline to
compare the LLM extractor against. It is tuned to the phrasing in the sample data;
the LLM extractor is the one meant to generalise.
"""

from __future__ import annotations

import re
from datetime import date

from .drugs import canonical_anticoagulant, find_anticoagulants
from .facts import (AnticoagPlanFact, ConsentFact, Extraction, HPFact, MedClaim,
                    ProcedureDateFact, VitalFact)

_US_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_BP = re.compile(r"(?:(?:NIBP|BP|Blood pressure)\s*:?\s*|(?=\d{2,3}/\d{2,3}\s*mmHg))(\d{2,3})/(\d{2,3})(?![\d/])(?:\s*mmHg)?")
_TEMP = re.compile(r"(?:(?:Temperature|Temp|T)\s*:?\s*)?(\d{2,3}\.\d)\s*°?\s*([CF])\b")


def _iso(us: str) -> str | None:
    m = _US_DATE.search(us)
    return date(int(m.group(3)), int(m.group(1)), int(m.group(2))).isoformat() if m else None


def _base_date(text: str, fallback: str | None) -> str | None:
    m = re.search(r"(?:Visit date|Date of service|Date):\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    return _iso(m.group(1)) if m else fallback


def _norm_proc(s: str) -> str:
    return re.sub(r"^elective\s+", "", s.lower()).strip()


def extract(sub) -> Extraction:
    ex = Extraction(hp_documents=[], consents=[], vitals=[], med_claims=[], anticoag_plans=[], procedure_dates=[])
    planned = _norm_proc((sub.procedure.procedure_type if sub.procedure else "") or "")

    for d in sub.documents:
        text, did = d.text or "", d.doc_id
        lines = [ln.rstrip() for ln in text.split("\n")]
        first = next((ln.strip() for ln in lines if ln.strip()), "")
        base = _base_date(text, d.date)

        # ---- H&P ----
        if re.search(r"HISTORY (AND|&) PHYSICAL|^H&P\b", first, re.I):
            m = re.search(r"Date of service:\s*(\d{1,2}/\d{1,2}/\d{4})", text)
            kind = "wellness_or_establish_care" if re.search(r"establish care|wellness", text, re.I) else "pre_op"
            ex.hp_documents.append(HPFact(doc_id=did, is_history_and_physical=True,
                                          date_of_service=_iso(m.group(1)) if m else d.date,
                                          encounter_kind=kind, quote=first))

        # ---- consent ----
        if re.search(r"CONSENT", first):
            pm = re.search(r"^Procedure:\s*(.+)$", text, re.M)
            proc = pm.group(1).strip() if pm else None
            match = None
            if proc and planned:
                a, b = _norm_proc(proc), planned
                match = a in b or b in a
            sig_line, status, sdate = None, "unclear", None
            for ln in lines:
                s = ln.strip()
                if re.search(r"signature on file - signed|/s/|Signed by patient:", s):
                    sig_line, status, sdate = s, "signed", _iso(s)
                    break
            if status != "signed":
                for ln in lines:
                    s = ln.strip()
                    if re.search(r"signature: _+|signature: pending|NOT COMPLETED|not yet signed|pending", s, re.I) and "Physician" not in s:
                        sig_line, status = s, "pending_or_blank"
                        break
            if sig_line:
                ex.consents.append(ConsentFact(doc_id=did, procedure_named=proc, matches_planned_procedure=match,
                                               patient_signature=status, signed_date=sdate, quote=sig_line))

        # ---- procedure date mentions ----
        for m in re.finditer(r"scheduled for (\d{1,2}/\d{1,2}/\d{4})", text):
            ex.procedure_dates.append(ProcedureDateFact(doc_id=did, date=_iso(m.group(1)), quote=m.group(0)))
        for m in re.finditer(r"date to be determined pending OR availability|date is (?:still )?pending[^.\n]*", text, re.I):
            ex.procedure_dates.append(ProcedureDateFact(doc_id=did, date=None, quote=m.group(0)))

        # ---- vitals ----
        last_time = None
        for ln in lines:
            if re.search(r"not recorded|Deferred", ln):
                continue
            events = []  # (pos, kind, match)
            for m in _BP.finditer(ln):
                events.append((m.start(), "bp", m))
            for m in _TEMP.finditer(ln):
                if "°F temporal" in ln[m.start():m.end() + 10] or True:
                    events.append((m.start(), "temp", m))
            if not events:
                continue
            times = [(t.start(), f"{int(t.group(1)):02d}:{t.group(2)}") for t in _TIME.finditer(ln)]
            for pos, kind, m in sorted(events, key=lambda e: e[0]):
                before = [t for p, t in times if p < pos]
                tm = before[-1] if before else last_time
                recheck = bool(re.search(r"(?i)addendum|repeat|re-?measured|rechecked", ln[:pos]))
                q = m.group(0).strip()
                if kind == "bp":
                    ex.vitals.append(VitalFact(doc_id=did, kind="blood_pressure", systolic=float(m.group(1)),
                                               diastolic=float(m.group(2)), temp_value=None, temp_unit=None,
                                               measured_date=base or "", measured_time=tm, is_recheck=recheck, quote=q))
                else:
                    ex.vitals.append(VitalFact(doc_id=did, kind="temperature", systolic=None, diastolic=None,
                                               temp_value=float(m.group(1)), temp_unit=m.group(2),
                                               measured_date=base or "", measured_time=tm, is_recheck=recheck, quote=q))
            if times:
                last_time = times[-1][1]

        # ---- medication claims ----
        for ln in lines:
            s = ln.strip()
            drugs = find_anticoagulants(s)
            if not drugs:
                continue
            low = s.lower()
            if re.match(r"^\d+\.\s", s) or low.startswith(("assessment", "reason for consult", "hpi")) and "maintained on" not in low:
                continue
            if "replaced with" in low or "replaced by" in low:
                for dr in drugs:
                    first_drug = canonical_anticoagulant(s)
                    ex.med_claims.append(MedClaim(doc_id=did, drug_as_written=dr,
                                                  status="discontinued_or_historical" if dr == first_drug else "taking", quote=s))
                continue
            if re.search(r"unable to confirm|not sure|unsure", low):
                st = "unclear"
            elif re.search(r"discontinued|stopped|no longer|not currently anticoag", low):
                st = "discontinued_or_historical"
            elif re.search(r"taking|^- |maintained on", low):
                st = "taking"
            else:
                continue
            for dr in drugs:
                ex.med_claims.append(MedClaim(doc_id=did, drug_as_written=dr, status=st, quote=s))

        # ---- anticoagulation plans ----
        for ln in lines:
            s = ln.strip()
            if not re.match(r"^\d+\.\s", s) or not canonical_anticoagulant(s):
                continue
            if not re.search(r"(?i)hold|stop|periprocedural|interruption", s):
                continue
            drug = canonical_anticoagulant(s)
            deferred = re.search(r"(?i)to be determined|will follow up|discuss interruption|specific recommendations", s)
            pre = "deferred" if deferred else ("explicit" if re.search(r"(?i)(hold|stop)\b.*\b(days?|hours?)|last dose", s) else "absent")
            if deferred:
                post = "deferred"
            elif re.search(r"(?i)will be addressed|addressed after", s):
                post = "deferred"
            elif re.search(r"(?i)\b(resume|restart)", s):
                post = "explicit"
            else:
                post = "absent"
            ex.anticoag_plans.append(AnticoagPlanFact(doc_id=did, drug_as_written=drug, plan_date=d.date,
                                                      pre_op=pre, post_op=post, quote=s))
    return ex
