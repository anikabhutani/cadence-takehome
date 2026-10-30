"""Rules-engine boundary tests (no LLM), grounding checks, extractor plumbing, and sample regression."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import PatientSubmission, triage_submission
from preop.drugs import canonical_anticoagulant
from preop.facts import (AnticoagPlanFact, ConsentFact, Extraction, HPFact, MedClaim, VitalFact)
from preop.rules import evaluate
from preop.verify import verify

PD = "2026-03-30"


def sub(risk="LOW", pdate=PD, labs=None, meds=None, vitals=None, docs=None):
    return PatientSubmission.model_validate({
        "procedure": {"procedure_type": "hernia repair", "procedure_risk": risk, "procedure_date": pdate},
        "labs": labs if labs is not None else [{"id": "l1", "code": "CBC", "display": "CBC", "effective_at": "2026-03-20T08:00:00Z", "status": "final"}],
        "medications": meds or [],
        "vitals": vitals if vitals is not None else [
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-25T09:00:00Z"},
            {"type": "temperature", "value_f": 98.6, "date": "2026-03-25T09:00:00Z"}],
        "documents": docs or [{"doc_id": "hp", "type": "H&P", "date": "2026-03-10", "text": "H&P"},
                              {"doc_id": "cn", "type": "Consent", "date": "2026-03-12", "text": "signed"}],
    })


def ex(**kw):
    base = dict(
        hp_documents=[HPFact(doc_id="hp", is_history_and_physical=True, date_of_service="2026-03-10", encounter_kind="pre_op", quote="H&P")],
        consents=[ConsentFact(doc_id="cn", procedure_named="hernia repair", matches_planned_procedure=True, patient_signature="signed", signed_date="2026-03-12", quote="signed")],
        vitals=[], med_claims=[], anticoag_plans=[], procedure_dates=[])
    base.update(kw)
    return Extraction(**base)


def cats(res):
    return sorted({i["category"] for i in res.issues})


def test_all_clear_is_ready():
    r = evaluate(sub(), ex(), [])
    assert r.decision == "READY" and r.issues == [] and r.rules_passed


@pytest.mark.parametrize("hp_date,ok", [("2026-02-28", True), ("2026-02-27", False)])  # day 30 inclusive, 31 fails
def test_hp_window_boundary(hp_date, ok):
    e = ex(hp_documents=[HPFact(doc_id="hp", is_history_and_physical=True, date_of_service=hp_date, encounter_kind="pre_op", quote="H&P")])
    assert (evaluate(sub(), e, []).decision == "READY") is ok


@pytest.mark.parametrize("lab_date,risk,ok", [
    ("2026-03-16", "HIGH", True), ("2026-03-15", "HIGH", False), ("2026-02-28", "LOW", True), ("2026-02-27", "MODERATE", False)])
def test_cbc_window_by_risk(lab_date, risk, ok):
    labs = [{"id": "c", "code": "CBC", "display": "CBC", "effective_at": lab_date + "T08:00:00Z", "status": "final"},
            {"id": "m", "code": "CMP", "display": "Comprehensive Metabolic Panel", "effective_at": "2026-03-28T08:00:00Z", "status": "final"}]
    assert (evaluate(sub(risk=risk, labs=labs), ex(), []).decision == "READY") is ok


def test_entered_in_error_and_bmp_do_not_count():
    labs = [{"id": "a", "code": "CBC", "display": "CBC", "effective_at": "2026-03-28T08:00:00Z", "status": "entered-in-error"},
            {"id": "b", "code": "BMP", "display": "Basic Metabolic Panel", "effective_at": "2026-03-28T08:00:00Z", "status": "final"}]
    assert cats(evaluate(sub(labs=labs), ex(), [])) == ["REQUIRED_TESTING"]
    # HIGH risk: BMP must not satisfy CMP
    labs = [{"id": "a", "code": "CBC", "display": "CBC", "effective_at": "2026-03-28T08:00:00Z", "status": "final"}] + labs[1:]
    r = evaluate(sub(risk="HIGH", labs=labs), ex(), [])
    assert any("CMP" in i["finding"] for i in r.issues)


def test_unknown_risk_and_missing_date_are_missing_data_not_ready():
    assert cats(evaluate(sub(risk=None), ex(), [])) == ["MISSING_REQUIRED_DATA"]
    r = evaluate(sub(pdate=None), ex(), [])
    assert cats(r) == ["MISSING_REQUIRED_DATA"] and r.decision == "NEEDS_FOLLOW_UP"


def test_vital_thresholds():
    def v(s, d, t):
        return [{"type": "blood_pressure", "systolic": s, "diastolic": d, "date": "2026-03-25T09:00:00Z"},
                {"type": "temperature", "value_f": t, "date": "2026-03-25T09:00:00Z"}]
    assert evaluate(sub(vitals=v(179, 109, 100.4)), ex(), []).decision == "READY"   # strictly below / not above
    for bad in (v(180, 80, 98.6), v(120, 110, 98.6), v(120, 80, 100.5)):
        assert evaluate(sub(vitals=bad), ex(), []).decision == "NOT_CLEARED"


def test_recheck_supersedes_and_celsius_converted():
    recheck = VitalFact(doc_id="hp", kind="blood_pressure", systolic=137, diastolic=78, temp_value=None, temp_unit=None,
                        measured_date="2026-03-26", measured_time="08:30", is_recheck=True, quote="137/78")
    s = sub(vitals=[{"type": "blood_pressure", "systolic": 190, "diastolic": 100, "date": "2026-03-26T08:06:00Z"},
                    {"type": "temperature", "value_f": 98.6, "date": "2026-03-26T08:06:00Z"}],
            docs=[{"doc_id": "hp", "type": "H&P", "date": "2026-03-10", "text": "137/78"}, {"doc_id": "cn", "type": "C", "date": "2026-03-12", "text": "signed"}])
    assert evaluate(s, ex(vitals=[recheck]), []).decision == "READY"
    fever_c = VitalFact(doc_id="hp", kind="temperature", systolic=None, diastolic=None, temp_value=38.1, temp_unit="C",
                        measured_date="2026-03-27", measured_time=None, is_recheck=False, quote="38.1 C")
    assert evaluate(sub(), ex(vitals=[fever_c]), []).decision == "NOT_CLEARED"  # 100.58 F
    ok_c = fever_c.model_copy(update={"temp_value": 38.0})
    assert evaluate(sub(), ex(vitals=[ok_c]), []).decision == "READY"            # 100.4 F exactly


def test_missing_vital_types():
    r = evaluate(sub(vitals=[{"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-25T09:00:00Z"}]), ex(), [])
    assert cats(r) == ["MISSING_REQUIRED_DATA"]


def plan(drug, pre="explicit", post="explicit"):
    return AnticoagPlanFact(doc_id="hp", drug_as_written=drug, plan_date="2026-03-10", pre_op=pre, post_op=post, quote="H&P")


def test_anticoag_matrix():
    meds = [{"name": "Eliquis 5 mg BID", "active": True}]
    assert cats(evaluate(sub(meds=meds), ex(), [])) == ["ANTICOAGULATION_MANAGEMENT"]                       # no plan
    assert cats(evaluate(sub(meds=meds), ex(anticoag_plans=[plan("apixaban", post="deferred")]), [])) == ["ANTICOAGULATION_MANAGEMENT"]
    assert cats(evaluate(sub(meds=meds), ex(anticoag_plans=[plan("warfarin")]), [])) == ["ANTICOAGULATION_MANAGEMENT"]  # wrong drug
    assert evaluate(sub(meds=meds), ex(anticoag_plans=[plan("Eliquis")]), []).decision == "READY"           # brand == generic
    assert evaluate(sub(meds=[{"name": "ASA 81 mg", "active": True}]), ex(), []).decision == "READY"        # antiplatelet
    assert evaluate(sub(meds=[{"name": "warfarin 5 mg", "active": False}]), ex(), []).decision == "READY"   # inactive
    unclear = [{"name": "Coumadin 5 mg", "active": None}]
    assert cats(evaluate(sub(meds=unclear), ex(), [])) == ["MISSING_REQUIRED_DATA"]
    claim = MedClaim(doc_id="hp", drug_as_written="Xarelto", status="taking", quote="H&P")                    # only in free text
    assert cats(evaluate(sub(), ex(med_claims=[claim]), [])) == ["ANTICOAGULATION_MANAGEMENT"]


def test_consent_requires_patient_signature_for_this_procedure():
    for kw in ({"patient_signature": "pending_or_blank"}, {"matches_planned_procedure": False}):
        c = ex().consents[0].model_copy(update=kw)
        assert cats(evaluate(sub(), ex(consents=[c]), [])) == ["REQUIRED_DOCUMENTATION"]
    assert cats(evaluate(sub(), ex(consents=[]), [])) == ["REQUIRED_DOCUMENTATION"]


def test_precedence_not_cleared_beats_followup_and_keeps_all_issues():
    r = evaluate(sub(vitals=[{"type": "blood_pressure", "systolic": 200, "diastolic": 80, "date": "2026-03-25T09:00:00Z"},
                             {"type": "temperature", "value_f": 98.6, "date": "2026-03-25T09:00:00Z"}]), ex(consents=[]), [])
    assert r.decision == "NOT_CLEARED" and cats(r) == ["ACUTE_SAFETY_EXCLUSION", "REQUIRED_DOCUMENTATION"]


def test_unverified_extraction_forces_human_review():
    r = evaluate(sub(), ex(), ["vital: quote not found"])
    assert r.decision == "NEEDS_FOLLOW_UP" and r.issues[0]["confidence"] == "low"


def test_verify_drops_fabricated_quotes_and_numbers():
    v = VitalFact(doc_id="d", kind="blood_pressure", systolic=157, diastolic=117, temp_value=None, temp_unit=None,
                  measured_date="2026-03-03", measured_time=None, is_recheck=False, quote="BP 157/117")
    only = lambda vit: ex(hp_documents=[], consents=[], vitals=[vit])  # noqa: E731
    good = verify(only(v), {"d": "09:05  BP 157/117, Temp 36.7"})
    assert good.clean and len(good.extraction.vitals) == 1
    assert not verify(only(v), {"d": "09:05  BP 120/80"}).clean                      # quote absent
    wrong = v.model_copy(update={"systolic": 117, "diastolic": 80})
    assert not verify(only(wrong), {"d": "BP 157/117"}).clean                          # numbers don't match quote


def test_drug_normalisation():
    assert canonical_anticoagulant("Coumadin 5 mg PO daily") == "warfarin"
    assert canonical_anticoagulant("edoxaban (Savaysa) 60 mg") == "edoxaban"
    assert canonical_anticoagulant("ASA 81 mg") is None and canonical_anticoagulant("clopidogrel 75") is None


def test_llm_extractor_caches_and_is_deterministic(tmp_path, monkeypatch):
    import sys
    from preop import llm_extract
    monkeypatch.setattr(llm_extract, "CACHE_DIR", tmp_path)
    calls = []
    payload = ex()

    class FakeResponses:
        def parse(self, **kw):
            calls.append(kw)
            return SimpleNamespace(output_parsed=payload)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: SimpleNamespace(responses=FakeResponses())))
    s = sub()
    a = llm_extract.extract(s, "gpt-4.1-mini")
    b = llm_extract.extract(s, "gpt-4.1-mini")
    assert a == b and len(calls) == 1                      # second call served from cache
    assert calls[0]["temperature"] == 0 and calls[0]["text_format"] is Extraction


SAMPLE = Path(__file__).resolve().parent.parent / "data" / "patients_sample_50.jsonl"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample data missing")
def test_rules_engine_matches_all_sample_labels_with_heuristic_extractor(monkeypatch):
    """Regression guard: with correct facts, the deterministic engine reproduces every human label."""
    monkeypatch.setenv("PREOP_EXTRACTOR", "heuristic")
    for line in SAMPLE.read_text().splitlines():
        row = json.loads(line)
        out = triage_submission(row["submission"], model="unused")
        assert out.decision == row["label"]["decision"], row["case_id"]
        assert sorted({i.category for i in out.issues}) == sorted(set(row["label"]["categories"])), row["case_id"]


def test_verify_snaps_garbled_quote_to_verbatim_source_line():
    v = VitalFact(doc_id="d", kind="temperature", systolic=None, diastolic=None, temp_value=36.7, temp_unit="C",
                  measured_date="2026-03-03", measured_time="09:05", is_recheck=False, quote="09:05  Temp 36.7 \x176C")
    res = verify(ex(hp_documents=[], consents=[], vitals=[v]), {"d": "09:05  BP 157/117, Temp 36.7 \u00b0C\nHR 80"})
    assert res.clean and res.extraction.vitals[0].quote == "09:05  BP 157/117, Temp 36.7 \u00b0C"
    # snapping must never rescue a wrong number
    bad = v.model_copy(update={"temp_value": 39.9})
    assert not verify(ex(hp_documents=[], consents=[], vitals=[bad]), {"d": "09:05  BP 157/117, Temp 36.7 \u00b0C"}).clean