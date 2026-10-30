"""The contract between extraction and the rules engine.

Everything a rule needs from free text is one of these facts. Each fact carries
``doc_id`` + a verbatim ``quote`` so (a) a nurse can verify it in seconds and
(b) ``verify.py`` can mechanically reject anything the extractor made up.

The extractor only *reports what the document says*. It never applies policy
(no "is this within 30 days", no "is this a fever") - that is rules.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# OpenAI strict structured outputs want every field present, so optional values
# are `X | None` with no defaults on the LLM-facing models.


class HPFact(BaseModel):
    doc_id: str
    is_history_and_physical: bool = Field(
        description="True only if the document itself IS a completed History & Physical "
        "(history + exam written by a clinician). False for notes that merely mention or "
        "reference an H&P, e.g. anesthesia/nursing notes saying 'H&P reviewed' or 'not yet on file'."
    )
    date_of_service: str | None = Field(
        description="ISO date (YYYY-MM-DD) of the encounter as written in the text, else null."
    )
    encounter_kind: Literal["pre_op", "wellness_or_establish_care", "other"]
    quote: str = Field(description="Verbatim heading or date line from the document.")


class ConsentFact(BaseModel):
    doc_id: str
    procedure_named: str | None = Field(description="Procedure the consent form is for, as written.")
    matches_planned_procedure: bool | None = Field(
        description="True/False if the named procedure is/is not the planned procedure given in the prompt; null if unclear."
    )
    patient_signature: Literal["signed", "pending_or_blank", "not_obtained", "unclear"] = Field(
        description="'signed' ONLY if the PATIENT's signature is evidenced (signature on file, /s/, "
        "'signed by patient', scanned signed copy). A physician attestation alone is NOT a patient signature. "
        "'pending_or_blank' for blank line, 'pending', not completed, printed for review."
    )
    signed_date: str | None = Field(description="ISO date the patient signed, if stated.")
    quote: str = Field(description="Verbatim line showing the signature status.")


class VitalFact(BaseModel):
    doc_id: str
    kind: Literal["blood_pressure", "temperature"]
    systolic: float | None
    diastolic: float | None
    temp_value: float | None = Field(description="Temperature number exactly as written.")
    temp_unit: Literal["F", "C"] | None = Field(description="Unit exactly as written.")
    measured_date: str = Field(description="ISO date of measurement (visit date if not stated otherwise).")
    measured_time: str | None = Field(
        description="HH:MM 24h. For a recheck written as 'N minutes later', compute from the earlier reading's time. Null if unknowable."
    )
    is_recheck: bool = Field(description="True for repeat/re-measured/addendum readings.")
    quote: str = Field(description="Verbatim text containing the numbers.")


class MedClaim(BaseModel):
    doc_id: str
    drug_as_written: str
    status: Literal["taking", "not_taking", "discontinued_or_historical", "unclear"] = Field(
        description="Current-use status per THIS document as of its date."
    )
    quote: str


class AnticoagPlanFact(BaseModel):
    doc_id: str
    drug_as_written: str
    plan_date: str | None
    pre_op: Literal["explicit", "absent", "deferred"] = Field(
        description="'explicit' = states when to stop/hold (or explicitly to continue). 'deferred' = hands decision to someone else / 'will follow' / 'to be determined'."
    )
    post_op: Literal["explicit", "absent", "deferred"] = Field(
        description="'explicit' = states when/how to resume. 'deferred' = 'will be addressed after surgery', 'per surgical team', etc."
    )
    quote: str = Field(description="Verbatim recommendation text (pre/post).")


class ProcedureDateFact(BaseModel):
    doc_id: str
    date: str | None = Field(description="ISO date only if a definite date is written; null if 'pending'/'TBD'.")
    quote: str


class Extraction(BaseModel):
    hp_documents: list[HPFact]
    consents: list[ConsentFact]
    vitals: list[VitalFact]
    med_claims: list[MedClaim]
    anticoag_plans: list[AnticoagPlanFact]
    procedure_dates: list[ProcedureDateFact]
