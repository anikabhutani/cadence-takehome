#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2.0.0", "pydantic>=2.8.0"]
# ///
"""Score triage_submission against the labels locally (no OpenAI Evals API needed).

Prints decision/category accuracy and every mismatch, false READYs first (the dangerous ones).
Use PREOP_EXTRACTOR=heuristic for an offline run.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from core import PatientSubmission, triage_submission

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/patients_sample_50.jsonl")
p.add_argument("--model", default="gpt-4.1-mini")
p.add_argument("--verbose", action="store_true")
a = p.parse_args()

rows = [json.loads(l) for l in Path(a.input).read_text().splitlines() if l.strip()]
dec_ok = cat_ok = 0
bad = []
for i, r in enumerate(rows):
    out = triage_submission(PatientSubmission.model_validate(r["submission"]), model=a.model)
    lab = r["label"]
    got_cats = sorted({x.category for x in out.issues})
    d, c = out.decision == lab["decision"], got_cats == sorted(set(lab["categories"]))
    dec_ok += d; cat_ok += c
    if not (d and c):
        bad.append((out.decision == "READY" and lab["decision"] != "READY", i, r["case_id"], lab, out, got_cats))
n = len(rows)
print(f"decision match: {dec_ok}/{n}   categories match: {cat_ok}/{n}")
for false_ready, i, cid, lab, out, got in sorted(bad, key=lambda b: (not b[0], b[1])):
    print(f"\n[{i}] {cid} {'** FALSE READY **' if false_ready else ''}")
    print(f"  expected {lab['decision']} {lab['categories']}")
    print(f"  got      {out.decision} {got}")
    for iss in out.issues:
        print(f"    - {iss.category}: {iss.finding}")
    if a.verbose:
        print("  label:", lab["rationale"])
