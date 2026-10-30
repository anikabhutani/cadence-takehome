import json, sys
from core import PatientSubmission
from preop.llm_extract import extract
from preop.verify import verify
rows = [json.loads(l) for l in open("data/patients_sample_50.jsonl")]
for i in (0, 3, 35):
    sub = PatientSubmission.model_validate(rows[i]["submission"])
    ex = extract(sub, "gpt-4.1-mini")
    v = verify(ex, {d.doc_id: d.text or "" for d in sub.documents})
    print(f"\n===== case {i} =====")
    print("DROPPED:")
    for d in v.dropped: print("  ", d)
    if i == 0:
        print("PLANS:")
        for p in ex.anticoag_plans: print("  ", p.model_dump_json())
        print("SOURCE LINES:")
        for d in sub.documents:
            for ln in d["text"].split("\n") if isinstance(d, dict) else d.text.split("\n"):
                if "Xarelto" in ln and ln.strip()[:2] in ("1.", "2.", "3."): print("  ", ln)
