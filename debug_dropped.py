import json, collections
from core import PatientSubmission
from preop.llm_extract import extract
from preop.verify import verify
rows = [json.loads(l) for l in open("data/patients_sample_50.jsonl")]
kinds = collections.Counter()
for i, r in enumerate(rows):
    sub = PatientSubmission.model_validate(r["submission"])
    v = verify(extract(sub, "gpt-4.1-mini"), {d.doc_id: d.text or "" for d in sub.documents})
    for d in v.dropped:
        kinds[d.split(":")[0] + " / " + ("quote" if "quote not found" in d else "values" if "values not present" in d else "doc_id")] += 1
        if sum(kinds.values()) <= 12:
            print(f"[{i}]", d)
print(kinds)
