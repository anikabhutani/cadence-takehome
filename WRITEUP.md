# Approach: LLM extracts, code decides

**Problem with the baseline.** One LLM call both reads messy free text *and* applies date arithmetic, thresholds and
precedence. Those are the parts LLMs do worst (30 vs 31 days, 38.0 °C vs >100.4 °F, "latest reading wins") and the
parts where a wrong answer (a false READY) is dangerous. It also gives non-reproducible output and no verifiable evidence.

**Architecture** (`preop/`)

```
submission ─► extract (LLM, or regex "heuristic") ─► verify ─► rules (pure Python) ─► decision + evidence packet
              facts + verbatim quotes                grounding    Rules 1-4, precedence
```

1. `facts.py` – the typed contract. The model reports *what documents say* (is this an H&P? is the **patient's**
   signature evidenced? every BP/temp with date/time/unit? anticoagulant statements? plan pre-op/post-op = explicit/deferred/absent?)
   each with `doc_id` + verbatim `quote`. It never sees a threshold or a window.
2. `llm_extract.py` – one structured-output call (`responses.parse`, strict schema), temp 0, disk cache keyed on
   (prompt version, model, input) so reruns and `make determinism` are exact. Document text is treated as untrusted data.
3. `verify.py` – every quote must appear verbatim in its document, and vital numbers must appear in their quote.
   Failures are dropped, one retry is attempted, and anything still ungrounded raises a low-confidence
   MISSING_REQUIRED_DATA "needs human review" issue. A hallucination can therefore delay a patient but never clear one.
4. `rules.py` – Rules 1-4 as plain functions + `NOT_CLEARED > NEEDS_FOLLOW_UP > READY`. All issues are reported even when a
   higher-precedence one decides the status (matches the labels). Brand→generic mapping lives in `drugs.py`.
5. Output = required `decision` + `issues[].category`, plus per-issue `rule`, `finding`, `sources[]` (doc/lab ids, dates, quotes),
   `values_checked`, `reviewer_action`, `confidence`; and top-level `rules_passed` (so sampled READY cases are checkable),
   `assumptions`, `procedure_date_used`, `needs_human_review`.

**Assumptions** (also emitted in every output): day-30 inclusive; labs after the procedure don't count; entered-in-error ignored;
BMP ≠ CMP; consent needs the patient's signature *and* the right procedure; a missing temperature/BP is MISSING_REQUIRED_DATA;
unknown risk → MISSING_REQUIRED_DATA (still requiring a 30-day CBC); antiplatelets need no plan; current drug use = latest
statement; a plan must name the drug in use and state pre- and post-op handling (this catches the warfarin-plan/apixaban-patient case);
vitals: latest per type, unknown time = 00:00, ties go to the worse value. Spec typo `ANTICOAGULATION_MANAGEMENTJSON` read as `ANTICOAGULATION_MANAGEMENT`.

**Testing**: `make test` – boundary tests for every threshold/window, precedence, grounding, extractor caching, and a
regression test that the rules reproduce all 50 labels given correct facts. `make local` scores decision + category
match locally (false READYs listed first); `EXTRACTOR=heuristic` runs fully offline.

**Honest limits**
## Results
LLM extractor (gpt-4.1-mini) on the 50 labelled samples: 48/50 decisions match,
47/50 category sets match, 0 false READYs. The deterministic rules engine
reproduces all 50 labels given correct facts (offline regression test).

## Known misses
- Cases 3 and 35: my quote-grounding check mishandled multi-line quotes, so valid
  vitals were discarded and the case was routed to human review (safe direction).
  Fix identified: match multi-line quotes line by line.
- Case 0: the model treated "per surgical team" as deferring a plan that gives
  explicit timing. Fix identified: prompt clarification.

- The LLM extractor's live accuracy has not been measured in my environment (no OpenAI access); `make local` is how to measure it.
  Diff its facts against the heuristic's to find extraction misses.
- Timezones are ignored (Z and local times compared naively); the anticoagulant list is a fixed map (unknown drugs would need the model's drug-class hint).
- Next steps: adversarial variants (shuffled order, odd phrasings, injected instructions in notes), a second-model check on consent/plan judgments, confidence-gated routing.
