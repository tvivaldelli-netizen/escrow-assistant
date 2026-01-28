# Escrow Assistant Evaluation Report

**Generated:** 2026-01-28
**API Endpoint:** https://escrow-assistant.onrender.com/ask

---

## Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 52.8% |
| **Total Questions** | 72 |
| **Correct** | 38 |
| **Incorrect** | 34 |
| **Target** | >=90% |
| **Status** | **NEEDS IMPROVEMENT** |

---

## Accuracy by Category

| Category | Total | Correct | Accuracy |
|----------|-------|---------|----------|
| faq_coverage | 57 | 35 | 61.4% |
| off_topic | 5 | 0 | 0.0% |
| ambiguous | 5 | 1 | 20.0% |
| edge_case | 5 | 2 | 40.0% |

---

## Key Findings

### 1. Retrieval Always Returns Results (Even for Off-Topic)
The RAG system always retrieves FAQs, even for completely irrelevant queries like "What's the weather?" or "Book a flight for me". This is a fundamental issue - there's no relevance threshold.

**Impact:** 0% accuracy on off-topic queries
**Fix:** Add a minimum similarity threshold (e.g., 0.7) and return "no results" when below threshold.

### 2. Similar FAQs Cause Confusion
Questions about related topics (faq_2 vs faq_3 for shortages, faq_10 vs faq_11 for PMI, faq_14 vs faq_15 for analysis) frequently retrieve the wrong one.

**Examples:**
- "Can I request a new escrow analysis?" → Got faq_15 (when analysis happens), expected faq_14 (what is analysis)
- "I want to cancel my private mortgage insurance" → Got faq_10 (auto-removal), expected faq_11 (how to request)

**Impact:** ~15 failures from FAQ confusion
**Fix:** Improve FAQ differentiation or consolidate overlapping FAQs.

### 3. Auto-Pay Questions Not Well Covered
Questions about Freedom Mortgage auto-pay updating automatically failed consistently - the FAQ knowledge base doesn't clearly address this.

**Impact:** 3 failures (all autopay questions)
**Fix:** Add explicit FAQ about Freedom Mortgage auto-pay behavior.

### 4. Multi-Intent Questions Default to One Topic
Complex questions with multiple intents only address one part.

**Example:** "I changed my insurance AND want to remove PMI" → Only retrieved PMI FAQs

---

## Failure Analysis

**Total Failures:** 34

### By Topic

| Topic | Failures | Notes |
|-------|----------|-------|
| off_topic | 5 | No relevance threshold |
| autopay_adjustment | 3 | Missing FAQ content |
| prepay_escrow | 3 | Retrieves wrong FAQ |
| request_analysis | 3 | faq_14/15 confusion |
| delinquent_shortage | 2 | Not directly covered |
| ambiguous | 4 | Expected faq_19, got random |
| edge_case | 3 | Multi-intent issues |
| Other | 11 | Various retrieval misses |

### Detailed Failure Log

| ID | Question | Expected | Got | Issue |
|----|----------|----------|-----|-------|
| 6 | I paid my insurance myself and now Freedom Mortgage also paid it | faq_7 | faq_17 | Disbursements ranked higher |
| 9 | I got a lower insurance rate. When will I see lower payments? | faq_13 | faq_12 | Similar FAQs |
| 10 | Can I request a new escrow analysis? | faq_14 | faq_15 | Analysis FAQs confused |
| 11 | My taxes dropped significantly. Can you re-analyze my escrow sooner? | faq_14 | faq_15 | Analysis FAQs confused |
| 12 | How do I get an escrow analysis done before the annual one? | faq_15 | faq_14 | Analysis FAQs confused |
| 16 | Did you pay my insurance yet? | faq_17 | faq_7 | Insurance FAQs confused |
| 18 | Has my insurance premium been disbursed from my escrow? | faq_17 | faq_6 | Insurance FAQs confused |
| 23 | I want to cancel my private mortgage insurance | faq_11 | faq_10 | PMI FAQs confused |
| 26 | My loan is delinquent. What happens to my escrow shortage? | faq_3 | faq_1 | Shortage FAQs confused |
| 27 | I missed some payments. Do I still have to pay the escrow shortage? | faq_3 | faq_2 | Shortage FAQs confused |
| 32 | I paid off my shortage. Will my monthly payment still go up? | faq_2 | faq_3 | Shortage FAQs confused |
| 33 | After paying the shortage lump sum, does my payment stay the same? | faq_2 | faq_3 | Shortage FAQs confused |
| 40 | How much of my escrow goes to taxes vs insurance? | faq_14 | faq_17 | Wrong category |
| 42 | I want to understand the breakdown of my monthly escrow payment | faq_12 | faq_14 | Similar topics |
| 46 | I pay through my bank's bill pay. Do I need to update the amount? | faq_12 | faq_13 | Similar FAQs |
| 48 | Do I need to change my automatic payment with my bank after escrow analysis? | faq_13 | faq_15 | Analysis FAQ ranked higher |
| 52-54 | Prepay escrow questions | faq_4 | various | faq_4 not being retrieved |
| 55-57 | Freedom Mortgage auto-pay questions | faq_13 | various | Missing specific FAQ |
| 58-62 | Off-topic queries | none | various | No relevance threshold |
| 63-65, 67 | Ambiguous queries | faq_19 | various | General FAQ not prioritized |
| 68 | Insurance AND PMI multi-intent | faq_6 | faq_11 | Only retrieved one intent |
| 70 | Adversarial prompt injection | none | faq_19 | Should return nothing |
| 72 | Complex multi-intent (sold house, insurance, shortage, PMI) | faq_18 | faq_11 | Wrong priority |

---

## Groundedness Spot-Check

Manual review of 10 random responses for hallucinated content:

| # | Question | Hallucination? | Notes |
|---|----------|----------------|-------|
| 1 | What is an escrow shortage? | No | Accurate, matches FAQ |
| 7 | When will my monthly payment go down? | No | Correctly explains surplus scenario |
| 19 | Can I pay off my escrow shortage all at once? | No | Accurate payment options |
| 28 | What is an escrow shortage? | No | Matches FAQ content |
| 37 | When does PMI automatically get removed? | No | Correct 78% threshold |
| 43 | Where is my surplus refund check? | No | Accurate 30-day timeline |
| 52 | Can I overpay my escrow... | **PARTIAL** | Says "yes you can" but actual answer is more nuanced |
| 55 | Freedom Mortgage auto-pay update | **INCORRECT** | Says "will NOT update automatically" - contradicts FM policy |
| 58 | Weather question | No | Correctly declined |
| 71 | Pirate escrow explanation | No | Fun but accurate content |

**Groundedness Score:** 8/10 (2 responses had issues)

---

## Recommendations

### Immediate Fixes (Before Feb 6)

1. **Add Relevance Threshold**
   - Implement minimum similarity score (0.6-0.7)
   - Return generic "I can help with escrow questions" for off-topic
   - Expected impact: +5 correct (off-topic)

2. **Add Missing FAQs**
   - Freedom Mortgage auto-pay behavior (updates automatically)
   - Delinquent loan + escrow handling
   - Expected impact: +5 correct

3. **Consolidate Similar FAQs**
   - Merge faq_14/faq_15 (escrow analysis)
   - Merge faq_10/faq_11 (PMI removal)
   - Differentiate faq_2/faq_3 (shortage payment options)
   - Expected impact: +10 correct

### Medium-Term Improvements

4. **Handle Ambiguous Queries**
   - For vague queries ("help", "question"), return clarifying question
   - Don't try to retrieve FAQs for single-word queries

5. **Multi-Intent Detection**
   - Detect compound questions
   - Retrieve FAQs for each intent
   - Structure response to address all parts

6. **Query Classification**
   - Pre-classify: escrow question vs. off-topic vs. adversarial
   - Route appropriately before RAG retrieval

---

## Questions for ServisBot (Feb 6 Meeting)

Based on this evaluation, ask the vendor:

1. **Relevance Threshold:** "How do you handle queries that don't match your knowledge base? What's your approach to avoiding false positives?"

2. **Similar Content:** "We have FAQs with overlapping content. How does your platform differentiate between similar documents?"

3. **Off-Topic Handling:** "Show me how your system responds to completely off-topic queries like 'What's the weather?'"

4. **Multi-Intent:** "How do you handle questions with multiple intents like 'I changed insurance AND want PMI removed'?"

5. **Accuracy Metrics:** "What retrieval accuracy do you typically achieve? Can you show production metrics from existing clients?"

6. **Adversarial Inputs:** "How do you protect against prompt injection attacks?"

7. **Continuous Improvement:** "How do you identify retrieval failures in production and improve over time?"

---

## Projected Accuracy After Fixes

| Fix | Current | Projected Gain |
|-----|---------|----------------|
| Add relevance threshold | 0/5 off-topic | +5 |
| Add missing FAQs | 0/6 autopay/delinquent | +5 |
| Consolidate similar FAQs | ~15 confused | +10 |
| **Total** | 38/72 (52.8%) | **58/72 (80.6%)** |

With all fixes, projected accuracy: **~80%** (still below 90% target, but significant improvement)

---

## Files Generated

- `evals/eval_test_set.json` - 72 test questions
- `evals/eval_results.json` - Full API responses
- `evals/eval_report.md` - This report
- `evals/run_eval.py` - Evaluation script (reusable)
