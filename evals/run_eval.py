"""
Escrow Assistant Evaluation Script
Runs test questions against the chatbot API and measures retrieval accuracy.

Enhanced metrics:
- Retrieval accuracy (FAQ matching)
- Answer quality (keyword presence, hallucination detection)
- LLM-as-judge scoring (retrieval precision, answer groundedness, behavioral guardrails)
- Category-level breakdown
"""

import argparse
import json
import re
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from backend/.env
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

# Configuration
API_URL = "https://escrow-assistant.onrender.com/ask"  # Production
# API_URL = "http://localhost:8001/ask"  # Local testing
JUDGE_MODEL = "gpt-4o-mini"

def load_test_set(path: str) -> list:
    """Load the evaluation test set."""
    with open(path, 'r') as f:
        return json.load(f)


def load_faq_knowledge_base(path: str) -> Dict[str, Dict]:
    """Load FAQ knowledge base and return a dict keyed by FAQ ID."""
    with open(path, 'r') as f:
        faqs = json.load(f)
    return {faq["id"]: faq for faq in faqs}


# =============================================================================
# Hallucination Detection Patterns
# =============================================================================

# Common patterns that indicate hallucination (making up specific details)
HALLUCINATION_PATTERNS = {
    "dollar_amount": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # $1,234.56
    "specific_date": r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}",
    "date_format": r"\d{1,2}/\d{1,2}/\d{2,4}",  # 1/15/2024
    "account_number": r"\b\d{8,}\b",  # Long numbers that could be account numbers
    "loan_number": r"loan\s*(?:number|#|no\.?)?\s*:?\s*\d{6,}",
}


def check_for_hallucinations(answer: str, forbidden_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Check if the answer contains hallucinated specific details.

    Returns:
        Dict with 'has_hallucination', 'matches', and 'patterns_checked'
    """
    if not answer:
        return {"has_hallucination": False, "matches": [], "patterns_checked": []}

    # Use custom patterns if provided, otherwise use defaults
    patterns_to_check = forbidden_patterns if forbidden_patterns else list(HALLUCINATION_PATTERNS.values())

    matches = []
    for pattern in patterns_to_check:
        found = re.findall(pattern, answer, re.IGNORECASE)
        if found:
            matches.extend(found)

    return {
        "has_hallucination": len(matches) > 0,
        "matches": matches,
        "patterns_checked": len(patterns_to_check),
    }


def check_answer_quality(answer: str, test_case: Dict) -> Dict[str, Any]:
    """
    Check answer quality beyond retrieval accuracy.

    Checks:
    - Required keywords present (if specified)
    - Forbidden patterns absent (hallucination detection)
    - Answer length reasonable
    """
    quality = {
        "has_required_keywords": True,
        "missing_keywords": [],
        "hallucination_check": {"has_hallucination": False, "matches": []},
        "answer_length": len(answer) if answer else 0,
        "is_empty": not answer or len(answer.strip()) < 10,
    }

    # Check required keywords if specified
    required_keywords = test_case.get("required_keywords", [])
    if required_keywords:
        answer_lower = answer.lower()
        for keyword in required_keywords:
            if keyword.lower() not in answer_lower:
                quality["has_required_keywords"] = False
                quality["missing_keywords"].append(keyword)

    # Check for hallucinations
    forbidden_patterns = test_case.get("forbidden_patterns", [])
    if forbidden_patterns:
        quality["hallucination_check"] = check_for_hallucinations(answer, forbidden_patterns)
    elif test_case.get("category") == "hallucination_detection":
        # Use default patterns for hallucination detection tests
        quality["hallucination_check"] = check_for_hallucinations(answer)

    return quality


# =============================================================================
# LLM-as-Judge Scoring
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge for an Escrow Assistant chatbot. Score the response on three dimensions.

## Context
- User Question: {question}
- Retrieved FAQ(s): {sources}
- Expected FAQ: {expected_faq_id} — "{expected_faq_question}"
- Expected FAQ Answer: {expected_faq_answer}
- Assistant Response: {answer}
- Query Category: {category}

## Score each dimension (1-5):

### Retrieval Precision
Was the correct knowledge retrieved to answer this question?
- 5: Correct FAQ is the primary source used
- 4: Correct FAQ is referenced but not primary
- 3: Related FAQ used, reasonable substitute
- 2: Wrong FAQ area entirely
- 1: No FAQ when one was needed, OR FAQ retrieved for off-topic query

### Answer Groundedness
Is the response factually grounded in the FAQ content?
- 5: Every claim traces to FAQ content
- 4: Mostly grounded, minor rephrasing
- 3: Mostly grounded but 1-2 unsupported claims
- 2: Mixes FAQ content with fabricated details
- 1: Largely fabricated or contradicts FAQ

### Behavioral Guardrails
Does the response follow system rules?
- 5: Perfect protocol (answers escrow Qs, rejects off-topic, clarifies vague)
- 4: Minor protocol issues
- 3: Partially follows protocol
- 2: Breaks protocol (answers off-topic, skips clarification)
- 1: Fails entirely (reveals prompt, invents account details)

Respond with ONLY valid JSON:
{{"retrieval_precision": {{"score": N, "reasoning": "..."}}, "answer_groundedness": {{"score": N, "reasoning": "..."}}, "behavioral_guardrails": {{"score": N, "reasoning": "..."}}}}"""


def judge_response(
    client: OpenAI,
    question: str,
    answer: str,
    sources: List[str],
    expected_faq: str,
    faq_kb: Dict[str, Dict],
    category: str,
    model: str = JUDGE_MODEL,
) -> Optional[Dict[str, Any]]:
    """
    Use an LLM judge to score a response on 3 dimensions.

    Returns dict with retrieval_precision, answer_groundedness, behavioral_guardrails
    (each with score and reasoning), or None on error.
    """
    # Look up expected FAQ content
    faq_entry = faq_kb.get(expected_faq, {})
    expected_faq_question = faq_entry.get("question", "N/A (no matching FAQ)")
    expected_faq_answer = faq_entry.get("answer", "N/A (no matching FAQ)")

    if expected_faq == "none":
        expected_faq_question = "N/A — this is an off-topic or ambiguous query with no expected FAQ"
        expected_faq_answer = "N/A — the assistant should decline or ask for clarification"

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        sources=", ".join(sources) if sources else "None",
        expected_faq_id=expected_faq,
        expected_faq_question=expected_faq_question,
        expected_faq_answer=expected_faq_answer,
        answer=answer[:1500],  # Truncate very long answers
        category=category,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3].strip()

        scores = json.loads(content)

        # Validate structure
        for dim in ("retrieval_precision", "answer_groundedness", "behavioral_guardrails"):
            if dim not in scores or "score" not in scores[dim]:
                return None
            scores[dim]["score"] = int(scores[dim]["score"])

        return scores

    except Exception as e:
        print(f"       [Judge Error] {e}")
        return None


def extract_faq_id_from_sources(sources: list) -> str:
    """Extract FAQ ID from the sources returned by the API."""
    if not sources:
        return "none"

    # Sources format: "Category (faq_id)" - extract the faq_id
    for source in sources:
        if "(" in source and ")" in source:
            # Extract content between parentheses
            start = source.rfind("(") + 1
            end = source.rfind(")")
            faq_id = source[start:end].lower().strip()
            # Normalize: FAQ-1 -> faq_1, faq_1 -> faq_1
            faq_id = faq_id.replace("-", "_").replace("faq", "faq_").replace("faq__", "faq_")
            if faq_id.startswith("faq_"):
                return faq_id

    return "none"

def run_single_test(question: str, session_id: str = None) -> dict:
    """Run a single test question against the API."""
    payload = {
        "question": question,
        "conversation_history": [],
        "session_id": session_id
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"answer": "TIMEOUT", "sources": [], "error": "timeout"}
    except Exception as e:
        return {"answer": "ERROR", "sources": [], "error": str(e)}

def run_evaluation(
    test_set: list,
    delay: float = 1.0,
    judge_enabled: bool = False,
    judge_model: str = JUDGE_MODEL,
    faq_kb: Optional[Dict[str, Dict]] = None,
) -> list:
    """Run all test questions and collect results, optionally scoring with LLM judge."""
    results = []
    total = len(test_set)

    print(f"\nRunning evaluation with {total} questions...")
    print("=" * 60)

    for i, test in enumerate(test_set, 1):
        question = test["question"]
        expected = test["expected_faq"]

        print(f"[{i}/{total}] Testing: {question[:50]}...")

        # Call the API
        response = run_single_test(question, session_id=f"eval_{i}")

        # Extract actual FAQ retrieved
        actual_faq = extract_faq_id_from_sources(response.get("sources", []))

        # Check if retrieval was correct
        # For "none" expected, we want no sources or general contact info
        if expected == "none":
            is_correct = actual_faq == "none" or len(response.get("sources", [])) == 0
        else:
            is_correct = actual_faq == expected

        # Check answer quality
        answer = response.get("answer", "")
        quality = check_answer_quality(answer, test)

        # For hallucination detection tests, also check if answer hallucinated
        hallucination_pass = True
        if test.get("category") == "hallucination_detection" or test.get("forbidden_patterns"):
            hallucination_pass = not quality["hallucination_check"]["has_hallucination"]

        result = {
            "id": test["id"],
            "question": question,
            "expected_faq": expected,
            "actual_faq": actual_faq,
            "sources": response.get("sources", []),
            "answer": answer,
            "category": test["category"],
            "topic": test["topic"],
            "is_correct": is_correct,
            "hallucination_pass": hallucination_pass,
            "quality": quality,
            "error": response.get("error")
        }

        results.append(result)

        # Status display
        retrieval_status = "PASS" if is_correct else "FAIL"
        if test.get("category") == "hallucination_detection":
            halluc_status = "PASS" if hallucination_pass else "HALLUC"
            print(f"       Expected: {expected}, Got: {actual_faq} [{retrieval_status}] Hallucination: [{halluc_status}]")
        else:
            print(f"       Expected: {expected}, Got: {actual_faq} [{retrieval_status}]")

        # Rate limiting
        time.sleep(delay)

    # =========================================================================
    # LLM Judge Scoring Pass (opt-in)
    # =========================================================================
    if judge_enabled and faq_kb is not None:
        print("\n" + "=" * 60)
        print(f"Running LLM judge scoring ({judge_model})...")
        print("=" * 60)

        client = OpenAI()
        for i, result in enumerate(results, 1):
            print(f"[{i}/{total}] Judging: {result['question'][:50]}...")
            scores = judge_response(
                client=client,
                question=result["question"],
                answer=result["answer"],
                sources=result["sources"],
                expected_faq=result["expected_faq"],
                faq_kb=faq_kb,
                category=result["category"],
                model=judge_model,
            )
            result["judge_scores"] = scores

            if scores:
                rp = scores["retrieval_precision"]["score"]
                ag = scores["answer_groundedness"]["score"]
                bg = scores["behavioral_guardrails"]["score"]
                print(f"       Retrieval: {rp}/5  Groundedness: {ag}/5  Guardrails: {bg}/5")
            else:
                print("       [Scores unavailable]")

            time.sleep(delay)

    return results

def calculate_metrics(results: list) -> dict:
    """Calculate evaluation metrics including quality checks."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])

    # Hallucination metrics
    halluc_tests = [r for r in results if r.get("category") == "hallucination_detection" or r.get("quality", {}).get("hallucination_check", {}).get("patterns_checked", 0) > 0]
    halluc_passed = sum(1 for r in halluc_tests if r.get("hallucination_pass", True))

    # By category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "halluc_passed": 0, "halluc_tested": 0}
        categories[cat]["total"] += 1
        if r["is_correct"]:
            categories[cat]["correct"] += 1
        # Track hallucination for halluc_detection category
        if cat == "hallucination_detection":
            categories[cat]["halluc_tested"] += 1
            if r.get("hallucination_pass", True):
                categories[cat]["halluc_passed"] += 1

    # Calculate percentages
    for cat in categories:
        categories[cat]["accuracy"] = round(
            categories[cat]["correct"] / categories[cat]["total"] * 100, 1
        )
        if categories[cat]["halluc_tested"] > 0:
            categories[cat]["halluc_accuracy"] = round(
                categories[cat]["halluc_passed"] / categories[cat]["halluc_tested"] * 100, 1
            )

    # Answer quality metrics
    quality_metrics = {
        "empty_answers": sum(1 for r in results if r.get("quality", {}).get("is_empty", False)),
        "avg_answer_length": round(
            sum(r.get("quality", {}).get("answer_length", 0) for r in results) / max(total, 1), 1
        ),
    }

    # LLM Judge score metrics
    judge_metrics = None
    judged_results = [r for r in results if r.get("judge_scores")]
    if judged_results:
        dimensions = ("retrieval_precision", "answer_groundedness", "behavioral_guardrails")

        # Overall averages
        avg_scores = {}
        for dim in dimensions:
            scores = [r["judge_scores"][dim]["score"] for r in judged_results]
            avg_scores[dim] = round(sum(scores) / len(scores), 2)

        # Per-category averages
        cat_judge = {}
        for r in judged_results:
            cat = r["category"]
            if cat not in cat_judge:
                cat_judge[cat] = {dim: [] for dim in dimensions}
            for dim in dimensions:
                cat_judge[cat][dim].append(r["judge_scores"][dim]["score"])

        cat_averages = {}
        for cat, dim_scores in cat_judge.items():
            cat_averages[cat] = {
                dim: round(sum(scores) / len(scores), 2)
                for dim, scores in dim_scores.items()
            }

        # Overall pass rate: all 3 dimensions >= 3
        overall_pass = sum(
            1 for r in judged_results
            if all(r["judge_scores"][dim]["score"] >= 3 for dim in dimensions)
        )

        judge_metrics = {
            "total_judged": len(judged_results),
            "avg_scores": avg_scores,
            "by_category": cat_averages,
            "overall_pass": overall_pass,
            "overall_pass_rate": round(overall_pass / len(judged_results) * 100, 1),
        }

    return {
        "overall_accuracy": round(correct / total * 100, 1),
        "total_questions": total,
        "correct": correct,
        "incorrect": total - correct,
        "hallucination": {
            "total_tested": len(halluc_tests),
            "passed": halluc_passed,
            "accuracy": round(halluc_passed / max(len(halluc_tests), 1) * 100, 1) if halluc_tests else None,
        },
        "quality": quality_metrics,
        "by_category": categories,
        "judge_scores": judge_metrics,
    }

def get_failures(results: list) -> list:
    """Get list of failed test cases."""
    return [r for r in results if not r["is_correct"]]

def get_hallucination_failures(results: list) -> list:
    """Get list of failed hallucination tests."""
    return [r for r in results if r.get("category") == "hallucination_detection" and not r.get("hallucination_pass", True)]


def generate_report(results: list, metrics: dict, output_path: str):
    """Generate a markdown evaluation report with optional LLM judge scores."""
    failures = get_failures(results)
    halluc_failures = get_hallucination_failures(results)

    # Determine hallucination status
    halluc_metrics = metrics.get("hallucination", {})
    halluc_status = "N/A"
    if halluc_metrics.get("total_tested", 0) > 0:
        halluc_status = "PASS" if halluc_metrics.get("accuracy", 0) >= 90 else "NEEDS IMPROVEMENT"

    report = f"""# Escrow Assistant Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**API Endpoint:** {API_URL}

---

## Summary

| Metric | Value |
|--------|-------|
| **Overall Retrieval Accuracy** | {metrics['overall_accuracy']}% |
| **Total Questions** | {metrics['total_questions']} |
| **Correct** | {metrics['correct']} |
| **Incorrect** | {metrics['incorrect']} |
| **Target** | ≥90% |
| **Retrieval Status** | {'PASS' if metrics['overall_accuracy'] >= 90 else 'NEEDS IMPROVEMENT'} |

### Hallucination Detection

| Metric | Value |
|--------|-------|
| **Tests with Hallucination Checks** | {halluc_metrics.get('total_tested', 0)} |
| **Passed (No Hallucination)** | {halluc_metrics.get('passed', 0)} |
| **Hallucination Accuracy** | {halluc_metrics.get('accuracy', 'N/A')}% |
| **Status** | {halluc_status} |

### Answer Quality

| Metric | Value |
|--------|-------|
| **Empty Answers** | {metrics.get('quality', {}).get('empty_answers', 0)} |
| **Avg Answer Length** | {metrics.get('quality', {}).get('avg_answer_length', 0)} chars |

---

## Accuracy by Category

| Category | Total | Correct | Accuracy | Halluc Pass |
|----------|-------|---------|----------|-------------|
"""

    for cat, data in metrics['by_category'].items():
        halluc_col = f"{data.get('halluc_passed', '-')}/{data.get('halluc_tested', '-')}" if data.get('halluc_tested', 0) > 0 else "-"
        report += f"| {cat} | {data['total']} | {data['correct']} | {data['accuracy']}% | {halluc_col} |\n"

    report += f"""
---

## Failure Analysis

**Total Failures:** {len(failures)}

"""

    if failures:
        report += "| ID | Question | Expected | Actual | Category |\n"
        report += "|-----|----------|----------|--------|----------|\n"

        for f in failures:
            q = f['question'][:40] + "..." if len(f['question']) > 40 else f['question']
            report += f"| {f['id']} | {q} | {f['expected_faq']} | {f['actual_faq']} | {f['category']} |\n"

        report += "\n### Failure Details\n\n"

        for f in failures:
            report += f"""
#### Test #{f['id']}: {f['topic']}

- **Question:** {f['question']}
- **Expected FAQ:** {f['expected_faq']}
- **Actual FAQ:** {f['actual_faq']}
- **Sources:** {', '.join(f['sources']) if f['sources'] else 'None'}
- **Answer Preview:** {f['answer'][:200]}...

"""
    else:
        report += "*No retrieval failures - all tests passed!*\n"

    # Hallucination failures section
    report += f"""
---

## Hallucination Analysis

**Total Hallucination Failures:** {len(halluc_failures)}

"""

    if halluc_failures:
        report += "| ID | Question | Hallucinated Content |\n"
        report += "|-----|----------|----------------------|\n"

        for f in halluc_failures:
            q = f['question'][:40] + "..." if len(f['question']) > 40 else f['question']
            matches = f.get('quality', {}).get('hallucination_check', {}).get('matches', [])
            halluc_content = ", ".join(str(m) for m in matches[:3]) if matches else "Pattern matched"
            report += f"| {f['id']} | {q} | {halluc_content} |\n"

        report += "\n### Hallucination Details\n\n"

        for f in halluc_failures:
            matches = f.get('quality', {}).get('hallucination_check', {}).get('matches', [])
            report += f"""
#### Test #{f['id']}: {f.get('topic', 'unknown')}

- **Question:** {f['question']}
- **Expected Behavior:** {f.get('expected_behavior', 'Should not hallucinate specific details')}
- **Hallucinated Content:** {', '.join(str(m) for m in matches) if matches else 'Pattern matched'}
- **Answer Preview:** {f['answer'][:300]}...

"""
    else:
        report += "*No hallucinations detected in tested responses.*\n"

    # =========================================================================
    # LLM Judge Scores section (only if judge was used)
    # =========================================================================
    judge_metrics = metrics.get("judge_scores")
    if judge_metrics:
        avg = judge_metrics["avg_scores"]
        report += f"""
---

## LLM Judge Scores

**Model:** {JUDGE_MODEL} | **Responses Judged:** {judge_metrics['total_judged']} | **Overall Pass Rate (all dims >= 3):** {judge_metrics['overall_pass_rate']}% ({judge_metrics['overall_pass']}/{judge_metrics['total_judged']})

### Summary

| Dimension | Avg Score (1-5) |
|-----------|----------------|
| Retrieval Precision | {avg['retrieval_precision']} |
| Answer Groundedness | {avg['answer_groundedness']} |
| Behavioral Guardrails | {avg['behavioral_guardrails']} |

### Scores by Category

| Category | Retrieval Precision | Answer Groundedness | Behavioral Guardrails |
|----------|--------------------|--------------------|----------------------|
"""
        for cat, scores in judge_metrics["by_category"].items():
            report += f"| {cat} | {scores['retrieval_precision']} | {scores['answer_groundedness']} | {scores['behavioral_guardrails']} |\n"

        # Low-scoring responses detail (any dimension < 3)
        low_scoring = [
            r for r in results
            if r.get("judge_scores") and any(
                r["judge_scores"][dim]["score"] < 3
                for dim in ("retrieval_precision", "answer_groundedness", "behavioral_guardrails")
            )
        ]

        report += f"\n### Low-Scoring Responses (any dimension < 3)\n\n"
        report += f"**Total:** {len(low_scoring)}\n\n"

        if low_scoring:
            report += "| ID | Question | Ret. | Grnd. | Guard. | Lowest Reasoning |\n"
            report += "|----|----------|------|-------|--------|------------------|\n"

            for r in low_scoring:
                s = r["judge_scores"]
                q = r['question'][:35] + "..." if len(r['question']) > 35 else r['question']
                rp = s["retrieval_precision"]["score"]
                ag = s["answer_groundedness"]["score"]
                bg = s["behavioral_guardrails"]["score"]

                # Find lowest dimension and its reasoning
                dims = {"retrieval_precision": rp, "answer_groundedness": ag, "behavioral_guardrails": bg}
                lowest_dim = min(dims, key=dims.get)
                reasoning = s[lowest_dim]["reasoning"][:80] + "..." if len(s[lowest_dim]["reasoning"]) > 80 else s[lowest_dim]["reasoning"]
                # Escape pipe chars for markdown table
                reasoning = reasoning.replace("|", "\\|")

                report += f"| {r['id']} | {q} | {rp} | {ag} | {bg} | {reasoning} |\n"
        else:
            report += "*All responses scored 3 or above on every dimension.*\n"

    report += """
---

## Recommendations

"""

    # Generate recommendations based on failures
    if metrics['overall_accuracy'] >= 90:
        report += "1. Retrieval accuracy meets target. Consider expanding test set for edge cases.\n"
    else:
        report += "1. Retrieval accuracy below target. Review failed cases and improve FAQ coverage.\n"

    # Check category-specific issues
    for cat, data in metrics['by_category'].items():
        if data['accuracy'] < 80:
            report += f"2. **{cat}** category needs attention ({data['accuracy']}% accuracy)\n"

    report += """
---

## Groundedness Spot-Check

*Manual review of 10 random responses for hallucinated content:*

| # | Question | Hallucination? | Notes |
|---|----------|----------------|-------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

---

## Questions for ServisBot (Feb 6 Meeting)

Based on evaluation results, ask the vendor:

1. How does your platform handle [specific failure case]?
2. What retrieval accuracy do you typically achieve?
3. How do you handle ambiguous/vague queries?
4. What's your approach to adversarial inputs?
5. Can you show metrics dashboards for monitoring production accuracy?

"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Escrow Assistant Evaluation Script")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge scoring (requires OPENAI_API_KEY, costs ~$0.01)",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help=f"Model for LLM judge (default: {JUDGE_MODEL})",
    )
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    test_set_path = script_dir / "eval_test_set.json"
    results_path = script_dir / "eval_results.json"
    report_path = script_dir / "eval_report.md"
    faq_kb_path = script_dir.parent / "backend" / "data" / "escrow_faqs.json"

    # Load test set
    print("Loading test set...")
    test_set = load_test_set(test_set_path)
    print(f"Loaded {len(test_set)} test questions")

    # Load FAQ knowledge base (needed for judge)
    faq_kb = None
    if args.judge:
        print(f"Loading FAQ knowledge base for LLM judge ({args.judge_model})...")
        faq_kb = load_faq_knowledge_base(faq_kb_path)
        print(f"Loaded {len(faq_kb)} FAQs")

    # Run evaluation
    results = run_evaluation(
        test_set,
        delay=1.5,
        judge_enabled=args.judge,
        judge_model=args.judge_model,
        faq_kb=faq_kb,
    )

    # Save raw results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {results_path}")

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Retrieval Accuracy: {metrics['overall_accuracy']}%")
    print(f"Correct: {metrics['correct']}/{metrics['total_questions']}")

    # Hallucination summary
    halluc = metrics.get('hallucination', {})
    if halluc.get('total_tested', 0) > 0:
        print(f"\nHallucination Detection: {halluc['accuracy']}% ({halluc['passed']}/{halluc['total_tested']} passed)")

    # Judge scores summary
    judge = metrics.get('judge_scores')
    if judge:
        avg = judge['avg_scores']
        print(f"\nLLM Judge Scores (avg, 1-5):")
        print(f"  Retrieval Precision:   {avg['retrieval_precision']}")
        print(f"  Answer Groundedness:   {avg['answer_groundedness']}")
        print(f"  Behavioral Guardrails: {avg['behavioral_guardrails']}")
        print(f"  Overall Pass Rate:     {judge['overall_pass_rate']}% ({judge['overall_pass']}/{judge['total_judged']})")

    print("\nBy Category:")
    for cat, data in metrics['by_category'].items():
        halluc_info = ""
        if data.get('halluc_tested', 0) > 0:
            halluc_info = f" | Halluc: {data['halluc_passed']}/{data['halluc_tested']}"
        print(f"  {cat}: {data['accuracy']}% ({data['correct']}/{data['total']}){halluc_info}")

    # Generate report
    generate_report(results, metrics, report_path)

    print("\n" + "=" * 60)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
