"""
Escrow Assistant Evaluation Script
Runs test questions against the chatbot API and measures retrieval accuracy.

Enhanced metrics:
- Retrieval accuracy (FAQ matching)
- Answer quality (keyword presence, hallucination detection)
- Category-level breakdown
"""

import json
import re
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configuration
API_URL = "https://escrow-assistant.onrender.com/ask"  # Production
# API_URL = "http://localhost:8001/ask"  # Local testing

def load_test_set(path: str) -> list:
    """Load the evaluation test set."""
    with open(path, 'r') as f:
        return json.load(f)

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

def run_evaluation(test_set: list, delay: float = 1.0) -> list:
    """Run all test questions and collect results."""
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
    }

def get_failures(results: list) -> list:
    """Get list of failed test cases."""
    return [r for r in results if not r["is_correct"]]

def get_hallucination_failures(results: list) -> list:
    """Get list of failed hallucination tests."""
    return [r for r in results if r.get("category") == "hallucination_detection" and not r.get("hallucination_pass", True)]


def generate_report(results: list, metrics: dict, output_path: str):
    """Generate a markdown evaluation report."""
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
    # Paths
    script_dir = Path(__file__).parent
    test_set_path = script_dir / "eval_test_set.json"
    results_path = script_dir / "eval_results.json"
    report_path = script_dir / "eval_report.md"

    # Load test set
    print("Loading test set...")
    test_set = load_test_set(test_set_path)
    print(f"Loaded {len(test_set)} test questions")

    # Run evaluation
    results = run_evaluation(test_set, delay=1.5)

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
