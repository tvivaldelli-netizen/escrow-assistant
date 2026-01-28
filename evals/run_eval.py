"""
Escrow Assistant Evaluation Script
Runs test questions against the chatbot API and measures retrieval accuracy.
"""

import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Configuration
API_URL = "https://escrow-assistant.onrender.com/ask"  # Production
# API_URL = "http://localhost:8000/ask"  # Local testing

def load_test_set(path: str) -> list:
    """Load the evaluation test set."""
    with open(path, 'r') as f:
        return json.load(f)

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

        result = {
            "id": test["id"],
            "question": question,
            "expected_faq": expected,
            "actual_faq": actual_faq,
            "sources": response.get("sources", []),
            "answer": response.get("answer", ""),
            "category": test["category"],
            "topic": test["topic"],
            "is_correct": is_correct,
            "error": response.get("error")
        }

        results.append(result)

        status = "PASS" if is_correct else "FAIL"
        print(f"       Expected: {expected}, Got: {actual_faq} [{status}]")

        # Rate limiting
        time.sleep(delay)

    return results

def calculate_metrics(results: list) -> dict:
    """Calculate evaluation metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])

    # By category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["is_correct"]:
            categories[cat]["correct"] += 1

    # Calculate percentages
    for cat in categories:
        categories[cat]["accuracy"] = round(
            categories[cat]["correct"] / categories[cat]["total"] * 100, 1
        )

    return {
        "overall_accuracy": round(correct / total * 100, 1),
        "total_questions": total,
        "correct": correct,
        "incorrect": total - correct,
        "by_category": categories
    }

def get_failures(results: list) -> list:
    """Get list of failed test cases."""
    return [r for r in results if not r["is_correct"]]

def generate_report(results: list, metrics: dict, output_path: str):
    """Generate a markdown evaluation report."""
    failures = get_failures(results)

    report = f"""# Escrow Assistant Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**API Endpoint:** {API_URL}

---

## Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | {metrics['overall_accuracy']}% |
| **Total Questions** | {metrics['total_questions']} |
| **Correct** | {metrics['correct']} |
| **Incorrect** | {metrics['incorrect']} |
| **Target** | ≥90% |
| **Status** | {'PASS' if metrics['overall_accuracy'] >= 90 else 'NEEDS IMPROVEMENT'} |

---

## Accuracy by Category

| Category | Total | Correct | Accuracy |
|----------|-------|---------|----------|
"""

    for cat, data in metrics['by_category'].items():
        report += f"| {cat} | {data['total']} | {data['correct']} | {data['accuracy']}% |\n"

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
        report += "*No failures - all tests passed!*\n"

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
    print(f"Overall Accuracy: {metrics['overall_accuracy']}%")
    print(f"Correct: {metrics['correct']}/{metrics['total_questions']}")
    print("\nBy Category:")
    for cat, data in metrics['by_category'].items():
        print(f"  {cat}: {data['accuracy']}% ({data['correct']}/{data['total']})")

    # Generate report
    generate_report(results, metrics, report_path)

    print("\n" + "=" * 60)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
