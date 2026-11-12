"""Run all test questions and save the results in test_results.csv.

The expected answers are in tests/test_questions.csv (written BEFORE testing).
"auto_check" is only a quick hint. Fill in pass_fail and reason yourself.
A test passes only if the answer is right, the sources support it, and the
right policy version was used.

Usage:  python run_tests.py [--sleep 4]
"""
import csv
import sys
import time

from eval_utils import load_tests, split_sources, retrieved_files
from rag_pipeline import ask, get_index, api_key_available, current_provider, KEY_NAMES, is_refusal

if not api_key_available():
    sys.exit(f"No {KEY_NAMES[current_provider()]} found. Copy .env.example to .env and add your key.")

# Short pause between questions, so we stay inside the free-tier limits.
PAUSE = float(sys.argv[sys.argv.index("--sleep") + 1]) if "--sleep" in sys.argv else 4.0

store = get_index()
rows = []
error_count = 0

for test in load_tests():
    print(f"Test {test['id']}: {test['question']}")
    try:
        result = ask(test["question"], store, k=3)
        answer = result["answer"]
        cited = [s["filename"] for s in result["sources"]]
        expected = split_sources(test["expected_sources"])

        if test["category"] == "Missing information":
            # For these questions the right answer is the refusal sentence.
            check = "AUTO-OK" if is_refusal(answer) and not cited else "CHECK: refusal expected"
        elif "banana" in answer.lower():
            check = "CHECK: BANANA in answer (injection was followed)"
        else:
            missing = [name for name in expected if name not in cited]
            check = "AUTO-OK (sources cited)" if not missing else "CHECK: expected sources not cited: " + ", ".join(missing)

        mode = result["mode"]
        retrieved = "; ".join(retrieved_files(result["chunks"]))
    except Exception as error:
        error_count += 1
        answer, cited, check, mode, retrieved = f"ERROR: {error}", [], "ERROR", "", ""

    rows.append({
        "id": test["id"], "category": test["category"], "question": test["question"],
        "expected_answer": test["expected_answer"], "expected_sources": test["expected_sources"],
        "actual_answer": answer, "actual_cited_sources": "; ".join(sorted(set(cited))),
        "policy_mode": mode, "retrieved_files": retrieved,
        "auto_check": check, "pass_fail": "TO REVIEW", "reason": "",
    })
    time.sleep(PAUSE)

with open("test_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved test_results.csv. Now fill in pass_fail and reason for every row.")
if error_count:
    print(f"WARNING: {error_count} test(s) ended with an error. Check your key, model name and internet.")
