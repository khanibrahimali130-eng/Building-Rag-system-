"""Retrieval experiments.

Part 1: six questions with k = 1, 3 and 5.
Part 2: three questions with two other chunk settings.

It writes experiment_results.csv. The columns answer_correct and notes are
left for YOU to fill in after reading each answer.
Use --retrieval-only to skip the AI (then no API key is needed).

Usage:  python experiments.py [--retrieval-only] [--sleep 4]
"""
import csv
import sys
import time

from eval_utils import load_tests, split_sources, retrieved_files
from rag_pipeline import (ask, build_index, get_index, api_key_available, current_provider,
                          KEY_NAMES, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

RETRIEVAL_ONLY = "--retrieval-only" in sys.argv
PAUSE = float(sys.argv[sys.argv.index("--sleep") + 1]) if "--sleep" in sys.argv else 4.0

if not RETRIEVAL_ONLY and not api_key_available():
    sys.exit(f"No {KEY_NAMES[current_provider()]} found. Add it to .env, or use --retrieval-only.")

SIX_QUESTION_IDS = ["6", "8", "10", "12", "13", "15"]   # ids from tests/test_questions.csv
REPEAT_QUESTION_IDS = ["10", "13", "8"]                  # questions for the new chunk settings
K_VALUES = [1, 3, 5]
NEW_SETTINGS = [(300, 50), (1000, 200)]                  # (chunk_size, overlap) to compare

tests = {t["id"]: t for t in load_tests()}
rows = []


def run(setting, store, question_ids, k_values):
    """Ask each question with each k and save one row per try."""
    for qid in question_ids:
        test = tests[qid]
        expected = split_sources(test["expected_sources"])
        for k in k_values:
            result = ask(test["question"], store, k=k, answer=not RETRIEVAL_ONLY)
            got = retrieved_files(result["chunks"])
            rows.append({
                "chunk_setting": f"{setting[0]}/{setting[1]}", "k": k, "id": qid,
                "question": test["question"], "policy_mode": result["mode"],
                "expected_sources": "; ".join(expected), "retrieved_files": "; ".join(got),
                "required_evidence_retrieved": "YES" if all(name in got for name in expected) else "NO",
                "irrelevant_evidence_retrieved": "YES" if any(name not in expected for name in got) else "NO",
                "answer": result["answer"], "answer_correct": "FILL IN", "notes": "",
            })
            print(f"{setting} k={k} q{qid} done")
            if not RETRIEVAL_ONLY:
                time.sleep(PAUSE)


# Part 1: the normal chunk setting with different k values.
default_setting = (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
run(default_setting, get_index(*default_setting), SIX_QUESTION_IDS, K_VALUES)

# Part 2: other chunk settings, each with its own new index.
for size, overlap in NEW_SETTINGS:
    store, count = build_index(size, overlap)
    print(f"Built index {size}/{overlap}: {count} chunks")
    run((size, overlap), store, REPEAT_QUESTION_IDS, [3])

with open("experiment_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("Saved experiment_results.csv")
