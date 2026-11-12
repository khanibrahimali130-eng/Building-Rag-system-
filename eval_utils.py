"""Small helpers used by both run_tests.py and experiments.py."""
import csv
from pathlib import Path

TEST_FILE = Path(__file__).parent / "tests" / "test_questions.csv"


def load_tests(path=TEST_FILE):
    """Read the test questions from the CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_sources(text):
    """Turn 'a.md; b.md' into ['a.md', 'b.md']."""
    return [s.strip() for s in text.split(";") if s.strip()]


def retrieved_files(chunks):
    """List the file names of the chunks, each name once, in order."""
    seen = []
    for chunk in chunks:
        name = chunk.metadata["filename"]
        if name not in seen:
            seen.append(name)
    return seen
