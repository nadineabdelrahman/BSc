import json
import os
from typing import Any, Iterable

from hybrid_extraction import extract_triples_hybrid
from verification import verify_triples
from logger import log_results
from generation import generate_answer


PROMPTS_FILE = "data/prompts.json"


def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected data/prompts.json to contain a list of objects.")

    questions = []
    for item in data:
        if isinstance(item, dict):
            q = item.get("question")
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())

    return questions


def run_batch(questions: Iterable[str]) -> None:
    questions = list(questions)
    total = len(questions)

    for i, question in enumerate(questions, start=1):
        print(f"[{i}/{total}] Processing: {question}")

        try:
            answer = generate_answer(question)
            triples = extract_triples_hybrid(answer, prompt_text=question)
            verified = verify_triples(triples)

            log_results(question, answer, verified)

            print(f"  Done. Extracted {len(triples)} triples, verified {len(verified)} triples.")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    if not os.path.exists(PROMPTS_FILE):
        raise FileNotFoundError(f"Could not find {PROMPTS_FILE}")

    questions = load_questions(PROMPTS_FILE)

    if not questions:
        raise ValueError("No questions found in data/prompts.json")

    run_batch(questions)
