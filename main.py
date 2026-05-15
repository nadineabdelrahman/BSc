import json
import os
from typing import Any, Iterable

from hybrid_extraction import extract_triples_hybrid
from verification import verify_triples
from logger import log_results
from generation import generate_answer
from correction import (
    apply_correction_loop,
    answer_has_unverified_quoted_title_claim,
    safe_no_claims_correction,
)

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



def has_supported_claims(claims: list[dict]) -> bool:
    return any(
        c.get("final_label") == "TRUE" or c.get("verdict") == "SUPPORTED"
        for c in claims
    )


def count_bad_claims(claims: list[dict]) -> int:
    return sum(
        1 for c in claims
        if c.get("verdict") in {"REFUTED", "NEI"}
        or c.get("final_label") in {"FALSE", "HALLUCINATION", "UNVERIFIABLE"}
    )


def should_accept_correction(before: list[dict], after: list[dict], corrected_answer: str = "") -> bool:
    before_bad = count_bad_claims(before)
    after_bad = count_bad_claims(after)

    before_has_supported = has_supported_claims(before)
    after_has_supported = has_supported_claims(after)

    if before_bad == 0:
        return False

    safe_fallback_markers = [
        "could not verify",
        "cannot confidently provide",
        "could not be verified",
        "cannot provide a factual answer",
        "could not confirm",
    ]

    corrected_l = str(corrected_answer or "").lower()
    is_safe_fallback = any(m in corrected_l for m in safe_fallback_markers)

    # If original answer had supported facts, do NOT accept a fallback that erases them.
    if before_has_supported and not after_has_supported and is_safe_fallback:
        return False

    # Normal case: correction is accepted if bad claims decrease.
    if after_bad < before_bad:
        return True

    # Fallback is acceptable only if there were no supported facts to preserve.
    if not before_has_supported and not after and is_safe_fallback:
        return True

    return False

def get_answer_label(verified: list[dict]) -> str:
    if not verified:
        return "NO_CLAIMS"

    labels = {c.get("final_label") for c in verified}

    if "HALLUCINATION" in labels or "FALSE" in labels:
        return "HALLUCINATED"
    elif labels == {"TRUE"}:
        return "CLEAN"
    else:
        return "UNVERIFIABLE"


def run_batch(questions: Iterable[str]) -> None:
    questions = list(questions)
    total = len(questions)

    for i, question in enumerate(questions, start=1):
        print(f"[{i}/{total}] Processing: {question}")

        try:
            answer = generate_answer(question)
            triples = extract_triples_hybrid(answer, prompt_text=question)
            verified = verify_triples(triples)

            # Keep original output for before/after logging
            original_answer = answer
            original_verified = verified

            # Correction-loop logging fields
            corrected_answer = None
            verified_corrected = []
            correction_attempted = False
            correction_accepted = False
            bad_claims_before = None
            bad_claims_after = None
            correction_reason = None

            # --------------------------------------------------
            # Case 1: No claims extracted, but answer looks factual
            # about a quoted title/entity.
            # Example: "The film 'Crystal Reef of Mars' was directed by X."
            # --------------------------------------------------
            if not verified and answer_has_unverified_quoted_title_claim(answer):
                correction_attempted = True
                corrected_answer = safe_no_claims_correction(question, answer)

                print("  Corrected answer:")
                print(corrected_answer)

                triples_corrected = extract_triples_hybrid(
                    corrected_answer,
                    prompt_text=question
                )
                verified_corrected = verify_triples(triples_corrected)

                bad_claims_before = 1
                bad_claims_after = count_bad_claims(verified_corrected)

                answer = corrected_answer
                verified = verified_corrected
                correction_accepted = True
                correction_reason = (
                    "Accepted because extraction produced no claims, but the original answer contained an unsupported factual claim about a quoted title/entity."
                )
                print("  Correction accepted for NO_CLAIMS factual answer.")

            # --------------------------------------------------
            # Case 2: Normal System-2 correction for detected bad claims
            # --------------------------------------------------
            elif verified:
                needs_correction = any(
                    c.get("final_label") in {"FALSE", "HALLUCINATION"}
                    for c in verified
                )

                if needs_correction:
                    correction_attempted = True

                    corrected_answer = apply_correction_loop(
                        question,
                        answer,
                        verified
                    )

                    print("  Corrected answer:")
                    print(corrected_answer)

                    # Re-run extraction + verification on corrected answer
                    triples_corrected = extract_triples_hybrid(
                        corrected_answer,
                        prompt_text=question
                    )
                    verified_corrected = verify_triples(triples_corrected)

                    bad_claims_before = count_bad_claims(verified)
                    bad_claims_after = count_bad_claims(verified_corrected)

                    print(f"  Bad claims before correction: {bad_claims_before}")
                    print(f"  Bad claims after correction: {bad_claims_after}")

                    if should_accept_correction(verified, verified_corrected, corrected_answer):
                        answer = corrected_answer
                        verified = verified_corrected
                        correction_accepted = True
                        correction_reason = (
                            "Accepted because corrected answer reduced unsupported/false claims or safely removed unverifiable claims."
                        )
                        print("  Correction accepted.")
                    else:
                        correction_accepted = False
                        correction_reason = (
                            "Rejected because corrected answer did not improve verification."
                        )
                        print("  Correction rejected because it did not improve verification.")

            answer_label = get_answer_label(verified)

            # If correction was accepted as a safe NO_CLAIMS fallback, label it as CLEAN
            # because the final answer no longer asserts the hallucinated fact.
            if correction_accepted and not verified and corrected_answer:
                answer_label = "CLEAN"

            log_results(
                question,
                answer,
                verified,
                reference_only_label=answer_label,

                # Before/after correction-loop fields
                original_answer=original_answer,
                corrected_answer=corrected_answer,
                original_claims=original_verified,
                corrected_claims=verified_corrected,
                correction_attempted=correction_attempted,
                correction_accepted=correction_accepted,
                bad_claims_before=bad_claims_before,
                bad_claims_after=bad_claims_after,
                correction_reason=correction_reason,
            )

            print(
                f"  Done. Extracted {len(triples)} triples, "
                f"verified {len(verified)} triples."
            )

        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    if not os.path.exists(PROMPTS_FILE):
        raise FileNotFoundError(f"Could not find {PROMPTS_FILE}")

    questions = load_questions(PROMPTS_FILE)

    if not questions:
        raise ValueError("No questions found in data/prompts.json")

    run_batch(questions)