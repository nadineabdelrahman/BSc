import json
import os
from datetime import datetime
from typing import Optional, Any

RESULTS_FILE = "results_7.json"


def log_results(
    prompt: str,
    answer: str,
    claims: list,
    verified_references: list = None,
    reference_only_label: str = None,

    # Correction-loop logging fields
    original_answer: Optional[str] = None,
    corrected_answer: Optional[str] = None,
    original_claims: Optional[list] = None,
    corrected_claims: Optional[list] = None,
    correction_attempted: bool = False,
    correction_accepted: bool = False,
    bad_claims_before: Optional[int] = None,
    bad_claims_after: Optional[int] = None,
    correction_reason: Optional[str] = None,
):
    record: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,

        # Final answer after possible correction
        "answer": answer,

        # Before/after correction-loop answers
        "original_answer": original_answer,
        "corrected_answer": corrected_answer,

        # Final claims after possible correction
        "claims": claims if claims else [],

        # Before/after claim sets
        "original_claims": original_claims if original_claims else [],
        "corrected_claims": corrected_claims if corrected_claims else [],

        # Correction-loop metadata
        "correction": {
            "attempted": correction_attempted,
            "accepted": correction_accepted,
            "bad_claims_before": bad_claims_before,
            "bad_claims_after": bad_claims_after,
            "reason": correction_reason,
        },

        # References
        "verified_references": verified_references if verified_references else [],

        # Summary label
        "reference_only_label": reference_only_label,
    }

    # Create file if missing
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Read safely
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            data = json.loads(content) if content else []
            if not isinstance(data, list):
                data = []
    except (json.JSONDecodeError, OSError):
        data = []

    data.append(record)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)