import json
import os
from datetime import datetime

RESULTS_FILE = "results_2.json"


def log_results(
    prompt: str,
    answer: str,
    claims: list,
    verified_references: list = None,
    reference_only_label: str = None
):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "answer": answer,

        # Claims (triples after full pipeline)
        "claims": claims if claims else [],

        # References (after verification)
        "verified_references": verified_references if verified_references else [],

        # Summary label (important for your thesis)
        "reference_only_label": reference_only_label
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