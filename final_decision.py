# final_decision.py
from typing import Dict, Any, List

def classify_claim_and_reference(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine claim verdict and reference verdicts.

    Possible outputs:
    - CLAIM_ONLY_SUPPORTED
    - CLAIM_ONLY_REFUTED
    - CLAIM_ONLY_NEI
    - CLAIM_SUPPORTED_REF_SUPPORTED
    - CLAIM_SUPPORTED_REF_PARTIAL
    - CLAIM_SUPPORTED_REF_FABRICATED
    - CLAIM_REFUTED_REF_SUPPORTED
    - CLAIM_REFUTED_REF_PARTIAL
    - CLAIM_REFUTED_REF_FABRICATED
    """
    claim_verdict = claim.get("verdict", "NEI")
    refs: List[Dict[str, Any]] = claim.get("references", [])

    if not refs:
        if claim_verdict == "SUPPORTED":
            final_label = "CLAIM_ONLY_SUPPORTED"
        elif claim_verdict == "REFUTED":
            final_label = "CLAIM_ONLY_REFUTED"
        else:
            final_label = "CLAIM_ONLY_NEI"

        claim["final_label"] = final_label
        return claim

    ref_verdicts = [r.get("verdict") for r in refs]

    if any(v == "FABRICATED" for v in ref_verdicts):
        ref_summary = "FABRICATED"
    elif any(v == "PARTIAL" for v in ref_verdicts):
        ref_summary = "PARTIAL"
    else:
        ref_summary = "SUPPORTED"

    claim["reference_summary"] = ref_summary
    claim["final_label"] = f"CLAIM_{claim_verdict}_REF_{ref_summary}"
    return claim

def classify_all(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [classify_claim_and_reference(c) for c in claims]