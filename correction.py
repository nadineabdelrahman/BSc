from typing import List, Dict, Any
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_evidence_text(claims: List[Dict[str, Any]]) -> str:
    evidence_lines = []

    for i, c in enumerate(claims, start=1):
        verdict = c.get("verdict")
        if verdict in {"REFUTED", "NEI"}:
            s = c.get("s")
            p = c.get("p")
            o = c.get("o")

            evidence_lines.append(f"Claim {i}: ({s}, {p}, {o})")
            evidence_lines.append(f"  Verdict: {verdict}")

            if c.get("final_label"):
                evidence_lines.append(f"  Final label: {c['final_label']}")

            if c.get("likely_hallucination") is True:
                evidence_lines.append("  Likely hallucination: true")

            if c.get("nei_type"):
                evidence_lines.append(f"  NEI type: {c['nei_type']}")

            if c.get("reason"):
                evidence_lines.append(f"  Reason: {c['reason']}")

            if c.get("explanation"):
                evidence_lines.append(f"  Explanation: {c['explanation']}")

            ge = c.get("graph_evidence", [])
            if ge:
                evidence_lines.append("  Graph evidence:")
                for e in ge:
                    evidence_lines.append(f"    - {e}")

            evidence_lines.append("")

    return "\n".join(evidence_lines).strip()


def _needs_safe_fallback(claims: List[Dict[str, Any]]) -> bool:
    """
    If we only have hallucination-style NEI/entity-linking-failure evidence
    and no positive correction evidence, prefer a cautious fallback answer.
    """
    bad_claims = [c for c in claims if c.get("verdict") in {"REFUTED", "NEI"}]
    if not bad_claims:
        return False

    all_hallucination_style = True
    has_positive_graph_fact = False

    for c in bad_claims:
        if c.get("likely_hallucination") is not True:
            all_hallucination_style = False

        for e in c.get("graph_evidence", []):
            if isinstance(e, dict) and e.get("type") != "entity_linking_failure":
                has_positive_graph_fact = True

    return all_hallucination_style and not has_positive_graph_fact


def _safe_fallback_answer(question: str, claims: List[Dict[str, Any]]) -> str:
    """
    Used when the system has evidence that the title/entity is likely fabricated
    but has no verified replacement fact to supply.
    """
    entities = []
    for c in claims:
        if c.get("verdict") in {"REFUTED", "NEI"}:
            if c.get("s"):
                entities.append(c["s"])

    entity_text = entities[0] if entities else "this entity"

    return (
        f'I could not verify "{entity_text}" in the knowledge graph, '
        f'so I cannot confidently provide a factual answer to the question: {question}'
    )


def apply_correction_loop(question: str, original_answer: str, claims: List[Dict[str, Any]]) -> str:
    """
    System-2 correction loop:
    Uses KG evidence to fix hallucinated claims.
    """

    if _needs_safe_fallback(claims):
        return _safe_fallback_answer(question, claims)

    evidence_text = build_evidence_text(claims)

    prompt = f"""
You are a strict fact-correction assistant.

Question:
{question}

Original answer:
{original_answer}

Verification findings:
{evidence_text}

Rules:
1. Use ONLY facts supported by the verification findings.
2. If a claim is marked NEI, hallucinated, or entity_linking_failure, do NOT restate it as true.
3. If the entity or title could not be verified, explicitly say that it could not be verified.
4. Do NOT preserve unsupported names, titles, or authors just because they appeared in the original answer.
5. If no verified correction is available, return a cautious answer stating that the fact could not be confirmed.
6. Keep the answer concise and natural.

Return ONLY the corrected answer.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()