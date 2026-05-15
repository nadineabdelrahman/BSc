from typing import List, Dict, Any, Tuple
from openai import OpenAI
import os
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SAFE_FALLBACK_MARKERS = [
    "could not verify",
    "cannot confidently provide",
    "could not be verified",
    "cannot provide a factual answer",
    "could not confirm",
]


def looks_like_safe_fallback(answer: str) -> bool:
    answer_l = str(answer or "").lower()
    return any(m in answer_l for m in SAFE_FALLBACK_MARKERS)


def _claim_to_fact_sentence(c: Dict[str, Any]) -> str | None:
    """
    Convert a verified TRUE claim into a simple natural-language fact.
    These facts should be preserved in the corrected answer.
    """
    s = c.get("s")
    p = c.get("p")
    o = c.get("o")

    if not s or not p or not o:
        return None

    if p == "written_by":
        return f'{s} was written by {o}.'

    if p == "publication_year":
        return f'{s} was published in {o}.'

    if p == "capital_of":
        return f'{s} is the capital of {o}.'

    if p == "located_in":
        return f'{s} is located in {o}.'

    if p == "headquarters_in":
        return f'{s} is headquartered in {o}.'

    if p == "founded_by":
        return f'{s} was founded by {o}.'

    if p == "founded_on":
        return f'{s} was founded in {o}.'

    if p == "educated_at":
        return f'{s} studied at {o}.'

    if p == "occupation":
        return f'{s} was a {o}.'

    if p == "instance_of":
        return f'{s} is a {o}.'

    return None


def _get_supported_facts(claims: List[Dict[str, Any]]) -> List[str]:
    facts = []

    for c in claims:
        if c.get("final_label") == "TRUE" or c.get("verdict") == "SUPPORTED":
            fact = _claim_to_fact_sentence(c)
            if fact and fact not in facts:
                facts.append(fact)

    return facts


def _extract_replacement_facts(c: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Extract graph-supported replacement facts from graph_evidence.
    These are used to correct a false claim.
    """
    facts: List[Tuple[str, str, str]] = []
    s = str(c.get("s") or "").strip()

    for e in c.get("graph_evidence", []):
        if not isinstance(e, dict):
            continue

        if e.get("type") == "entity_linking_failure":
            continue

        relation = e.get("relation") or e.get("predicate")
        obj = e.get("object")

        if relation and obj:
            facts.append((s, str(relation), str(obj)))

    return facts


def build_evidence_text(claims: List[Dict[str, Any]]) -> str:
    evidence_lines = []

    supported_facts = _get_supported_facts(claims)

    evidence_lines.append("SUPPORTED FACTS THAT MUST BE KEPT IF RELEVANT:")
    if supported_facts:
        for fact in supported_facts:
            evidence_lines.append(f"- {fact}")
    else:
        evidence_lines.append("- NONE")

    evidence_lines.append("")
    evidence_lines.append("BAD CLAIMS THAT MUST BE REMOVED OR CORRECTED:")

    bad_found = False

    for i, c in enumerate(claims, start=1):
        verdict = c.get("verdict")
        final_label = c.get("final_label")

        if verdict in {"REFUTED", "NEI"} or final_label in {"FALSE", "HALLUCINATION", "UNVERIFIABLE"}:
            bad_found = True

            s = c.get("s")
            p = c.get("p")
            o = c.get("o")

            evidence_lines.append(f"Claim {i}: ({s}, {p}, {o})")
            evidence_lines.append(f"  Verdict: {verdict}")
            evidence_lines.append(f"  Final label: {final_label}")

            if c.get("likely_hallucination") is True:
                evidence_lines.append("  Likely hallucination: true")

            if c.get("nei_type"):
                evidence_lines.append(f"  NEI type: {c['nei_type']}")

            if c.get("reason"):
                evidence_lines.append(f"  Reason: {c['reason']}")

            if c.get("explanation"):
                evidence_lines.append(f"  Explanation: {c['explanation']}")

            replacement_facts = _extract_replacement_facts(c)

            if replacement_facts:
                evidence_lines.append("  Correct replacement facts supported by the graph:")
                for subj, rel, obj in replacement_facts:
                    evidence_lines.append(f"    - ({subj}, {rel}, {obj})")
            else:
                evidence_lines.append("  Correct replacement facts supported by the graph: NONE")

            evidence_lines.append("")

    if not bad_found:
        evidence_lines.append("- NONE")

    return "\n".join(evidence_lines).strip()


def _needs_safe_fallback(claims: List[Dict[str, Any]]) -> bool:
    """
    Safe fallback is only allowed when there are bad hallucination-style claims
    AND there are no supported facts to preserve.
    """
    supported_facts = _get_supported_facts(claims)
    if supported_facts:
        return False

    bad_claims = [
        c for c in claims
        if c.get("verdict") in {"REFUTED", "NEI"}
        or c.get("final_label") in {"FALSE", "HALLUCINATION", "UNVERIFIABLE"}
    ]

    if not bad_claims:
        return False

    all_hallucination_style = True
    has_positive_replacement_fact = False

    for c in bad_claims:
        if c.get("likely_hallucination") is not True and c.get("final_label") != "HALLUCINATION":
            all_hallucination_style = False

        if _extract_replacement_facts(c):
            has_positive_replacement_fact = True

    return all_hallucination_style and not has_positive_replacement_fact


def _safe_fallback_answer(question: str, claims: List[Dict[str, Any]]) -> str:
    entities = []

    for c in claims:
        if c.get("verdict") in {"REFUTED", "NEI"} or c.get("final_label") in {"FALSE", "HALLUCINATION", "UNVERIFIABLE"}:
            if c.get("s"):
                entities.append(str(c["s"]))

    entity_text = entities[0] if entities else "this entity"

    return (
        f'I could not verify "{entity_text}" in the knowledge graph, '
        f'so I cannot confidently provide a factual answer to the question: {question}'
    )


def apply_correction_loop(question: str, original_answer: str, claims: List[Dict[str, Any]]) -> str:
    """
    System-2 correction loop.

    It should:
    1. Keep supported claims.
    2. Remove refuted/unverifiable claims.
    3. Use replacement graph evidence if available.
    4. Use fallback only if there are no supported facts to preserve.
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
1. Produce a corrected factual answer, not an explanation of the verification process.
2. Keep the SUPPORTED FACTS if they answer the question or are relevant to the original answer.
3. Remove every claim listed under BAD CLAIMS unless a graph-supported replacement fact is provided.
4. If replacement graph facts exist, use them as the corrected factual content.
5. Do not include refuted, false, NEI, unverifiable, or hallucinated claims in the corrected answer.
6. Do not say that a supported fact could not be verified.
7. Only use a cautious fallback if there are no supported facts and no replacement graph facts.
8. Do not mention "graph evidence", "verdict", "NEI", or "refuted" in the final answer.
9. Keep the answer concise and natural.

Return ONLY the corrected answer.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()


def answer_has_unverified_quoted_title_claim(answer: str) -> bool:
    text = str(answer or "")

    has_quoted_title = bool(re.search(r"['\"“”].+?['\"“”]", text))

    claim_patterns = [
        r"\bwas directed by\b",
        r"\bwas authored by\b",
        r"\bwas written by\b",
        r"\bwas created by\b",
        r"\bwas founded by\b",
        r"\bwas published in\b",
    ]

    has_claim_pattern = any(re.search(p, text, flags=re.IGNORECASE) for p in claim_patterns)

    return has_quoted_title and has_claim_pattern


def safe_no_claims_correction(question: str, answer: str) -> str:
    quoted = re.findall(r"['\"“”](.+?)['\"“”]", answer or "")
    entity = quoted[0] if quoted else "the mentioned entity"

    return (
        f'I could not verify "{entity}" in the knowledge graph, '
        f'so I cannot confidently provide a factual answer to the question: {question}'
    )