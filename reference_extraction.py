# reference_extraction.py
import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXTRACTION_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# Basic patterns for citation-like text.
INLINE_CITATION_PATTERNS = [
    r"\([A-Z][A-Za-z\-]+(?: et al\.)?,\s*\d{4}\)",         # (Smith, 2020), (Kolli et al., 2025)
    r"\[[0-9]+\]",                                         # [1], [2]
    r"\bdoi:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",           # DOI
    r"https?://doi\.org/10\.\d{4,9}/[-._;()/:A-Z0-9]+"     # DOI URL
]

def has_reference_signals(text: str) -> bool:
    if not text or not text.strip():
        return False

    for pattern in INLINE_CITATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    # Common bibliography cues
    bibliography_cues = [
        "references",
        "bibliography",
        "works cited",
        "citations"
    ]
    lowered = text.lower()
    return any(cue in lowered for cue in bibliography_cues)

def extract_reference_like_lines(text: str) -> List[str]:
    """
    Heuristic pre-pass:
    pull lines that look like references or bibliography entries.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = []

    ref_line_patterns = [
        r"^\[[0-9]+\]\s+",
        r"^[A-Z][A-Za-z\-]+.*\(\d{4}\)",
        r"^.+doi:\s*10\.",
        r"^.+https?://doi\.org/10\."
    ]

    for line in lines:
        if any(re.search(p, line, flags=re.IGNORECASE) for p in ref_line_patterns):
            candidates.append(line)

    # Also catch inline author-year references from full paragraph text
    inline_matches = re.findall(
        r"\(([A-Z][A-Za-z\-]+(?: et al\.)?),\s*(\d{4})\)",
        text
    )
    for author, year in inline_matches:
        candidates.append(f"{author}, {year}")

    # deduplicate while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def extract_references_with_llm(answer_text: str) -> List[Dict[str, Any]]:
    """
    Extract references from text and normalize them into a structured form.
    This is reference extraction, not claim extraction.
    """
    prompt = f"""
You are extracting references/citations mentioned in a model answer.

Task:
1. Find all references or citation mentions in the text.
2. Return ONLY references that are explicitly mentioned or strongly implied by a citation pattern.
3. Convert each reference into structured JSON.

Return a JSON array only.
Each item must have:
- raw_text: original citation/reference text as seen
- title: null if unknown
- authors: list of strings, empty list if unknown
- year: integer or null
- venue: null if unknown
- doi: null if unknown
- url: null if unknown
- citation_marker: null or the in-text marker such as "(Smith, 2020)" or "[3]"

Do not invent missing metadata.
Do not output explanation.

Text:
\"\"\"
{answer_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You extract references from text and return strict JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    refs = data.get("references", [])
    if not isinstance(refs, list):
        return []

    normalized = []
    for ref in refs:
        normalized.append({
            "raw_text": ref.get("raw_text"),
            "title": ref.get("title"),
            "authors": ref.get("authors", []),
            "year": ref.get("year"),
            "venue": ref.get("venue"),
            "doi": ref.get("doi"),
            "url": ref.get("url"),
            "citation_marker": ref.get("citation_marker"),
        })
    return normalized

def extract_references(answer_text: str) -> List[Dict[str, Any]]:
    """
    Main entry point.
    If there are no reference signals, return [] immediately.
    """
    if not has_reference_signals(answer_text):
        return []

    heuristic_candidates = extract_reference_like_lines(answer_text)

    # If there are weak signals but no clear candidate lines, still let the LLM inspect.
    refs = extract_references_with_llm(answer_text)

    # Add heuristic-only placeholders if LLM missed obvious inline citations
    raw_texts = {r.get("raw_text") for r in refs if r.get("raw_text")}
    for cand in heuristic_candidates:
        if cand not in raw_texts:
            refs.append({
                "raw_text": cand,
                "title": None,
                "authors": [],
                "year": None,
                "venue": None,
                "doi": None,
                "url": None,
                "citation_marker": cand if cand.startswith("(") or cand.startswith("[") else None,
            })

    return refs