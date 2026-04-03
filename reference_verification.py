# reference_verification.py
import requests
import re
from typing import Dict, Any, List, Optional
from rapidfuzz.fuzz import ratio

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
USER_AGENT = "HallucinationFirewall/1.0 (research project; contact: your_email@example.com)"

def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def _safe_get_title(item: Dict[str, Any]) -> Optional[str]:
    titles = item.get("title", [])
    if titles and isinstance(titles, list):
        return titles[0]
    return None

def _safe_get_year(item: Dict[str, Any]) -> Optional[int]:
    for field in ["published-print", "published-online", "issued", "created"]:
        part = item.get(field, {})
        date_parts = part.get("date-parts", [])
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None

def _safe_get_authors(item: Dict[str, Any]) -> List[str]:
    authors = []
    for a in item.get("author", []):
        given = a.get("given", "").strip()
        family = a.get("family", "").strip()
        full = f"{given} {family}".strip()
        if full:
            authors.append(full)
    return authors
def _safe_get_venue(item: Dict[str, Any]) -> Optional[str]:
    venues = item.get("container-title", [])
    if venues and isinstance(venues, list):
        return venues[0]
    return None
def _normalize_author_token(name: str) -> str:
    """
    Normalize an author string to a comparable last-name-like token.
    Examples:
      'Maynez, J.' -> 'maynez'
      'Joshua Maynez' -> 'maynez'
    """
    if not name:
        return ""

    n = name.strip().lower()
    n = re.sub(r"\bet al\.?\b", "", n).strip()
    n = re.sub(r"[^\w\s,.-]", "", n)

    if "," in n:
        last = n.split(",")[0].strip()
        return last

    parts = n.split()
    return parts[-1] if parts else ""


def _has_et_al(authors: List[str], raw_text: str = "") -> bool:
    joined = " ".join(authors).lower()
    raw = (raw_text or "").lower()
    return "et al" in joined or "et al" in raw


def _author_mismatch(extracted_authors: List[str], matched_authors: List[str], raw_text: str = "") -> bool:
    """
    Conservative author mismatch:
    - if 'et al.' appears, do not require full author overlap
    - compare normalized surnames
    """
    if not extracted_authors or not matched_authors:
        return False

    extracted_tokens = {
        _normalize_author_token(a) for a in extracted_authors if _normalize_author_token(a)
    }
    matched_tokens = {
        _normalize_author_token(a) for a in matched_authors if _normalize_author_token(a)
    }

    if not extracted_tokens or not matched_tokens:
        return False

    overlap = extracted_tokens & matched_tokens

    # If citation uses et al., one matching surname is enough
    if _has_et_al(extracted_authors, raw_text):
        return len(overlap) == 0

    # Otherwise still be practical: require at least one overlap
    return len(overlap) == 0
def _title_similarity(a: str, b: str) -> float:
    """
    Fuzzy title similarity using RapidFuzz.
    Returns a score in [0,1].
    """
    if not a or not b:
        return 0.0

    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    return ratio(a_norm, b_norm) / 100.0
def _venue_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return ratio(_normalize_text(a), _normalize_text(b)) / 100.0
def search_crossref_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    doi = doi.strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(f"{CROSSREF_WORKS_URL}/{doi}", headers=headers, timeout=20)
    if resp.status_code != 200:
        return None
    data = resp.json().get("message", {})
    return data or None

def search_crossref_by_title(title: str, rows: int = 5) -> List[Dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT}
    params = {"query.title": title, "rows": rows}
    resp = requests.get(CROSSREF_WORKS_URL, headers=headers, params=params, timeout=20)
    if resp.status_code != 200:
        return []
    return resp.json().get("message", {}).get("items", [])

def verify_single_reference(ref: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verdicts:
    - SUPPORTED: strong match
    - PARTIAL: found something close, but metadata mismatches
    - FABRICATED: not found
    """
    extracted_title = ref.get("title")
    extracted_year = ref.get("year")
    extracted_authors = ref.get("authors", [])
    extracted_doi = ref.get("doi")
    extracted_venue = ref.get("venue")

    best_match = None
    match_reason = None
# Empty title handling
# Empty title handling
    if (not extracted_title or not str(extracted_title).strip()) and not extracted_doi:
        return {
            **ref,
            "verdict": "FABRICATED",
            "verification_source": "crossref",
            "matched_title": None,
            "matched_doi": None,
            "matched_year": None,
            "matched_authors": [],
            "matched_venue": None,
            "match_reason": None,
            "metadata_mismatches": ["missing_title"]
        }
    # 1) DOI lookup is strongest
    if extracted_doi:
        item = search_crossref_by_doi(extracted_doi)
        if item:
            best_match = item
            match_reason = "doi"

    # 2) fallback to title search
    if not best_match and extracted_title:
        candidates = search_crossref_by_title(extracted_title, rows=5)
        if candidates:
            scored = []
            for c in candidates:
                candidate_title = _safe_get_title(c) or ""
                sim = _title_similarity(extracted_title, candidate_title)
                scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] >= 0.65:
                best_match = scored[0][1]
                match_reason = "title"
    
    if not best_match:
        return {
            **ref,
            "verdict": "FABRICATED",
            "verification_source": "crossref",
            "matched_title": None,
            "matched_doi": None,
            "matched_year": None,
            "matched_authors": [],
            "match_reason": None,
            "matched_venue": None,
            "metadata_mismatches": ["No matching scholarly record found"]
        }
    
    matched_title = _safe_get_title(best_match)
    matched_year = _safe_get_year(best_match)
    matched_authors = _safe_get_authors(best_match)
    matched_doi = best_match.get("DOI")
    matched_venue = _safe_get_venue(best_match)

    mismatches = []
    if extracted_doi:
        extracted_doi_clean = extracted_doi.strip().lower().replace("https://doi.org/", "")
        matched_doi_clean = (matched_doi or "").strip().lower()

        if not matched_doi_clean:
            mismatches.append("doi")
        elif extracted_doi_clean != matched_doi_clean:
            mismatches.append("doi")
    if extracted_title and _title_similarity(extracted_title, matched_title or "") < 0.85:
        mismatches.append("title")

    if extracted_year and matched_year and extracted_year != matched_year:
        mismatches.append("year")
    if extracted_venue and matched_venue:
        if _venue_similarity(extracted_venue, matched_venue) < 0.75:
            mismatches.append("venue")

    if _author_mismatch(extracted_authors, matched_authors, ref.get("raw_text", "")):
        mismatches.append("authors")

    # -------- STRONGER VERDICT LOGIC --------
 
    if extracted_title and matched_title:
        if _title_similarity(extracted_title, matched_title) < 0.75:
            return {
                **ref,
                "verdict": "FABRICATED",
                "verification_source": "crossref",
                "matched_title": matched_title,
                "matched_doi": matched_doi,
                "matched_year": matched_year,
                "matched_authors": matched_authors,
                "match_reason": match_reason,
                "matched_venue": matched_venue,
                "metadata_mismatches": ["low_title_similarity"]
            }


    mismatch_count = len(mismatches)

        # Strong case: everything matches
    if mismatch_count == 0:
        verdict = "SUPPORTED"

        # DOI mismatch is very strong evidence
    elif "doi" in mismatches:
        verdict = "FABRICATED"

        # If title is wrong + something else → very likely fabricated
    elif "title" in mismatches and (
            "authors" in mismatches or "year" in mismatches or "venue" in mismatches
        ):
        verdict = "FABRICATED"

        # Too many mismatches → fabricated
    elif mismatch_count >= 2:
        verdict = "FABRICATED"

        # Minor issue → partial
    else:
        verdict = "PARTIAL"

    return {
        **ref,
        "verdict": verdict,
        "verification_source": "crossref",
        "matched_title": matched_title,
        "matched_doi": matched_doi,
        "matched_year": matched_year,
        "matched_authors": matched_authors,
        "match_reason": match_reason,
        "matched_venue": matched_venue,
        "metadata_mismatches": mismatches
    }

def verify_references(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [verify_single_reference(ref) for ref in refs]