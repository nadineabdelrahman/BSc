# reference_verification.py
import requests
import re
from typing import Dict, Any, List, Optional
from rapidfuzz.fuzz import ratio
import xml.etree.ElementTree as ET
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
ARXIV_API_URL = "http://export.arxiv.org/api/query"
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
def _is_plausible_crossref_match(
    extracted_title: str,
    extracted_year: Optional[int],
    extracted_authors: List[str],
    candidate: Dict[str, Any],
    raw_text: str = ""
) -> bool:
    matched_title = _safe_get_title(candidate) or ""
    matched_year = _safe_get_year(candidate)
    matched_authors = _safe_get_authors(candidate)

    title_sim = _title_similarity(extracted_title or "", matched_title)

    # Reject clearly weak title matches
    if title_sim < 0.82:
        return False

    # Strong title match is enough
    if title_sim >= 0.92:
        return True

    # Otherwise require at least one supporting signal:
    year_ok = (not extracted_year or not matched_year or extracted_year == matched_year)
    author_ok = not _author_mismatch(extracted_authors, matched_authors, raw_text)

    return year_ok or author_ok

def _is_arxiv_reference(ref: Dict[str, Any]) -> bool:
    venue = (ref.get("venue") or "").lower()
    url = (ref.get("url") or "").lower()
    raw_text = (ref.get("raw_text") or "").lower()
    doi = (ref.get("doi") or "").lower()

    return (
        "arxiv" in venue
        or "arxiv.org" in url
        or "arxiv" in raw_text
        or "arxiv" in doi
    )


def _extract_arxiv_id(ref: Dict[str, Any]) -> Optional[str]:
    candidates = [
        ref.get("url") or "",
        ref.get("raw_text") or "",
        ref.get("doi") or "",
        ref.get("title") or ""
    ]

    patterns = [
        r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arxiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)"
    ]

    for text in candidates:
        if not text:
            continue
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)

    return None


def _safe_get_arxiv_title(entry: ET.Element) -> Optional[str]:
    title_el = entry.find("{http://www.w3.org/2005/Atom}title")
    if title_el is not None and title_el.text:
        return title_el.text.strip()
    return None


def _safe_get_arxiv_year(entry: ET.Element) -> Optional[int]:
    published_el = entry.find("{http://www.w3.org/2005/Atom}published")
    if published_el is not None and published_el.text:
        m = re.match(r"(\d{4})-", published_el.text.strip())
        if m:
            return int(m.group(1))
    return None


def _safe_get_arxiv_authors(entry: ET.Element) -> List[str]:
    authors = []
    for author_el in entry.findall("{http://www.w3.org/2005/Atom}author"):
        name_el = author_el.find("{http://www.w3.org/2005/Atom}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())
    return authors


def search_arxiv_by_id(arxiv_id: str) -> Optional[Dict[str, Any]]:
    if not arxiv_id or not str(arxiv_id).strip():
        return None

    arxiv_id = arxiv_id.strip()

    headers = {"User-Agent": USER_AGENT}
    params = {"id_list": arxiv_id}
    resp = requests.get(ARXIV_API_URL, params=params, headers=headers, timeout=20)

    if resp.status_code != 200:
        return None

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is None:
        return None

    title = _safe_get_arxiv_title(entry)
    year = _safe_get_arxiv_year(entry)
    authors = _safe_get_arxiv_authors(entry)

    # Do not accept empty / malformed records
    if not title or not str(title).strip():
        return None

    return {
        "title": title.strip(),
        "year": year,
        "authors": authors,
        "venue": "arXiv",
        "doi": None
    }


def search_arxiv_by_title(title: str, max_results: int = 5) -> List[Dict[str, Any]]:
    params = {
        "search_query": f'ti:"{title}"',
        "start": 0,
        "max_results": max_results
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(ARXIV_API_URL, params=params, headers=headers, timeout=20)

    if resp.status_code != 200:
        return []

    root = ET.fromstring(resp.text)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")

    results = []
    for entry in entries:
        results.append({
            "title": _safe_get_arxiv_title(entry),
            "year": _safe_get_arxiv_year(entry),
            "authors": _safe_get_arxiv_authors(entry),
            "venue": "arXiv",
            "doi": None
        })
    return results


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

def verify_single_reference_arxiv(ref: Dict[str, Any]) -> Dict[str, Any]:
    extracted_title = ref.get("title")
    extracted_year = ref.get("year")
    extracted_authors = ref.get("authors", [])

    best_match = None
    match_reason = None
    if not extracted_title:
        return {
            **ref,
            "verdict": "FABRICATED",
            "verification_source": "arxiv",
            "metadata_mismatches": ["missing_title"]
        }
    arxiv_id = _extract_arxiv_id(ref)
    if arxiv_id:
        item = search_arxiv_by_id(arxiv_id)
        if item:
            best_match = item
            match_reason = "arxiv_id"

    if not best_match and extracted_title:
        candidates = search_arxiv_by_title(extracted_title, max_results=5)
        if candidates:
            scored = []
            for c in candidates:
                candidate_title = c.get("title") or ""
                sim = _title_similarity(extracted_title, candidate_title)
                scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] >= 0.75:
                best_match = scored[0][1]
                match_reason = "arxiv_title"

    if not best_match:
        return {
            **ref,
            "verdict": "FABRICATED",
            "verification_source": "arxiv",
            "matched_title": None,
            "matched_doi": None,
            "matched_year": None,
            "matched_authors": [],
            "matched_venue": None,
            "match_reason": None,
            "metadata_mismatches": ["No matching arXiv record found"]
        }

    matched_title = best_match.get("title")
    matched_year = best_match.get("year")
    matched_authors = best_match.get("authors", [])
    matched_venue = best_match.get("venue")
    matched_doi = best_match.get("doi")

    mismatches = []

    title_sim = _title_similarity(extracted_title or "", matched_title or "")
    if extracted_title and title_sim < 0.85:
        mismatches.append("title")

    if extracted_year and matched_year and extracted_year != matched_year:
        mismatches.append("year")

    if _author_mismatch(extracted_authors, matched_authors, ref.get("raw_text", "")):
        mismatches.append("authors")

    mismatch_count = len(mismatches)

    if mismatch_count == 0:
        verdict = "SUPPORTED"
    elif title_sim < 0.75:
        verdict = "FABRICATED"
        mismatches = ["low_title_similarity"]
    elif mismatch_count >= 3:
        verdict = "FABRICATED"
    else:
        verdict = "PARTIAL"

    return {
        **ref,
        "verdict": verdict,
        "verification_source": "arxiv",
        "matched_title": matched_title,
        "matched_doi": matched_doi,
        "matched_year": matched_year,
        "matched_authors": matched_authors,
        "matched_venue": matched_venue,
        "match_reason": match_reason,
        "metadata_mismatches": mismatches
    }

def try_arxiv_fallback(ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try arXiv even if the reference was not explicitly tagged as arXiv.
    Useful when the LLM hallucinates conference venue metadata for a real preprint.
    """
    extracted_title = ref.get("title")
    if not extracted_title or not str(extracted_title).strip():
        return None

    candidates = search_arxiv_by_title(extracted_title, max_results=5)
    if not candidates:
        return None

    scored = []
    for c in candidates:
        candidate_title = c.get("title") or ""
        sim = _title_similarity(extracted_title, candidate_title)
        scored.append((sim, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_sim, best_match = scored[0]

    if best_sim < 0.88:
        return None

    matched_title = best_match.get("title")
    matched_year = best_match.get("year")
    matched_authors = best_match.get("authors", [])
    matched_venue = best_match.get("venue")
    extracted_year = ref.get("year")
    extracted_authors = ref.get("authors", [])

    mismatches = []
    if extracted_year and matched_year and extracted_year != matched_year:
        mismatches.append("year")

    if _author_mismatch(extracted_authors, matched_authors, ref.get("raw_text", "")):
        mismatches.append("authors")

    if len(mismatches) == 0:
        verdict = "SUPPORTED"
    elif len(mismatches) == 1:
        verdict = "PARTIAL"
    else:
        verdict = "PARTIAL"

    return {
        **ref,
        "verdict": verdict,
        "verification_source": "arxiv_fallback",
        "matched_title": matched_title,
        "matched_doi": None,
        "matched_year": matched_year,
        "matched_authors": matched_authors,
        "matched_venue": matched_venue,
        "match_reason": "arxiv_fallback_title",
        "metadata_mismatches": mismatches
    }

def verify_single_reference(ref: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verdicts:
    - SUPPORTED: strong match
    - PARTIAL: found something close, but metadata mismatches
    - FABRICATED: not found
    """
    if _is_arxiv_reference(ref):
        return verify_single_reference_arxiv(ref)
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
            if scored:
                top_candidate = scored[0][1]
                if _is_plausible_crossref_match(
                    extracted_title,
                    extracted_year,
                    extracted_authors,
                    top_candidate,
                    ref.get("raw_text", "")
                ):
                    best_match = top_candidate
                    match_reason = "title"
    
    if not best_match:
        arxiv_fallback = try_arxiv_fallback(ref)
        if arxiv_fallback is not None:
            return arxiv_fallback

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
        if _title_similarity(extracted_title, matched_title) < 0.82:
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

    if mismatch_count == 0:
        verdict = "SUPPORTED"

    elif "doi" in mismatches:
        verdict = "FABRICATED"

    elif "title" in mismatches and mismatch_count >= 3:
        verdict = "FABRICATED"

    elif mismatch_count >= 3:
        verdict = "FABRICATED"

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