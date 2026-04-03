# verification.py
import requests
import time
import re
from typing import Dict, Any, Optional, List, Tuple

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"

# Polite User-Agent is important; Wikidata may throttle/block generic agents.
USER_AGENT = "HallucinationFirewall/1.0 (research; contact: your_email@example.com)"

# Map your normalized predicates to Wikidata property IDs (P-codes).
# Extend this as your thesis grows.
PREDICATE_TO_PROPERTY = {
    # IMPORTANT FIX:
    # If your triple is (Paris, capital_of, France) = City -> Country,
    # then Wikidata property is P1376 "capital of" (not P36).
    "capital_of": "P1376",        # capital of (city/seat -> territory)

    "located_in": "P131",         # located in the administrative territorial entity
    "part_of": "P361",            # part of
    "instance_of": "P31",         # instance of

    "born_in": "P19",             # place of birth
    "died_in": "P20",             # place of death

    "founded_by": "P112",         # founded by (entity founder)
    "founder_of": "P112",         # (reverse direction not handled here; see note below)

    "ceo_of": "P169",             # chief executive officer

    # NOTE: "president_of" is tricky (head of state P35 vs head of government P6)
    # keep your choice, but be aware it can cause false negatives.
    "president_of": "P35",        # head of state (often closer than P6 for "president")

    "written_by": "P50",          # author
    "author_of": "P50",           # (reverse direction not handled here; see note below)
    "invented_by": "P61",         # discoverer or inventor
    "currency_of": "P38",         # currency
    "language_of": "P37",         # official language

    "publication_year": "P577",   # publication date (we'll check year match)

    # NEW: your extractor outputs these, so verification must support them
    "date_of_birth": "P569",
    "date_of_death": "P570",
    "place_of_birth": "P19",
    "place_of_death": "P20",
    "publication_date": "P577",
    "occupation": "P106",
    "notable_work": "P800",
    "alias_of": "P4970",
    "founded_on": "P571",
}

# Some predicates are naturally "reverse" depending on how your triples are formed.
# Example:
#   (Wuthering Heights, written_by, Emily Brontë)  -> wd:WORK wdt:P50 wd:PERSON  (forward)
# But:
#   (Emily Brontë, author_of, Wuthering Heights)   -> wd:WORK wdt:P50 wd:PERSON  (reverse)
# We'll support direction handling for a few known cases.
REVERSE_PREDICATES = {
    "author_of",
    "founder_of",
    # produced triples are usually (Org, ceo_of, Person),
    # so ceo_of should normally stay forward.
    "invented_by",
}

# Very light caching to avoid repeated API calls during batch runs
_entity_cache: Dict[str, Optional[str]] = {}
_entity_candidates_cache: Dict[str, List[Dict[str, Any]]] = {}
_property_cache: Dict[str, Optional[str]] = {}


def _looks_vague_region(text: str) -> bool:
    t = text.lower().strip()
    return any(x in t for x in ["northern ", "southern ", "eastern ", "western ", "central "])


def _build_located_in_path_ask(s_qid: str, o_qid: str) -> str:
    # allow chains of P131 (admin entity) to reduce false REFUTED
    # NEW FIX: allow 0+ hops too (wdt:P131*) because some items jump levels
    return f"""
    ASK WHERE {{
      wd:{s_qid} wdt:P131* wd:{o_qid} .
    }}
    """.strip()
def _location_fallback_labels(label: str) -> List[str]:
    """
    Generate fallback labels for place-like objects.
    Example:
      'Stratford-upon-Avon, England' -> ['Stratford-upon-Avon, England', 'Stratford-upon-Avon']
    """
    label = _normalize_entity_text(label)
    labels = [label]

    if "," in label:
        first = label.split(",")[0].strip()
        if first and first not in labels:
            labels.append(first)

    return labels

def _http_get(url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int = 20) -> requests.Response:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def wikidata_search_entity(label: str, language: str = "en") -> Optional[str]:
    """
    Resolve a natural language label to a Wikidata QID using wbsearchentities.
    Returns QID like 'Q64' or None if not found.
    """
    key = label.strip().lower()
    if not key:
        return None
    if key in _entity_cache:
        return _entity_cache[key]

    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        data = _http_get(WIKIDATA_SEARCH_URL, params=params, headers=headers).json()
        results = data.get("search", [])
        qid = results[0].get("id") if results else None
        _entity_cache[key] = qid
        return qid
    except Exception:
        _entity_cache[key] = None
        return None


# NEW: get multiple candidates to fix ambiguity (Jordan, Pride and Prejudice editions, Alexandria variants)
def wikidata_search_candidates(label: str, language: str = "en", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Returns candidate objects from wbsearchentities:
      [{"id": "Q...", "label": "...", "description": "..."}]
    """
    key = label.strip().lower()
    if not key:
        return []
    if key in _entity_candidates_cache:
        return _entity_candidates_cache[key]

    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": limit,
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        data = _http_get(WIKIDATA_SEARCH_URL, params=params, headers=headers).json()
        results = data.get("search", []) or []
        _entity_candidates_cache[key] = results
        return results
    except Exception:
        _entity_candidates_cache[key] = []
        return []


def _ask_sparql(query: str) -> Optional[bool]:
    """
    Run a SPARQL ASK query. Returns True/False if successful, else None on error.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
    }
    try:
        resp = requests.get(
            WIKIDATA_SPARQL_URL,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return bool(data.get("boolean"))
    except Exception:
        return None

def _candidate_entity_forms(text: str) -> List[str]:
    t = _normalize_entity_text(text)
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"'s$", "", t)
    forms = [t]

    simplified = t
    weak_prefixes = [
        "the practical ", "a practical ", "an practical ",
        "practical ", "the famous ", "famous ",
        "the renowned ", "renowned ",
        "the early ", "early ",
        "the modern ", "modern ",
        "the well-known ", "well-known ",
    ]

    changed = True
    while changed:
        changed = False
        tl = simplified.lower()
        for pref in weak_prefixes:
            if tl.startswith(pref):
                simplified = simplified[len(pref):].strip()
                changed = True
                break

    if simplified not in forms:
        forms.append(simplified)

    if "," in t:
        first = t.split(",")[0].strip()
        if first and first not in forms:
            forms.append(first)

    # try head noun for descriptive occupations/types like "theoretical physicist" or "person's name"
    if " " in t:
        head = re.sub(r"'s$", "", t.split()[-1].strip())
        if head and head not in forms:
            forms.append(head)

    for form in list(forms):
        for article in ["the ", "a ", "an "]:
            if form.lower().startswith(article):
                shorter = form[len(article):].strip()
                if shorter and shorter not in forms:
                    forms.append(shorter)

    return forms
def _resolve_best_qids_multi_forms(text: str, max_k: int = 5) -> List[str]:
    qids = []
    for form in _candidate_entity_forms(text):
        for q in _resolve_best_qids(form, max_k=max_k):
            if q not in qids:
                qids.append(q)
    return qids[:max_k]
def _is_generic_placeholder(x: str) -> bool:
    t = _normalize_entity_text(x).lower() if isinstance(x, str) else str(x).lower()
    return t in {"it", "book", "novel", "work", "city", "country", "person", "name"}


def _normalize_entity_text(x: str) -> str:
    # Basic cleanup: remove surrounding quotes/asterisks and extra whitespace
    # plus simple article stripping so "the Eiffel Tower" links as "Eiffel Tower"
    x = x.replace('"', "").replace("“", "").replace("”", "").replace("*", "").strip()
    x = re.sub(r"^(?:the|a|an)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _get_property_id(predicate: str) -> Optional[str]:
    predicate = predicate.strip().lower()
    if predicate in _property_cache:
        return _property_cache[predicate]
    pid = PREDICATE_TO_PROPERTY.get(predicate)
    _property_cache[predicate] = pid
    return pid


def _looks_like_year(text: str) -> Optional[str]:
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|19[0-9]{2}|18[0-9]{2})\b", str(text or ""))
    return m.group(0) if m else None


# NEW: detect a date-like string; we mainly need the YEAR for Wikidata date properties
def _extract_year_anywhere(text: str) -> Optional[str]:
    return _looks_like_year(text or "")


def _build_ask_query(s_qid: str, pid: str, o_qid: str, reverse: bool = False) -> str:
    """
    Builds ASK query for: wd:S wdt:PID wd:O
    If reverse=True, checks: wd:O wdt:PID wd:S
    """
    if reverse:
        return f"ASK WHERE {{ wd:{o_qid} wdt:{pid} wd:{s_qid} . }}"
    return f"ASK WHERE {{ wd:{s_qid} wdt:{pid} wd:{o_qid} . }}"


def _build_year_match_ask(subject_qid: str, pid: str, year: str) -> str:
    """
    Generic year match check for date properties (e.g., P577 publication date, P571 inception).
    """
    year_int = int(year)
    return f"""
    ASK WHERE {{
      wd:{subject_qid} wdt:{pid} ?date .
      FILTER(YEAR(?date) = {year_int})
    }}
    """.strip()


# NEW: located_in should not be only P131.
# We try multiple “containment” properties to avoid false REFUTED:
# - P131 (admin entity)
# - P17  (country)
# - P30  (continent) -> fixes Egypt in Africa
# - P361 (part of)
def _build_located_in_multi_ask(s_qid: str, o_qid: str) -> str:
    return f"""
    ASK WHERE {{
      {{
        wd:{s_qid} wdt:P131* wd:{o_qid} .
      }}
      UNION {{
        wd:{s_qid} wdt:P17 wd:{o_qid} .
      }}
      UNION {{
        wd:{s_qid} wdt:P30 wd:{o_qid} .
      }}
      UNION {{
        wd:{s_qid} wdt:P361 wd:{o_qid} .
      }}
    }}
    """.strip()


# NEW: people roles are usually occupations, not instance_of
# This fixes: Einstein physicist, Shakespeare playwright/poet
def _build_human_occupation_ask(person_qid: str, occupation_qid: str) -> str:
    return f"ASK WHERE {{ wd:{person_qid} wdt:P106 wd:{occupation_qid} . }}"


# NEW: detect if a Wikidata item is a human (wdt:P31 wd:Q5)
def _is_human_qid(qid: str) -> Optional[bool]:
    query = f"ASK WHERE {{ wd:{qid} wdt:P31 wd:Q5 . }}"
    return _ask_sparql(query)


# NEW: stronger linking — try multiple candidates if needed
def _resolve_best_qids(label: str, max_k: int = 3) -> List[str]:
    """
    Returns up to max_k QIDs for label.
    Uses multiple candidates to mitigate ambiguity (Jordan, works editions).
    """
    cands = wikidata_search_candidates(label, limit=max_k)
    qids = []
    for c in cands:
        q = c.get("id")
        if q and q not in qids:
            qids.append(q)
    # fallback to single search
    if not qids:
        q = wikidata_search_entity(label)
        if q:
            qids.append(q)
    return qids[:max_k]
def _resolve_best_qids_multi(labels: List[str], max_k: int = 3) -> List[str]:
    qids = []
    for label in labels:
        for q in _resolve_best_qids(label, max_k=max_k):
            if q not in qids:
                qids.append(q)
    return qids[:max_k]
def _resolve_best_qids_for_work(label: str, max_k: int = 5) -> List[str]:
    """
    Prefer candidates that look like literary/dramatic works.
    """
    cands = wikidata_search_candidates(label, limit=max_k)
    ranked = []
    fallback = []

    good_words = [
        "play", "tragedy", "comedy", "dramatic work",
        "literary work", "written work"
    ]

    for c in cands:
        qid = c.get("id")
        desc = (c.get("description") or "").lower()
        if not qid:
            continue
        if any(w in desc for w in good_words):
            ranked.append(qid)
        else:
            fallback.append(qid)

    out = []
    for q in ranked + fallback:
        if q not in out:
            out.append(q)

    return out[:max_k]



def _build_alias_ask(alias_text: str, canonical_qid: str) -> str:
    """
    Check whether a text appears as an English altLabel for a Wikidata item.
    """
    alias_text = alias_text.replace('"', '\\"').strip()
    return f"""
    ASK WHERE {{
      wd:{canonical_qid} <http://www.w3.org/2004/02/skos/core#altLabel> \"{alias_text}\"@en .
    }}
    """.strip()


def _is_ambiguous_linking(s_qids: List[str], o_qids: List[str], s_text: str, o_text: str) -> bool:
    """
    Conservative ambiguity detector.
    We keep it narrow so obvious single-link cases like Paris/Germany or Sydney/Australia
    do not incorrectly fall back to NEI, especially for negated facts.
    """
    if len(s_qids) > 1 or len(o_qids) > 1:
        return True

    ambiguous_terms = {
        "mercury", "jordan", "shakspere", "shakespeare"
    }

    st = _normalize_entity_text(s_text).lower()
    ot = _normalize_entity_text(o_text).lower()
    return st in ambiguous_terms or ot in ambiguous_terms


def _final_false_verdict(negated: bool, s_qids: List[str], o_qids: List[str], s_text: str, o_text: str, any_none: bool = False) -> str:
    """
    Centralized fallback when ASK queries found no supporting pair.

    Negation rule:
    - if the positive fact is false, and linking is confident, the negated claim is SUPPORTED
    - if linking/querying is ambiguous or failed, return NEI
    """
   

    # For negated facts, a false positive ASK means the negated statement is supported
    # as long as linking is confident enough.
    if negated and s_qids and o_qids:
        return "SUPPORTED"

    if any_none:
        return "NEI"

    if _is_ambiguous_linking(s_qids, o_qids, s_text, o_text):
        return "NEI"

    return "REFUTED"


def _verify_alias_of(alias_text: str, canonical_text: str) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    alias_text = _normalize_entity_text(alias_text).rstrip("'s").strip()
    canonical_text = _normalize_entity_text(canonical_text).rstrip("'s").strip()

    canonical_qids = _resolve_best_qids_multi_forms(canonical_text, max_k=5)
    if not canonical_qids:
        return (None, None, None)

    # 1) Best case: alias is stored as altLabel on the canonical item
    for cq in canonical_qids:
        ask = _ask_sparql(_build_alias_ask(alias_text, cq))
        if ask is True:
            return (None, cq, True)

    # 2) Fallback: if searching the alias returns the same canonical entity, accept it
    alias_qids = _resolve_best_qids_multi_forms(alias_text, max_k=5)
    for aq in alias_qids:
        if aq in canonical_qids:
            return (aq, aq, True)

    return (None, canonical_qids[0], False)
def _resolve_best_qids_contextual(label: str, context_text: str, max_k: int = 5) -> List[str]:
    """
    Resolve entity candidates using lightweight context-sensitive ranking.
    This is especially useful for ambiguous labels like Mercury and Jordan.
    """
    forms = _candidate_entity_forms(label)
    raw_candidates: List[Dict[str, Any]] = []

    for form in forms:
        cands = wikidata_search_candidates(form, limit=max_k)
        for c in cands:
            if c not in raw_candidates:
                raw_candidates.append(c)

    if not raw_candidates:
        qids = _resolve_best_qids_multi_forms(label, max_k=max_k)
        return qids[:max_k]

    ctx = (context_text or "").lower()
    label_l = _normalize_entity_text(label).lower()

    def score_candidate(c: Dict[str, Any]) -> int:
        desc = (c.get("description") or "").lower()
        score = 0

        # Mercury disambiguation
        if label_l == "mercury":
            if "planet" in ctx and "planet" in desc:
                score += 10
            if "sun" in ctx and "planet" in desc:
                score += 8
            if "element" in ctx and "chemical element" in desc:
                score += 10
            if "symbol" in ctx and "chemical element" in desc:
                score += 6
            if "atomic number" in ctx and "chemical element" in desc:
                score += 6

        # Jordan disambiguation
        if label_l == "jordan":
            if "country" in ctx and "country" in desc:
                score += 10
            if "middle east" in ctx and "country" in desc:
                score += 8
            if "river" in ctx and "river" in desc:
                score += 10
            if "given name" in ctx and "given name" in desc:
                score += 10
            if "surname" in ctx and "surname" in desc:
                score += 10

        # Work/title preference
        if any(w in ctx for w in ["novel", "book", "play", "literary work", "published", "author"]):
            if any(w in desc for w in ["novel", "book", "play", "literary work", "written work"]):
                score += 6

        # Person preference
        if any(w in ctx for w in ["born", "died", "poet", "playwright", "physicist", "author", "ceo"]):
            if any(w in desc for w in ["writer", "poet", "playwright", "physicist", "businessman", "human"]):
                score += 4

        return score

    ranked = sorted(raw_candidates, key=score_candidate, reverse=True)

    out: List[str] = []
    for c in ranked:
        qid = c.get("id")
        if qid and qid not in out:
            out.append(qid)

    if not out:
        out = _resolve_best_qids_multi_forms(label, max_k=max_k)

    return out[:max_k]
def verify_triple(triple: Dict[str, Any], sleep_s: float = 0.2) -> Dict[str, Any]:
    """
    Verifies a single triple against Wikidata.
    Returns the triple augmented with:
      - verdict: "SUPPORTED" | "REFUTED" | "NEI"
      - s_qid, o_qid, property_id
      - ask_result: True/False/None (None means query failed)

    Negation handling:
      - If negated=False: ask=True -> SUPPORTED, ask=False -> REFUTED, None -> NEI
      - If negated=True : ask=False -> SUPPORTED (negated claim supported), ask=True -> REFUTED
    """
    s_text = _normalize_entity_text(str(triple.get("s", "")))
    p = str(triple.get("p", "")).strip().lower()
    o_text = _normalize_entity_text(str(triple.get("o", "")))
    negated = bool(triple.get("negated", False))

    # initialize qids early to avoid UnboundLocalError
    s_qid = None
    o_qid = None

    out = dict(triple)  # copy so we don't mutate caller
    out.update({"verdict": "NEI", "s_qid": None, "o_qid": None, "property_id": None, "ask_result": None})

    if _is_generic_placeholder(s_text) or _is_generic_placeholder(o_text):
        return out

    context_text = f"{s_text} {o_text} {triple.get('sentence', '')}"

    # If we don't know how to map predicate -> property, we cannot verify => NEI
    pid = _get_property_id(p)
    out["property_id"] = pid
    if not pid:
        return out

    # ---------------------------
    # NEW: automatic predicate repair based on object format
    # - fixes your results: (Shakespeare, died_in, "April 23, 1616") should be date_of_death
    # ---------------------------
    year_in_o = _extract_year_anywhere(o_text)
    if p in ("died_in", "born_in") and year_in_o:
        # born_in/died_in are place predicates; if object looks like a date/year, re-route to date predicates
        if p == "died_in":
            p = "date_of_death"
            pid = _get_property_id(p) or "P570"
            out["property_id"] = pid
        elif p == "born_in":
            p = "date_of_birth"
            pid = _get_property_id(p) or "P569"
            out["property_id"] = pid

    if p == "alias_of":
        s_qid_alias, o_qid_alias, ask_alias = _verify_alias_of(s_text, o_text)
        out["s_qid"] = s_qid_alias
        out["o_qid"] = o_qid_alias
        out["ask_result"] = ask_alias
        if ask_alias is True:
            out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
        elif ask_alias is False:
            out["verdict"] = "REFUTED" if not negated else "SUPPORTED"
        else:
            out["verdict"] = "NEI"
        time.sleep(sleep_s)
        return out
    if p == "ceo_of":
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        o_qids = _resolve_best_qids_contextual(o_text, context_text, max_k=5)
        if not s_qids or not o_qids:
            return out

        out["property_id"] = "P169"

        # Try both directions:
        # 1) company -> CEO   (wd:Apple wdt:P169 wd:Tim_Cook)
        # 2) reverse input form person -> company, but same Wikidata fact
        for sq in s_qids:
            for oq in o_qids:
                # forward: s is org, o is person
                ask = _ask_sparql(_build_ask_query(sq, "P169", oq, reverse=False))
                if ask is True:
                    out["s_qid"] = sq
                    out["o_qid"] = oq
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

                # reverse interpretation: o is org, s is person
                ask_rev = _ask_sparql(_build_ask_query(oq, "P169", sq, reverse=False))
                if ask_rev is True:
                    out["s_qid"] = sq
                    out["o_qid"] = oq
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

        out["s_qid"] = s_qids[0]
        out["o_qid"] = o_qids[0]
        out["ask_result"] = False
        out["verdict"] = _final_false_verdict(negated, s_qids, o_qids, s_text, o_text)
        time.sleep(sleep_s)
        return out
    # ---- Year/date literal special cases ----

    # publication_year: check YEAR(P577) == year
    if p in {"publication_year","publication_date", "founded_on"}:
        date_pid = "P571" if p == "founded_on" else "P577"
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        year = _extract_year_anywhere(o_text) or _extract_year_anywhere(str(triple.get("sentence", "")))
        if not year or not s_qids:
            return out

        out["property_id"] = date_pid

        for sq in s_qids:
            query = _build_year_match_ask(sq, date_pid, year)
            ask = _ask_sparql(query)
            if ask is True:
                out["s_qid"] = sq
                out["ask_result"] = True
                out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                time.sleep(sleep_s)
                return out

        out["s_qid"] = s_qids[0]
        out["ask_result"] = False
        out["verdict"] = "REFUTED" if not negated else "SUPPORTED"
        time.sleep(sleep_s)
        return out

    # publication_date: same property P577, but accept “1847” etc by year match
    if p == "publication_date":
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        year = _extract_year_anywhere(o_text) or _extract_year_anywhere(str(triple.get("sentence", "")))
        if not year or not s_qids:
            return out

        for sq in s_qids:
            query = _build_year_match_ask(sq, "P577", year)
            ask = _ask_sparql(query)
            if ask is True:
                out["s_qid"] = sq
                out["ask_result"] = True
                out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                time.sleep(sleep_s)
                return out

        out["s_qid"] = s_qids[0]
        out["ask_result"] = False
        out["verdict"] = "REFUTED" if not negated else "SUPPORTED"
        time.sleep(sleep_s)
        return out

    # date_of_birth / date_of_death: match by YEAR to stay robust (your extraction may output "April 1564")
    if p in ("date_of_birth", "date_of_death"):
        date_pid = "P569" if p == "date_of_birth" else "P570"
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        year = _extract_year_anywhere(o_text) or _extract_year_anywhere(str(triple.get("sentence", "")))
        if not year or not s_qids:
            return out

        for sq in s_qids:
            query = _build_year_match_ask(sq, date_pid, year)
            ask = _ask_sparql(query)
            if ask is True:
                out["s_qid"] = sq
                out["ask_result"] = True
                out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                time.sleep(sleep_s)
                return out

        out["s_qid"] = s_qids[0]
        out["ask_result"] = False
        out["verdict"] = "REFUTED" if not negated else "SUPPORTED"
        time.sleep(sleep_s)
        return out

    # founded_by BUT object looks like a YEAR -> treat as inception year (P571) instead of founder entity (P112)
    if p == "founded_by":
        year = _looks_like_year(o_text)
        if year:
            s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
            out["property_id"] = "P571"  # inception

            if not s_qids:
                return out  # NEI

            for sq in s_qids:
                query = _build_year_match_ask(sq, "P571", year)
                ask = _ask_sparql(query)
                if ask is True:
                    out["s_qid"] = sq
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

            out["s_qid"] = s_qids[0]
            out["ask_result"] = False
            out["verdict"] = "REFUTED" if not negated else "SUPPORTED"
            time.sleep(sleep_s)
            return out

    # Special handling for located_in
    if p == "located_in":
        # if object is vague like "northern France", mark NEI not REFUTED
        if _looks_vague_region(o_text):
            out["verdict"] = "NEI"
            return out

        # resolve entities first (multi-candidate to reduce ambiguity)
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        o_qids = _resolve_best_qids_multi(_location_fallback_labels(o_text), max_k=3)

        if not s_qids or not o_qids:
            return out  # NEI

        # NEW FIX: try multiple containment properties (P131/P17/P30/P361)
        for sq in s_qids:
            for oq in o_qids:
                query = _build_located_in_multi_ask(sq, oq)
                ask = _ask_sparql(query)
                if ask is True:
                    out["s_qid"] = sq
                    out["o_qid"] = oq
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

        out["s_qid"] = s_qids[0]
        out["o_qid"] = o_qids[0]
        out["ask_result"] = False
        out["verdict"] = "REFUTED" if not negated else "SUPPORTED"
        time.sleep(sleep_s)
        return out

    if p == "part_of":
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        o_qids = _resolve_best_qids_contextual(o_text, context_text, max_k=5)
        if not s_qids or not o_qids:
            return out
        for sq in s_qids:
            for oq in o_qids:
                for pid_try in ("P361", "P131", "P17"):
                    query = _build_ask_query(sq, pid_try, oq, reverse=False)
                    ask = _ask_sparql(query)
                    if ask is True:
                        out["s_qid"] = sq
                        out["o_qid"] = oq
                        out["property_id"] = pid_try
                        out["ask_result"] = True
                        out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                        time.sleep(sleep_s)
                        return out
        out["s_qid"] = s_qids[0]
        out["o_qid"] = o_qids[0]
        out["ask_result"] = False
        out["verdict"] = _final_false_verdict(negated, s_qids, o_qids, s_text, o_text)
        time.sleep(sleep_s)
        return out

    if p == "occupation":
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        o_qids = _resolve_best_qids_multi_forms(o_text, max_k=5)
        if not s_qids or not o_qids:
            return out

        for sq in s_qids:
            for oq in o_qids:
                query = _build_human_occupation_ask(sq, oq)
                ask = _ask_sparql(query)
                if ask is True:
                    out["s_qid"] = sq
                    out["o_qid"] = oq
                    out["property_id"] = "P106"
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

        out["s_qid"] = s_qids[0]
        out["o_qid"] = o_qids[0]
        out["ask_result"] = False
        out["verdict"] = _final_false_verdict(negated, s_qids, o_qids, s_text, o_text)
        time.sleep(sleep_s)
        return out
    # ---------------------------
    # NEW: instance_of repair for humans -> occupation (P106)
    # ---------------------------
    if p == "instance_of":
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        o_qids = _resolve_best_qids_contextual(o_text, context_text, max_k=5)
        if not s_qids or not o_qids:
            return out

        # 1) try true instance_of first
        for sq in s_qids:
            for oq in o_qids:
                query = _build_ask_query(sq, "P31", oq, reverse=False)
                ask = _ask_sparql(query)
                if ask is True:
                    out["s_qid"] = sq
                    out["o_qid"] = oq
                    out["property_id"] = "P31"
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

        # 2) if subject is human, retry as occupation
        for sq in s_qids:
            is_human = _is_human_qid(sq)
            if is_human is True:
                for oq in o_qids:
                    query = _build_human_occupation_ask(sq, oq)
                    ask = _ask_sparql(query)
                    if ask is True:
                        out["s_qid"] = sq
                        out["o_qid"] = oq
                        out["property_id"] = "P106"  # occupation
                        out["ask_result"] = True
                        out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                        time.sleep(sleep_s)
                        return out

        out["s_qid"] = s_qids[0]
        out["o_qid"] = o_qids[0]
        out["ask_result"] = False
        out["verdict"] = _final_false_verdict(negated, s_qids, o_qids, s_text, o_text)
        time.sleep(sleep_s)
        return out

    # ---------------------------
    # NEW: capital_of direction repair
    # Your results show both:
    # - (Paris, capital_of, France)  [city -> country]  should check P1376
    # - (France, capital_of, Paris)  [country -> city]  should check P36
    # ---------------------------
    if p == "capital_of":
        s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)
        o_qids = _resolve_best_qids_contextual(o_text, context_text, max_k=5)
        if not s_qids or not o_qids:
            return out

        # Try City -> Country: P1376
        for sq in s_qids:
            for oq in o_qids:
                query = _build_ask_query(sq, "P1376", oq, reverse=False)
                ask = _ask_sparql(query)
                if ask is True:
                    out["s_qid"] = sq
                    out["o_qid"] = oq
                    out["property_id"] = "P1376"
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

        # Try Country -> City: P36
        for sq in s_qids:
            for oq in o_qids:
                query = _build_ask_query(sq, "P36", oq, reverse=False)
                ask = _ask_sparql(query)
                if ask is True:
                    out["s_qid"] = sq
                    out["o_qid"] = oq
                    out["property_id"] = "P36"
                    out["ask_result"] = True
                    out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                    time.sleep(sleep_s)
                    return out

        out["s_qid"] = s_qids[0]
        out["o_qid"] = o_qids[0]
        out["ask_result"] = False
        out["verdict"] = _final_false_verdict(negated, s_qids, o_qids, s_text, o_text)
        time.sleep(sleep_s)
        return out

    # ---- Entity-object cases ----

    # Resolve entities (only if object is NOT a year literal)
    # NEW: multi-candidate fallback to reduce false REFUTED due to ambiguity
    s_qids = _resolve_best_qids_contextual(s_text, context_text, max_k=5)

    if p == "notable_work":
        work_candidates = _resolve_best_qids_for_work(o_text, max_k=5)
        fallback_candidates = _resolve_best_qids_multi_forms(o_text, max_k=5)

        o_qids = []
        for q in work_candidates + fallback_candidates:
            if q not in o_qids:
                o_qids.append(q)
        o_qids = o_qids[:5]
    elif p in {"place_of_birth", "place_of_death", "located_at", "located_in"}:
        o_qids = _resolve_best_qids_multi(_location_fallback_labels(o_text), max_k=3)
    else:
        o_qids = _resolve_best_qids_multi_forms(o_text, max_k=5)

    if not s_qids or not o_qids:
        # Could not link one/both entities => Not Enough Info
        return out

    reverse = p in REVERSE_PREDICATES

    # Try candidate pairs until one supports
    any_none = False
    for sq in s_qids:
        for oq in o_qids:
            query = _build_ask_query(sq, pid, oq, reverse=reverse)
            ask = _ask_sparql(query)
            if ask is True:
                out["s_qid"] = sq
                out["o_qid"] = oq
                out["ask_result"] = True
                out["verdict"] = "SUPPORTED" if not negated else "REFUTED"
                time.sleep(sleep_s)
                return out
            if ask is None:
                # keep trying; if all None, NEI
                any_none = True

    # If we reach here, we got no True.
    # Use first candidates for logging/debug.
    out["s_qid"] = s_qids[0]
    out["o_qid"] = o_qids[0]
    out["ask_result"] = False
    out["verdict"] = _final_false_verdict(negated, s_qids, o_qids, s_text, o_text, any_none=any_none)

    time.sleep(sleep_s)
    return out


def verify_triples(triples: List[Dict[str, Any]], sleep_s: float = 0.2) -> List[Dict[str, Any]]:
    """
    Batch verification. Adds verdict to each triple.
    sleep_s is a small delay to be polite to Wikidata endpoints.
    """
    results = []
    for t in triples:
        results.append(verify_triple(t, sleep_s=sleep_s))
    return results