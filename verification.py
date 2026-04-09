import re
import time
from typing import Dict, Any, Optional, List, Tuple

import requests

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "HallucinationFirewall/1.0 (research; contact: your_email@example.com)"

PREDICATE_TO_PROPERTY = {
    "capital_of": "P1376",
    "located_in": "P131",
    "part_of": "P361",
    "instance_of": "P31",
    "born_in": "P19",
    "died_in": "P20",
    "place_of_birth": "P19",
    "place_of_death": "P20",
    "date_of_birth": "P569",
    "date_of_death": "P570",
    "founded_by": "P112",
    "founder_of": "P112",
    "ceo_of": "P169",
    "president_of": "P35",
    "written_by": "P50",
    "author_of": "P50",
    "invented_by": "P61",
    "currency_of": "P38",
    "language_of": "P37",
    "publication_year": "P577",
    "publication_date": "P577",
    "occupation": "P106",
    "notable_work": "P800",
    "alias_of": "P4970",
    "founded_on": "P571",
    "headquarters_in": "P159",
    "country_of_origin": "P495",
    "genre": "P136",
    "spouse": "P26",
    "child_of": "P40",
    "educated_at": "P69",
    "population_of": "P1082",
    "area_of": "P2046",
    "located_at": "P276",
}

REVERSE_PREDICATES = {"author_of", "founder_of", "president_of", "child_of"}
WEAK_REFUTATION_PREDICATES = {"located_in", "part_of","instance_of", "occupation", "notable_work", "educated_at", "located_at"}

_entity_cache: Dict[str, Optional[str]] = {}
_entity_candidates_cache: Dict[str, List[Dict[str, Any]]] = {}


def _http_get(url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int = 20) -> requests.Response:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def _normalize_entity_text(x: str) -> str:
    x = str(x or "")
    x = x.replace('"', '').replace('“', '').replace('”', '').replace('*', '').strip()
    x = re.sub(r"^[-•*]+\s*", "", x)
    x = re.sub(r"^(?:the|a|an)\s+", "", x, flags=re.I)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _looks_like_year(text: str) -> Optional[str]:
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|18[0-9]{2})\b", str(text or ""))
    return m.group(0) if m else None


def wikidata_search_candidates(label: str, language: str = "en", limit: int = 6) -> List[Dict[str, Any]]:
    key = label.strip().lower()
    if not key:
        return []
    cache_key = f"{language}:{limit}:{key}"
    if cache_key in _entity_candidates_cache:
        return _entity_candidates_cache[cache_key]
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": limit,
    }
    headers = {"User-Agent": USER_AGENT}
    last: List[Dict[str, Any]] = []
    for attempt in range(3):
        try:
            data = _http_get(WIKIDATA_SEARCH_URL, params=params, headers=headers).json()
            out = data.get("search", []) or []
            if out:
                _entity_candidates_cache[cache_key] = out
                return out
            last = out
        except Exception:
            last = []
        time.sleep(0.15 * (attempt + 1))
    _entity_candidates_cache[cache_key] = last
    return last


def wikidata_search_entity(label: str, language: str = "en") -> Optional[str]:
    key = f"{language}:{label.strip().lower()}"
    if not label.strip():
        return None
    if key in _entity_cache:
        return _entity_cache[key]
    cands = wikidata_search_candidates(label, language=language, limit=3)
    qid = cands[0].get("id") if cands else None
    if qid is not None:
        _entity_cache[key] = qid
    return qid


def _candidate_entity_forms(text: str) -> List[str]:
    t = _normalize_entity_text(text)
    forms = [t]
    if "," in t:
        first = t.split(",")[0].strip()
        if first and first not in forms:
            forms.append(first)
    if t.endswith(" Inc") and t not in forms:
        forms.append(t)
    if t.endswith(" Inc."):
        forms.append(t[:-1])
    return [f for f in forms if f]


def _resolve_best_qids(label: str, max_k: int = 5) -> List[str]:
    out: List[str] = []
    for form in _candidate_entity_forms(label):
        for c in wikidata_search_candidates(form, limit=max_k):
            qid = c.get("id")
            if qid and qid not in out:
                out.append(qid)
    return out[:max_k]


def _resolve_best_qids_contextual(label: str, context_text: str, max_k: int = 5) -> List[str]:
    raw_candidates: List[Dict[str, Any]] = []
    for form in _candidate_entity_forms(label):
        for c in wikidata_search_candidates(form, limit=max_k):
            if c not in raw_candidates:
                raw_candidates.append(c)
    if not raw_candidates:
        return _resolve_best_qids(label, max_k=max_k)

    ctx = (context_text or "").lower()
    label_l = _normalize_entity_text(label).lower()

    def score_candidate(c: Dict[str, Any]) -> int:
        cand_label = (c.get("label") or "").lower()
        desc = (c.get("description") or "").lower()
        score = 0

        if label_l == "jordan":
            if any(k in ctx for k in ["country", "capital", "amman", "middle east"]):
                if "country" in desc or "hashemite kingdom" in desc:
                    score += 20
            if any(k in ctx for k in ["basketball", "nba", "bulls", "mvp", "championship"]):
                if "basketball" in desc or "player" in desc:
                    score += 20
            if "river" in ctx and "river" in desc:
                score += 15

        if label_l == "apple" or label_l == "apple inc":
            if any(k in ctx for k in ["company", "technology", "iphone", "mac", "inc", "founded by"]):
                if "technology company" in desc or "company" in desc or cand_label == "apple inc.":
                    score += 20
            if any(k in ctx for k in ["fruit", "tree", "edible"]):
                if "fruit" in desc or "apple" in desc:
                    score += 15

        if label_l == "tokyo":
            if any(k in ctx for k in ["capital", "japan", "city"]):
                if any(k in desc for k in ["capital city", "metropolis", "prefecture", "city"]):
                    score += 15
                if "japan" in desc:
                    score += 10

        if label_l.startswith("facebook"):
            if any(k in ctx for k in ["company", "launched", "mark zuckerberg", "technology"]):
                if "company" in desc or "social media" in desc:
                    score += 20

        # General contextual cues.
        if any(k in ctx for k in ["company", "founded", "headquartered", "ceo"]):
            if any(k in desc for k in ["company", "corporation", "business", "organization"]):
                score += 6
        if any(k in ctx for k in ["country", "capital", "continent", "middle east"]):
            if any(k in desc for k in ["country", "city", "capital", "territory"]):
                score += 5
        if any(k in ctx for k in ["player", "basketball", "scientist", "poet", "author"]):
            if any(k in desc for k in ["player", "scientist", "poet", "writer", "human"]):
                score += 5

        # Prefer exact label match.
        if cand_label == label_l:
            score += 3
        if label_l.replace(" inc", "") == cand_label.replace(" inc.", ""):
            score += 2

        return score

    ranked = sorted(raw_candidates, key=score_candidate, reverse=True)
    out: List[str] = []
    for c in ranked:
        qid = c.get("id")
        if qid and qid not in out:
            out.append(qid)
    return out[:max_k]

def _desc(c: Dict[str, Any]) -> str:
    return (c.get("description") or "").lower()

def _label(c: Dict[str, Any]) -> str:
    return (c.get("label") or "").lower()

def _candidate_role_score(c: Dict[str, Any], predicate: str, is_subject: bool, context_text: str = "") -> int:
    desc = _desc(c)
    lab = _label(c)
    pred = (predicate or "").strip().lower()
    ctx = (context_text or "").lower()
    score = 0

    # capital_of: subject should be city/capital, object should be country/state
    if pred == "capital_of":
        if is_subject:
            if any(k in desc for k in ["city", "capital", "commune", "metropolis"]):
                score += 25
            if "country" in desc:
                score -= 10
        else:
            if any(k in desc for k in ["country", "state", "sovereign state", "republic", "kingdom"]):
                score += 25
            if any(k in desc for k in ["city", "capital", "commune"]):
                score -= 10

    # located_in: subject often physical entity/place; object must be place-like
    elif pred == "located_in":
        if is_subject:
            if any(k in desc for k in [
                "river", "city", "town", "village", "building", "museum", "mountain",
                "university", "airport", "bridge", "lake", "country", "region", "island"
            ]):
                score += 20
        else:
            if any(k in desc for k in [
                "country", "continent", "city", "state", "province", "region",
                "administrative territorial entity", "territory"
            ]):
                score += 25

    elif pred == "educated_at":
        if is_subject:
            if any(k in desc for k in ["physicist", "scientist", "writer", "human", "person", "businessperson", "entrepreneur"]):
                score += 15
        else:
            if any(k in desc for k in ["university", "college", "school", "educational institution"]):
                score += 30

    elif pred == "written_by":
        if is_subject:
            if any(k in desc for k in ["book", "novel", "work", "poem", "play", "literary work", "film"]):
                score += 25
            if "human" in desc:
                score -= 10
        else:
            if any(k in desc for k in ["human", "writer", "author", "poet", "novelist"]):
                score += 25

    elif pred == "founded_by":
        if is_subject:
            if any(k in desc for k in ["company", "organization", "corporation", "business"]):
                score += 25
        else:
            if any(k in desc for k in ["human", "businessperson", "entrepreneur", "person"]):
                score += 20

    elif pred == "invented_by":
        if is_subject:
            if any(k in desc for k in ["device", "invention", "telephone", "technology", "object", "artifact"]):
                score += 20
            if "human" in desc:
                score -= 10
        else:
            if any(k in desc for k in ["human", "inventor", "scientist", "engineer", "person"]):
                score += 25

    elif pred == "instance_of":
        if not is_subject:
            if any(k in desc for k in [
                "class", "type", "category", "Wikimedia disambiguation page",
                "concept", "country", "city", "river", "human", "fictional country"
            ]):
                score += 15

    # extra boost from text itself
    if pred == "located_in" and "river" in ctx and "river" in desc:
        score += 20
    if pred == "capital_of" and "capital" in ctx and any(k in desc for k in ["city", "capital"]):
        score += 10
    if pred == "educated_at" and any(k in ctx for k in ["student", "phd", "university"]):
        if any(k in desc for k in ["university", "school", "college"]):
            score += 10

    # prefer exact label
    if lab == _normalize_entity_text(c.get("label", "")).lower():
        score += 1

    return score
def _ask_sparql(query: str) -> Optional[bool]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    try:
        resp = requests.get(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"}, headers=headers, timeout=30)
        resp.raise_for_status()
        return bool(resp.json().get("boolean"))
    except Exception:
        return None


def _select_sparql(query: str) -> List[Dict[str, str]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    try:
        resp = requests.get(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"}, headers=headers, timeout=30)
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        out = []
        for b in bindings:
            row = {k: v.get("value") for k, v in b.items()}
            out.append(row)
        return out
    except Exception:
        return []


def _build_ask_query(s_qid: str, pid: str, o_qid: str, reverse: bool = False) -> str:
    if reverse:
        return f"ASK WHERE {{ wd:{o_qid} wdt:{pid} wd:{s_qid} . }}"
    return f"ASK WHERE {{ wd:{s_qid} wdt:{pid} wd:{o_qid} . }}"


def _build_year_match_ask(subject_qid: str, pid: str, year: str) -> str:
    return f"ASK WHERE {{ wd:{subject_qid} p:{pid} ?st . ?st ps:{pid} ?dt . FILTER(YEAR(?dt) = {year}) }}"


def _build_located_in_union_ask(s_qid: str, o_qid: str) -> str:
    return f"""
    ASK WHERE {{
      {{ wd:{s_qid} wdt:P131* wd:{o_qid} . }}
      UNION {{ wd:{s_qid} wdt:P17 wd:{o_qid} . }}
      UNION {{ wd:{s_qid} wdt:P30 wd:{o_qid} . }}
      UNION {{ wd:{s_qid} wdt:P361 wd:{o_qid} . }}
      UNION {{ wd:{s_qid} wdt:P206 wd:{o_qid} . }}
      UNION {{ wd:{s_qid} wdt:P276 wd:{o_qid} . }}
    }}
    """.strip()


def _get_property_id(predicate: str) -> Optional[str]:
    return PREDICATE_TO_PROPERTY.get((predicate or "").strip().lower())


def _rank_candidates_for_role(
    label: str,
    predicate: str,
    is_subject: bool,
    context_text: str,
    max_k: int = 5
) -> List[str]:
    raw_candidates: List[Dict[str, Any]] = []
    for form in _candidate_entity_forms(label):
        for c in wikidata_search_candidates(form, limit=max_k):
            if c not in raw_candidates:
                raw_candidates.append(c)

    if not raw_candidates:
        return []

    ranked = sorted(
        raw_candidates,
        key=lambda c: (
            _candidate_role_score(c, predicate, is_subject, context_text),
            1 if (_label(c) == _normalize_entity_text(label).lower()) else 0
        ),
        reverse=True
    )

    out: List[str] = []
    for c in ranked:
        qid = c.get("id")
        if qid and qid not in out:
            out.append(qid)
    return out[:max_k]


def _resolve_subject_object_qids(
    s: str,
    o: str,
    sentence: str,
    predicate: str
) -> Tuple[Optional[str], Optional[str]]:
    context = f"{sentence} :: predicate={predicate} :: subject={s} :: object={o}"

    subject_qids = _rank_candidates_for_role(s, predicate, True, context, max_k=5)
    object_qids = _rank_candidates_for_role(o, predicate, False, context, max_k=5)

    return (subject_qids[0] if subject_qids else None,
            object_qids[0] if object_qids else None)

def _capital_counter_evidence(s_qid: str, o_qid: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    evidence: List[Dict[str, str]] = []

    # What capital does the object country/state have?
    rows = _select_sparql(
        f"SELECT ?capLabel WHERE {{ wd:{o_qid} wdt:P36 ?cap . "
        f"SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3"
    )
    for r in rows:
        if r.get("capLabel"):
            evidence.append({
                "predicate": "capital_of",
                "subject": "object",
                "relation": "has_capital",
                "object": r.get("capLabel", "")
            })

    # Optionally: what countries/states is the city capital of?
    rows2 = _select_sparql(
        f"SELECT ?countryLabel WHERE {{ wd:{s_qid} wdt:P1376 ?country . "
        f"SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3"
    )
    for r in rows2:
        if r.get("countryLabel"):
            evidence.append({
                "predicate": "capital_of",
                "subject": "subject",
                "relation": "capital_of",
                "object": r.get("countryLabel", "")
            })

    explanation = None
    if evidence:
        explanation = f"Graph evidence includes capital relations such as {', '.join(e['object'] for e in evidence[:3])}."
    return evidence, explanation


def _founded_by_counter_evidence(s_qid: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    rows = _select_sparql(f"SELECT ?founderLabel WHERE {{ wd:{s_qid} wdt:P112 ?founder . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 5")
    evidence = [{"predicate": "founded_by", "subject": "subject", "relation": "founded_by", "object": r.get("founderLabel", "")} for r in rows]
    explanation = None
    if evidence:
        explanation = f"Refuted: the graph lists founders such as {', '.join(e['object'] for e in evidence[:3])}."
    return evidence, explanation


def _headquarters_counter_evidence(s_qid: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    rows = _select_sparql(f"SELECT ?locLabel WHERE {{ wd:{s_qid} wdt:P159 ?loc . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3")
    evidence = [{"predicate": "headquarters_in", "subject": "subject", "relation": "headquarters_in", "object": r.get("locLabel", "")} for r in rows]
    explanation = f"Refuted: the graph places headquarters in {evidence[0]['object']}." if evidence else None
    return evidence, explanation


def _year_counter_evidence(s_qid: str, pid: str, pred_name: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    rows = _select_sparql(f"SELECT ?dt WHERE {{ wd:{s_qid} p:{pid} ?st . ?st ps:{pid} ?dt . }} LIMIT 3")
    evidence = []
    for r in rows:
        dt = r.get("dt", "")
        m = _looks_like_year(dt)
        if m:
            evidence.append({"predicate": pred_name, "subject": "subject", "relation": pred_name, "object": m})
    explanation = f"Refuted: the graph stores year {evidence[0]['object']}." if evidence else None
    return evidence, explanation


def _location_counter_evidence(s_qid: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    queries = [
        f"SELECT ?locLabel WHERE {{ wd:{s_qid} wdt:P131 ?loc . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3",
        f"SELECT ?locLabel WHERE {{ wd:{s_qid} wdt:P17 ?loc . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3",
        f"SELECT ?locLabel WHERE {{ wd:{s_qid} wdt:P30 ?loc . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3",
        f"SELECT ?locLabel WHERE {{ wd:{s_qid} wdt:P361 ?loc . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3",
        f"SELECT ?locLabel WHERE {{ wd:{s_qid} wdt:P206 ?loc . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3",
        f"SELECT ?locLabel WHERE {{ wd:{s_qid} wdt:P276 ?loc . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 3",
    ]
    evidence: List[Dict[str, str]] = []
    seen = set()
    for q in queries:
        for r in _select_sparql(q):
            obj = r.get("locLabel", "")
            if obj and obj not in seen:
                seen.add(obj)
                evidence.append({"predicate": "located_in", "subject": "subject", "relation": "located_in", "object": obj})
    explanation = f" The graph instead connects the subject to places such as {', '.join(e['object'] for e in evidence[:3])}." if evidence else None
    return evidence, explanation

def _instance_counter_evidence(s_qid: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    rows = _select_sparql(f"SELECT ?clsLabel WHERE {{ wd:{s_qid} wdt:P31 ?cls . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 5")
    evidence = [{"predicate": "instance_of", "subject": "subject", "relation": "instance_of", "object": r.get("clsLabel", "")} for r in rows if r.get("clsLabel")]
    explanation = f"Nearby graph types include {', '.join(e['object'] for e in evidence[:3])}." if evidence else None
    return evidence, explanation

def _invented_by_counter_evidence(item_qid: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    rows = _select_sparql(f"SELECT ?invLabel WHERE {{ wd:{item_qid} wdt:P61 ?inv . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 5")
    evidence = [{"predicate": "invented_by", "subject": "item", "relation": "invented_by", "object": r.get("invLabel", "")} for r in rows if r.get("invLabel")]
    explanation = f"Graph credits inventors such as {', '.join(e['object'] for e in evidence[:3])}." if evidence else None
    return evidence, explanation

def _educated_at_counter_evidence(s_qid: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    rows = _select_sparql(f"SELECT ?schoolLabel WHERE {{ wd:{s_qid} wdt:P69 ?school . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 5")
    evidence = [{"predicate": "educated_at", "subject": "subject", "relation": "educated_at", "object": r.get("schoolLabel", "")} for r in rows if r.get("schoolLabel")]
    explanation = f" Graph links the subject to institutions such as {', '.join(e['object'] for e in evidence[:3])}." if evidence else None
    return evidence, explanation

def _written_by_counter_evidence(work_qid: Optional[str], author_qid: Optional[str] = None) -> Tuple[List[Dict[str, str]], Optional[str]]:
    evidence: List[Dict[str, str]] = []
    if work_qid:
        rows = _select_sparql(f"SELECT ?authorLabel WHERE {{ wd:{work_qid} wdt:P50 ?author . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 5")
        for r in rows:
            if r.get("authorLabel"):
                evidence.append({"predicate": "written_by", "subject": "subject", "relation": "written_by", "object": r.get("authorLabel", "")})
    if not evidence and author_qid:
        rows = _select_sparql(f"SELECT ?workLabel WHERE {{ ?work wdt:P50 wd:{author_qid} . SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }} LIMIT 5")
        for r in rows:
            if r.get("workLabel"):
                evidence.append({"predicate": "written_by", "subject": "object", "relation": "author_of", "object": r.get("workLabel", "")})
    explanation = f"The graph links the work/author neighborhood to {', '.join(e['object'] for e in evidence[:3])}." if evidence else None
    return evidence, explanation
def _norm_text(x: str) -> str:
    x = _normalize_entity_text(x or "")
    return re.sub(r"\s+", " ", x).strip().lower()


def _evidence_contains_claimed_object(
    evidence: List[Dict[str, str]],
    claimed_object: str
) -> bool:
    target = _norm_text(claimed_object)
    if not target:
        return False

    for e in evidence:
        obj = _norm_text(e.get("object", ""))
        if not obj:
            continue

        # exact match
        if obj == target:
            return True

        # soft containment for cases like:
        # "Stanford University" vs "Leland Stanford Junior University"
        if target in obj or obj in target:
            return True

    return False
def _should_emit_refuted(
    predicate: str,
    ask_result: Optional[bool],
    evidence: List[Dict[str, str]],
    claimed_object: str
) -> bool:
    if ask_result is not False:
        return False

    # Critical fix:
    # if graph evidence already contains the claimed object,
    # this is NOT refutation.
    if _evidence_contains_claimed_object(evidence, claimed_object):
        return False

    if predicate in WEAK_REFUTATION_PREDICATES:
        return bool(evidence)

    return True

def verify_triple(triple: Dict[str, Any]) -> Dict[str, Any]:
    t = dict(triple)
    s = _normalize_entity_text(t.get("s", ""))
    p = str(t.get("p", "")).strip().lower()
    o = _normalize_entity_text(t.get("o", ""))
    sentence = str(t.get("sentence", ""))
    negated = bool(t.get("negated", False))

    t.setdefault("graph_evidence", [])
    t.setdefault("explanation", None)

    pid = t.get("property_id") or _get_property_id(p)
    reverse = bool(t.get("reverse", False)) or (p in REVERSE_PREDICATES)
    t["property_id"] = pid
    t["reverse"] = reverse

    if not s or not p or not o or not pid:
        t["verdict"] = "NEI"
        t["ask_result"] = None
        t["s_qid"] = None
        t["o_qid"] = None
        return t

    s_qid, o_qid = _resolve_subject_object_qids(s, o, sentence, p)
    t["s_qid"] = s_qid
    t["o_qid"] = o_qid

    if not s_qid:
        t["verdict"] = "NEI"
        t["ask_result"] = False
        t["graph_evidence"] = [{"type": "entity_linking_failure", "entity": s}]
        t["explanation"] = "Subject could not be linked to the knowledge graph."
        return t

    ask_result: Optional[bool] = None

    if p in {"founded_on", "publication_year", "date_of_birth", "date_of_death"}:
        year = _looks_like_year(o) or _looks_like_year(sentence)
        if year:
            ask_result = _ask_sparql(_build_year_match_ask(s_qid, pid, year))
        elif p == "capital_of":
            if s_qid and o_qid:
                # First try the direct city -> country relation if present in Wikidata
                ask_result = _ask_sparql(f"ASK WHERE {{ wd:{s_qid} wdt:P1376 wd:{o_qid} . }}")

                # Then try the canonical country -> capital relation
                if ask_result is not True:
                    ask_result = _ask_sparql(f"ASK WHERE {{ wd:{o_qid} wdt:P36 wd:{s_qid} . }}")
                    if ask_result is True:
                        t["property_id"] = "P36"
    elif p == "instance_of":
        if o_qid:
            ask_result = _ask_sparql(f"ASK WHERE {{ wd:{s_qid} wdt:P31/wdt:P279* wd:{o_qid} . }}")
            if ask_result is not True:
                ask_result = _ask_sparql(f"ASK WHERE {{ wd:{s_qid} wdt:P31 ?type . ?type wdt:P279* wd:{o_qid} . }}")
    elif p == "located_in":
        if o_qid:
            ask_result = _ask_sparql(_build_located_in_union_ask(s_qid, o_qid))
    elif p == "invented_by":
        if s_qid and o_qid:
            ask_result = _ask_sparql(_build_ask_query(s_qid, pid, o_qid, reverse=False))
            if ask_result is not True:
                ask_result = _ask_sparql(_build_ask_query(o_qid, pid, s_qid, reverse=False))
    else:
        if o_qid:
            ask_result = _ask_sparql(_build_ask_query(s_qid, pid, o_qid, reverse=reverse))

    t["ask_result"] = ask_result
    if ask_result is None:
        ask_result = False
        t["ask_result"] = False

    if ask_result is True:
        t["verdict"] = "REFUTED" if negated else "SUPPORTED"
        return t
    if negated and ask_result is False:
        t["verdict"] = "SUPPORTED"
        return t

    evidence: List[Dict[str, str]] = []
    explanation: Optional[str] = None

    if ask_result is False:
        if p == "capital_of" and s_qid and o_qid:
            evidence, explanation = _capital_counter_evidence(s_qid, o_qid)
        elif p == "founded_by" and s_qid:
            evidence, explanation = _founded_by_counter_evidence(s_qid)
        elif p == "headquarters_in" and s_qid:
            evidence, explanation = _headquarters_counter_evidence(s_qid)
        elif p in {"founded_on", "publication_year", "date_of_birth", "date_of_death"} and s_qid:
            evidence, explanation = _year_counter_evidence(s_qid, pid, p)
        elif p == "located_in" and s_qid:
            evidence, explanation = _location_counter_evidence(s_qid)
        elif p == "instance_of" and s_qid:
            evidence, explanation = _instance_counter_evidence(s_qid)
        elif p == "invented_by":
            item_qid = s_qid if s_qid and _normalize_entity_text(s).lower() == s.lower() else s_qid
            if item_qid or o_qid:
                evidence, explanation = _invented_by_counter_evidence(item_qid or o_qid)
        elif p == "educated_at" and s_qid:
            evidence, explanation = _educated_at_counter_evidence(s_qid)
        elif p == "written_by" and (s_qid or o_qid):
            evidence, explanation = _written_by_counter_evidence(s_qid, o_qid)

    t["graph_evidence"] = evidence
    t["explanation"] = explanation
    if ask_result is False and _evidence_contains_claimed_object(evidence, o):
        t["verdict"] = "REFUTED" if negated else "SUPPORTED"
        if not t["explanation"]:
            t["explanation"] = "Supported: graph evidence contains the claimed object even though the exact ASK query failed."
        return t

    if _should_emit_refuted(p, ask_result, evidence, o):
        t["verdict"] = "REFUTED"
    else:
        t["verdict"] = "NEI"
    return t


def verify_triples(triples: List[Dict[str, Any]], sleep_seconds: float = 0.1) -> List[Dict[str, Any]]:
    out = []
    for t in triples:
        out.append(verify_triple(t))
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return out
