#This file should:
#call the LLM extractor
#call the spaCy extractor
#normalize spaCy output into the LLM-style schema
#merge
#deduplicate
#return one final list

#Before merging, convert each spaCy triple into the same format as the LLM triples.
#A good safe rule is:keep only mapped spaCy triples first
#extract with LLM ,extract with spaCy,keep all good LLM triples,add only spaCy triples that are new and useful
#Why: your LLM extractor already tries to keep only KG-verifiable claims and drops bad predicates. 
#If the same triple appears from both extractors, keep the better one.

from typing import List, Dict, Any, Tuple
import re

from extraction import (
    extract_triples_llm,
    _clean_entity,
    _normalize_predicate,
    _map_predicate_to_wikidata,
    _looks_like_year,
    _is_unverifiable_predicate,
    ALLOWED_PREDICATES,
)

from extraction_spacy import extract_triples_spacy


# Predicates that spaCy may emit but your current verifier does not support directly.
# We drop them in conservative mode to keep the hybrid pipeline clean.
UNSUPPORTED_SPACY_PREDICATES = {
    "is",
    "published_under_pseudonym",
    "related_to",
}

# Very weak raw predicates that usually create noise if merged blindly.
WEAK_RAW_PREDICATES = {
    "be", "have", "has", "had", "do", "make", "say", "go", "come", "use", "be_in"
}

def _canonicalize_place_object(p: str, o: str) -> str:
    """
    For place predicates, prefer the most linkable location span.
    Example:
      'Stratford-upon-Avon, England' -> 'Stratford-upon-Avon'
    """
    p = str(p or "").strip().lower()
    o = _clean_entity(o)

    if p in {"place_of_birth", "place_of_death", "located_at","located_in", "headquarters_in"}:
        if "," in o:
            first = o.split(",")[0].strip()
            if first:
                return first

    return o
def _is_year_like(text: str) -> bool:
    return _looks_like_year(str(text or "")) is not None
def _is_reference_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()

    # author-year citation pattern + quoted title
    if re.search(r"\b[A-Z][A-Za-z\-.]+(?:,\s*[A-Z]\.)*(?:\s*&\s*[A-Z][A-Za-z\-.]+(?:,\s*[A-Z]\.)*)?(?:,\s*et al\.)?\s*\(\d{4}\)", s):
        return True

    # common bibliography signals
    if '"' in s and re.search(r"\(\d{4}\)", s):
        return True

    if "doi:" in s.lower() or "arxiv:" in s.lower():
        return True

    return False


def _looks_like_reference_entity(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.search(r"\bet al\.?\b", t, flags=re.IGNORECASE)) or bool(re.search(r"\(\d{4}\)", t))
def _semantically_valid_triple(s: str, p: str, o: str, sentence: str) -> bool:
    s_l = str(s).strip().lower()
    o_l = str(o).strip().lower()
    sent_l = str(sentence).strip().lower()

    if not s_l or not p or not o_l:
        return False

    # Generic placeholders
    bad_entities = {
        "it", "he", "she", "they", "this", "that", "book", "novel",
        "work", "city", "country", "person", "name", "world"
    }
    if s_l in bad_entities or o_l in bad_entities:
        return False

    # population must be numeric-ish, not descriptive text
    if p == "population_of":
        if not re.search(r"\d", o_l):
            return False
        if any(x in o_l for x in ["km", "kilometer", "mile", "meter", "square"]):
            return False

    # area must be numeric-ish
    if p == "area_of":
        if not re.search(r"\d", o_l):
            return False

    # occupation should be profession-like, not descriptive phrases
    if p == "occupation":
        bad_occ = {
            "cultural hub", "economic hub", "port city", "largest city",
            "second-largest city", "ph.d. student"
        }
        if o_l in bad_occ:
            return False

        # allow short clean profession labels, including negated ones
        allowed_occ = {
            "physicist", "theoretical physicist", "playwright", "poet", "writer",
            "author", "scientist", "mathematician", "chemist", "biologist",
            "philosopher", "engineer", "composer", "artist", "painter",
            "musician", "actor", "director", "inventor", "politician"
        }

        if o_l in allowed_occ:
            return True

        if len(o_l.split()) > 3:
            return False

    # notable_work should usually point from person -> work title, not place/entity fragments
    if p == "notable_work":
        if any(x in o_l for x in ["city", "country", "history", "culture"]):
            return False
        if o_l in {"alexandria", "europe", "world"}:
            return False
        if any(x in o_l for x in ["championship", "championships", "mvp", "award", "awards", "medalist", "medal", "player"]):
            return False

    # founded_by should not have date-like or organization-reverse objects
    if p == "founded_by":
        if _looks_like_year(o_l):
            return False
        if "founded by" not in sent_l and "founded" not in sent_l:
            return False

    # founded_on/date/publication predicates must have year-like objects
    if p in {"founded_on", "publication_year", "date_of_birth", "date_of_death"}:
        if not _looks_like_year(o_l):
            return False

    # capital_of must connect entity-like names, not clause fragments
    if p == "capital_of":
        if any(x in s_l for x in ["capital of", "capital city of"]):
            return False
        if any(x in o_l for x in ["capital of", "capital city of", " is incorrect", " is true", " is false"]):
            return False

    # instance_of should not take descriptive clauses as objects
    if p == "instance_of":
        if len(o_l.split()) > 5 and not any(k in o_l for k in ["programming language", "chemical element"]):
            return False

    # alias_of should not be used for pure symbols/codes like Hg or JPY
    if p == "alias_of":
        if re.fullmatch(r"[A-Z]{1,5}", o.strip()):
            return False

    # part_of should not take long explanatory text
    if p == "part_of":
        if len(o_l.split()) > 4:
            return False
        if any(x in o_l for x in ["history", "result", "reason", "because"]):
            return False

    # located_in / located_at should not take vague/scenic objects
    if p in {"located_in", "located_at", "headquarters_in"}:
        bad_loc = {
            "world", "history", "culture", "civilization"
        }
        if o_l in bad_loc:
            return False
        if any(x in o_l for x in [
            "river", "coast", "shore", "bank", "corner",
            "north", "south", "east", "west",
            "northern", "southern", "eastern", "western"
        ]):
            return False

    return True

def _extract_extra_date_fact(t: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    If a sentence already produced place_of_birth/place_of_death but also contains a year,
    add the corresponding date_of_birth/date_of_death triple.
    """
    s = _clean_entity(t.get("s", ""))
    p = str(t.get("p", "")).strip().lower()
    sentence = str(t.get("sentence", "")).strip()
    negated = bool(t.get("negated", False))

    if not s or not sentence:
        return None

    year = _looks_like_year(sentence)
    if not year:
        m = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|19[0-9]{2}|18[0-9]{2})\b", sentence)
        year = m.group(0) if m else None

    if not year:
        return None

    if p == "place_of_birth":
        new_p = "date_of_birth"
    elif p == "place_of_death":
        new_p = "date_of_death"
    else:
        return None

    pid, reverse = _map_predicate_to_wikidata(new_p)

    return {
        "s": s,
        "p": new_p,
        "o": year,
        "sentence": sentence,
        "confidence": min(float(t.get("confidence", 0.6)), 0.95),
        "negated": negated,
        "property_id": pid,
        "reverse": reverse,
        "source": ["hybrid_rule"],
    }
def _too_long_entity(text: str, max_words: int = 8) -> bool:
    return len(str(text or "").split()) > max_words


def _is_generic_entity(text: str) -> bool:
    t = _clean_entity(text).lower()
    return t in {"it", "he", "she", "they", "this", "that", "book", "novel", "work", "city", "country", "person", "name"}
def _normalize_entity_phrase(text: str) -> str:
    t = _clean_entity(text).strip()
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)

    weak_prefixes = [
        "the practical ",
        "a practical ",
        "an practical ",
        "practical ",
        "the famous ",
        "the well-known ",
        "the renowned ",
        "renowned ",
        "famous ",
        "well-known ",
        "the early ",
        "early ",
        "the modern ",
        "modern ",
    ]

    changed = True
    while changed:
        changed = False
        tl = t.lower()
        for pref in weak_prefixes:
            if tl.startswith(pref):
                t = t[len(pref):].strip()
                changed = True
                break

    return t
def _normalize_llm_triple(t: dict) -> dict | None:
    s = _normalize_entity_phrase(t.get("s", ""))
    p = _normalize_predicate(t.get("p", ""))
    o = _normalize_entity_phrase(t.get("o", ""))
    sentence = str(t.get("sentence", "")).strip()
    negated = bool(t.get("negated", False))
    confidence = float(t.get("confidence", 0.8))
    if _is_reference_sentence(sentence):
        return None

    if _looks_like_reference_entity(s) or _looks_like_reference_entity(o):
        return None
    if p == "related_to":
        return None
    if not s or not p or not o or not sentence:
        return None

    repaired = _repair_birth_death_predicate(p, o, sentence)
    if repaired is None:
        return None
    p, o = repaired
    o = _canonicalize_place_object(p, o)
    if p == "alias_of":
        o_clean = str(o).strip()
        if re.fullmatch(r"[A-Z]{2,6}", o_clean):
            return None
        if len(o_clean) <= 3:
            return None
    
    if p == "publication_year":
        year = _looks_like_year(o) or _looks_like_year(sentence)
        if not year:
            return None
        o = year
    if not _semantically_valid_triple(s, p, o, sentence):
        return None
    pid, reverse = _map_predicate_to_wikidata(p)


    return {
        "s": s,
        "p": p,
        "o": o,
        "sentence": sentence,
        "confidence": confidence,
        "negated": negated,
        "property_id": pid,
        "reverse": reverse,
        "source": ["llm"],
    }
def _repair_birth_death_predicate(p: str, o: str, sentence: str) -> tuple[str, str] | None:
    p = str(p or "").strip().lower()
    o = str(o or "").strip()

    obj_year = _looks_like_year(o)

    if p in {"born_in", "place_of_birth"}:
        if obj_year:
            return ("date_of_birth", obj_year)
        return ("place_of_birth", o)

    if p in {"died_in", "place_of_death"}:
        if obj_year:
            return ("date_of_death", obj_year)
        return ("place_of_death", o)

    if p == "date_of_birth":
        if obj_year:
            return ("date_of_birth", obj_year)
        return None

    if p == "date_of_death":
        if obj_year:
            return ("date_of_death", obj_year)
        return None

    return (p, o)

def _normalize_spacy_predicate(p: str, sentence: str, obj: str) -> str:
    """
    Convert spaCy predicate into a canonical predicate where possible.
    Conservative approach:
    - keep already good predicates
    - map some known patterns
    - otherwise return normalized version, which may later be filtered out
    """
    p = str(p or "").strip().lower().replace(" ", "_")
    sentence_l = str(sentence or "").lower()

    # Keep already canonical predicates
    if p in ALLOWED_PREDICATES:
        return p

    # Conservative repairs
    if p == "written_by":
        return "written_by"

    if p == "author_of":
        return "author_of"

    if p == "alias_of":
        return "alias_of"

    if p == "publication_year":
        return "publication_year"

    if p == "publication_date":
        return "publication_date"

    if p == "place_of_birth":
        return "place_of_birth"

    if p == "place_of_death":
        return "place_of_death"

    if p == "capital_of":
        return "capital_of"

    if p == "located_in":
        return "located_in"

    # If the sentence clearly indicates a year/publication fact, force it
    if "published" in sentence_l and _is_year_like(obj):
        return "publication_year"

    # Keep unsupported special outputs as-is so filter can decide
    return p


def _normalize_spacy_triple(t: Dict[str, Any], default_confidence: float = 0.60) -> Dict[str, Any] | None:
    """
    Convert spaCy triple into the LLM-style schema:
    {
        "s": "...",
        "p": "...",
        "o": "...",
        "sentence": "...",
        "confidence": 0.60,
        "negated": False,
        "property_id": "...",
        "reverse": False,
        "source": ["spacy"]
    }
    """
    s = _normalize_entity_phrase(t.get("s", ""))
    raw_p = str(t.get("p", "")).strip()
    o =_normalize_entity_phrase(t.get("o", ""))
    sentence = str(t.get("sentence", "")).strip()
    negated = bool(t.get("negated", False))
    mapped = bool(t.get("mapped", False))

    if not s or not raw_p or not o or not sentence:
        return None

    p = _normalize_spacy_predicate(raw_p, sentence, o)

    # Publication year repair: object must be a year
    if p == "publication_year":
        year = _looks_like_year(o) or _looks_like_year(sentence)
        if not year:
            return None
        o = year
    if p == "alias_of":
        o_clean = str(o).strip()
        if re.fullmatch(r"[A-Z]{2,6}", o_clean):
            return None
        if len(o_clean) <= 3:
            return None
    # Canonical normalization from extraction.py
    # If p is not in allowed predicates, this may turn it into related_to.
    p = _normalize_predicate(p)
    repaired = _repair_birth_death_predicate(p, o, sentence)
    if repaired is None:
        return None
    p, o = repaired
    o = _canonicalize_place_object(p, o)
    if _is_unverifiable_predicate(p):
        return None
    if not _semantically_valid_triple(s, p, o, sentence):
        return None
    pid, reverse = _map_predicate_to_wikidata(p)

    return {
        "s": s,
        "p": p,
        "o": o,
        "sentence": sentence,
        "confidence": float(default_confidence),
        "negated": negated,
        "property_id": pid,
        "reverse": reverse,
        "mapped": mapped,
        "source": ["spacy"],
    }


def _keep_spacy_triple(t: Dict[str, Any]) -> bool:
    """
    Conservative filtering:
    - keep only mapped or canonical useful triples
    - drop unsupported/noisy predicates
    - drop long/vague entities
    """
    if not t:
        return False

    p = str(t.get("p", "")).strip().lower()
    pid = t.get("property_id")
    mapped = bool(t.get("mapped", False))
    s = str(t.get("s", ""))
    o = str(t.get("o", ""))

    if not p or not s or not o:
        return False

    # Drop raw/noisy unsupported predicates
    if p in UNSUPPORTED_SPACY_PREDICATES:
        return False

    if p in WEAK_RAW_PREDICATES:
        return False

    if p == "related_to":
        return False

    # Avoid very long spans that hurt linking
    if _too_long_entity(s) or _too_long_entity(o):
        return False
    if _is_generic_entity(s) or _is_generic_entity(o):
        return False
    # Drop weak descriptive location triples like:
    # Paris located_in Seine River
    # Alexandria located_at Mediterranean coast
    o_l = str(o).strip().lower()
    sent_l = str(t.get("sentence", "")).strip().lower()

    if p in {"located_in", "located_at"}:
        bad_location_words = {
            "river", "sea", "coast", "shore", "bank", "banks",
            "corner", "part", "north", "south", "east", "west",
            "northern", "southern", "eastern", "western"
        }

        if any(w in o_l for w in bad_location_words):
            return False

        # extra protection for phrases like "along the Seine River"
        if any(k in sent_l for k in ["along the", "on the coast", "by the coast", "near the coast"]):
            return False
       # drop descriptive / explanatory objects
    if len(str(o).split()) > 6:
        return False

    # predicate-specific semantic blocks
    if p == "population_of" and not re.search(r"\d", str(o)):
        return False

    if p == "occupation":
        if str(o).lower() in {"cultural hub", "economic hub", "port city", "ph.d. student"}:
            return False

    if p == "part_of":
        if any(x in str(o).lower() for x in ["history", "result", "reason", "because"]):
            return False

    if p == "notable_work":
        if str(o).lower() in {"alexandria", "world", "history"}:
            return False
    # Keep if it is already canonical + verifiable
    if p in ALLOWED_PREDICATES and (pid is not None):
        return True

    # Keep mapped spaCy triples that normalized into a supported predicate
    if mapped and pid is not None:
        return True

    return False


def _merge_sources(existing: Any, new_source: str) -> List[str]:
    if isinstance(existing, list):
        if new_source not in existing:
            return existing + [new_source]
        return existing
    if isinstance(existing, str):
        return [existing, new_source] if existing != new_source else [existing]
    return [new_source]


def _triple_key(t: Dict[str, Any]) -> Tuple[str, str, str, bool, str, bool]:
    return (
        str(t.get("s", "")).lower(),
        str(t.get("p", "")).lower(),
        str(t.get("o", "")).lower(),
        bool(t.get("negated", False)),
        str(t.get("property_id") or "").lower(),
        bool(t.get("reverse", False)),
    )


def _triple_score(t: Dict[str, Any]) -> Tuple[int, float]:
    """
    Higher is better.
    Prefer:
    1) mapped property_id
    2) higher confidence
    """
    pid_bonus = 1 if t.get("property_id") else 0
    conf = float(t.get("confidence", 0.0) or 0.0)
    return (pid_bonus, conf)


def merge_triples(
    llm_triples: List[Dict[str, Any]],
    spacy_triples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge with LLM-first priority.
    If duplicate keys exist, keep the stronger triple.
    Also merge provenance sources.
    """
    merged: Dict[Tuple[str, str, str, bool, str, bool], Dict[str, Any]] = {}

    # Add LLM triples first
    for t in llm_triples:
        t2 = dict(t)
        if "source" not in t2:
            t2["source"] = ["llm"]
        elif isinstance(t2["source"], str):
            t2["source"] = [t2["source"]]
        key = _triple_key(t2)
        merged[key] = t2

    # Add/compare spaCy triples
    for t in spacy_triples:
        key = _triple_key(t)

        if key not in merged:
            merged[key] = t
            continue

        existing = merged[key]

        # Merge provenance
        existing["source"] = _merge_sources(existing.get("source"), "spacy")

        # Replace only if spaCy version scores higher
        if _triple_score(t) > _triple_score(existing):
            t = dict(t)
            t["source"] = existing["source"]
            merged[key] = t

    return list(merged.values())


 #   Main hybrid extraction entry point.

   # Strategy:
    #- extract LLM triples (primary)
   # - extract spaCy triples (secondary)
    #- normalize/filter spaCy triples
    #- merge and deduplicate
   
def extract_triples_hybrid(
    answer_text: str,
    max_triples_llm: int = 12,
    include_spacy: bool = True,
) -> List[Dict[str, Any]]:
    raw_llm = extract_triples_llm(answer_text, max_triples=max_triples_llm)

    llm_triples: List[Dict[str, Any]] = []
    for t in raw_llm:
        nt = _normalize_llm_triple(t)
        if nt is not None:
            llm_triples.append(nt)

    raw_spacy = []
    normalized_spacy: List[Dict[str, Any]] = []

    if include_spacy:
        raw_spacy = extract_triples_spacy(answer_text)

        for t in raw_spacy:
            nt = _normalize_spacy_triple(t)
            if nt and _keep_spacy_triple(nt):
                normalized_spacy.append(nt)

    # First merge LLM + spaCy
    merged = merge_triples(llm_triples, normalized_spacy)

    # Add extra rule-based triples for mixed birth/death sentences
    extra_triples: List[Dict[str, Any]] = []
    for t in merged:
        extra = _extract_extra_date_fact(t)
        if extra is not None:
            extra_triples.append(extra)

    # Merge again so duplicates are removed safely
    return merge_triples(merged, extra_triples)