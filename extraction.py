import os
import json
import re
from typing import List, Dict, Any, Tuple

from predicate_schema import (
    ALLOWED_PREDICATES,
    canonicalize_predicate,
    canonicalize_predicate_with_metadata,
    is_unverifiable_predicate as schema_is_unverifiable_predicate,
    map_predicate_to_wikidata as schema_map_predicate_to_wikidata,
    maybe_remap_predicate_by_object as schema_maybe_remap_predicate_by_object,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXTRACTION_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

#Knowledge graphs use fixed predicates.So extraction must align with KG structure.
#so here i used an aproach of giving my llm a list to choose which is the closest to the predicate
#in following code i will make the predicate related_to if it doesnot exist in this allowed predicate list
#i will change this option to Map this predicate to the closest Wikidata property ID.
ALLOWED_PREDICATES = list(ALLOWED_PREDICATES)

#defines a function to safely parse json (crashes if json format is wrong)
#Makes the pipeline robust, prevents your experiment runs from breaking.
def _safe_json_load(s: str) -> Any:
    """Try to load JSON safely. Returns None if invalid."""
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------------------------
# NEW helpers to fix issues:
# 1) Better robustness when model returns extra text around JSON.
# 2) Clean entities/punctuation/quotes so linking works better.
# 3) Drop unverifiable fluffy predicates early (thesis-worthy extraction).
# 4) Ensure publication_year object is a YEAR not a PERSON.
# 5) Enforce direction consistency (optional, heuristics-based).
# 6) ✅ FIX MAPPING ISSUE: Map canonical predicates to Wikidata property IDs + direction.
# ---------------------------

# ✅ NEW: canonical predicate -> (Wikidata property_id, reverse_flag)
# reverse_flag=True means: to verify (s,p,o) we should ask (o PID s) in Wikidata.
# Example: your schema uses (city, capital_of, country) but Wikidata stores (country, P36, city).
PREDICATE_TO_WIKIDATA: Dict[str, Tuple[str, bool]] = {
    # geography / containment
    "capital_of": ("P1376", False),          # Wikidata: (country P36 capitalCity)
    "located_in": ("P131", False),        # located in admin entity (often needs path queries)
    "part_of": ("P361", False),           # part of

    # types
    "instance_of": ("P31", False),        # instance of

    # people facts (place vs date)
    "born_in": ("P19", False),            # place of birth
    "died_in": ("P20", False),            # place of death
    "place_of_birth": ("P19", False),
    "place_of_death": ("P20", False),
    "date_of_birth": ("P569", False),
    "date_of_death": ("P570", False),

    # org / roles
    "founded_by": ("P112", False),        # founded by / founder (org -> person)
    "founder_of": ("P112", True),         # (person founder_of org) == reverse of P112
    "ceo_of": ("P169", False),            # Current extracted schema is usually (organization, ceo_of, person)
    "president_of": ("P35", True),        # head of state (country P35 person). Your schema: (person president_of country)
    "headquarters_in": ("P159", False),   # headquarters location

    # works / authorship
    "written_by": ("P50", False),         # author (work -> person)
    "author_of": ("P50", True),           # (person author_of work) reverse of P50
    "notable_work": ("P800", False),      # notable work (person -> work)

    # inventions
    "invented_by": ("P61", False),        # discoverer or inventor (item -> person)

    # country properties (stored on country in Wikidata)
    "currency_of": ("P38", False),        # Current extracted schema is usually (country, currency_of, currency)
    "language_of": ("P37", False),        # Current extracted schema is usually (country, language_of, language)
    "country_of_origin": ("P495", False), # country of origin

    # misc
    "genre": ("P136", False),             # genre
    "spouse": ("P26", False),             # spouse

    # NOTE: child_of is tricky on Wikidata because parents are P22/P25 and children are P40.
    # We'll verify "child_of" using parent->child (P40) in reverse direction:
    # (child child_of parent) becomes ASK(parent P40 child)
    "child_of": ("P40", True),

    "educated_at": ("P69", False),
    "occupation": ("P106", False),

    # quantitative
    "population_of": ("P1082", False),    # population
    "area_of": ("P2046", False),          # area

    # generic location (used for buildings/landmarks)
    "located_at": ("P276", False),        # location

    # time for works
    "publication_date": ("P577", False),  # publication date
    "publication_year": ("P577", False),  # publication date (we verify YEAR(P577) == YYYY)
    "alias_of": ("P4970", False),        # alt label / spelling variant
    "founded_on": ("P571", False),
}

def _map_predicate_to_wikidata(p: str) -> Tuple[str | None, bool]:
    return schema_map_predicate_to_wikidata(p)

def _extract_json_array_from_text(text: str) -> Any:
    """
    NEW: In case the model returns extra text, try to recover the first JSON array substring.
    This prevents your batch runs from dying just because the model added one extra character.
    """
    if not text:
        return None

    # quick success case
    data = _safe_json_load(text)
    if isinstance(data, list):
        return data

    # try to find a JSON array in the string
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        data2 = _safe_json_load(snippet)
        if isinstance(data2, list):
            return data2

    return None

def _clean_entity(text: str) -> str:
    t = str(text or "").strip()

    # remove markdown / quotes
    t = re.sub(r"^\*+(.*?)\*+$", r"\1", t)
    t = t.strip().strip("“”\"'`")

    # remove articles
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)

    # 🔥 remove instruction tails (CRITICAL FOR THESIS)
    t = re.split(
        r"\b(?:verify|check|analyze|validate|confirm|assess)\b.*",
        t,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    # remove contrast clauses
    t = re.split(r"\s*,\s*not\s+", t, maxsplit=1, flags=re.IGNORECASE)[0]
    t = re.split(r"\s+but\s+not\s+", t, maxsplit=1, flags=re.IGNORECASE)[0]

    # remove punctuation
    t = t.rstrip(".,;:!?)\"”’`")
    t = re.sub(r"\s+", " ", t).strip()

    return t

def _looks_like_clause_fragment(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return True
    if re.search(r"\b(?:not|but|however|although|rather than|instead of|which is|that is|because)\b", t):
        return True
    if len(t.split()) > 9:
        return True
    return False

def _triple_relevant_to_prompt(s: str, o: str, sentence: str, prompt_text: str | None) -> bool:
    if not prompt_text:
        return True
    prompt_l = str(prompt_text or "").lower()
    for piece in (s, o, sentence):
        piece_l = str(piece or "").lower()
        if piece_l and piece_l in prompt_l:
            return True
    prompt_tokens = {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", prompt_l) if len(w) >= 4}
    claim_tokens = {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", f"{s} {o} {sentence}".lower()) if len(w) >= 4}
    return bool(prompt_tokens & claim_tokens)

def _looks_like_year(text: str) -> str | None:
    """
    NEW: detect a 4-digit year literal.
    Returns the year string if found.
    """
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|19[0-9]{2}|18[0-9]{2})\b", str(text or ""))
    return m.group(0) if m else None

def _is_unverifiable_predicate(p: str) -> bool:
    return schema_is_unverifiable_predicate(p)

NON_HUMAN_INSTANCE_OBJECTS = {
    "planet", "chemical element", "programming language", "river", "car brand",
    "technology company", "fruit", "mountain", "city", "country", "tower",
    "landmark", "novella", "novel", "book", "play", "company", "corporation",
    "software", "species", "big cat", "wild cat", "element", "university",
    "social networking site", "social media platform", "e-commerce platform",
    "electric vehicle manufacturer", "electric car manufacturer"
}

def _maybe_remap_predicate_by_object(p: str, o: str) -> str:
    return schema_maybe_remap_predicate_by_object(p, o)

NON_HUMAN_INSTANCE_OBJECTS = {
    "planet", "chemical element", "programming language", "river", "car brand",
    "technology company", "fruit", "mountain", "city", "country", "tower",
    "landmark", "novella", "novel", "book", "play", "company", "corporation",
    "software", "species", "big cat", "wild cat", "element"
}


def _maybe_remap_predicate_by_object(p: str, o: str) -> str:
    return schema_maybe_remap_predicate_by_object(p, o)


def _normalize_predicate(p: str) -> str:
    return canonicalize_predicate(p)

def _is_generic_entity_phrase(text: str) -> bool:
    """
    Drop vague placeholder entities that are not linkable on their own.
    These should be resolved upstream, not verified as standalone entities.
    """
    t = _clean_entity(text).lower()
    generic = {
        "it", "he", "she", "they", "this", "that", "these", "those",
        "book", "the book", "novel", "the novel", "work", "the work",
        "city", "country", "person", "name"
    }
    return t in generic


def _looks_like_role_tenure_sentence(sentence: str) -> bool:
    s = sentence.lower()
    return any(k in s for k in [
        "since ", "has held", "has been in the role", "in the role",
        "succeeding", "chosen as", "chosen to", "appointed", "named", "became"
    ])


def _repair_or_drop_time_role_misextraction(s: str, p: str, o: str, sentence: str) -> tuple[str, str] | None:
    """
    Prevent common bad extractions like:
    - Tim Cook date_of_birth 2011
    - Canberra founded_by 1908
    - Apple publication_year 2011
    """
    year = _looks_like_year(o)
    sent_l = sentence.lower()

    if _is_generic_entity_phrase(s) or _is_generic_entity_phrase(o):
        return None

    if p in {"date_of_birth", "date_of_death"}:
        if not any(k in sent_l for k in ["born", "died"]):
            return None

    if p in {"publication_year", "publication_date"}:
        if "published" not in sent_l and "released" not in sent_l and "founded" not in sent_l:
            return None

    if p == "founded_by" and year:
        if not any(k in sent_l for k in ["founded", "founded on", "founded in"]):
            return None

    if p == "ceo_of" and year:
        return None

    if _looks_like_role_tenure_sentence(sentence) and p in {"date_of_birth", "date_of_death", "publication_year", "publication_date", "founded_on"}:
        return None

    return (p, o)


def _direction_fix(triple: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conservative direction repair.
    Only fix cases that are very likely reversed.
    Do NOT aggressively rewrite working triples.
    """
    t = dict(triple)
    s = _clean_entity(t.get("s", ""))
    p = str(t.get("p", "")).strip().lower()
    o = _clean_entity(t.get("o", ""))
    sent = str(t.get("sentence", "")).strip().lower()

    if not s or not p or not o:
        return t

    def swap() -> Dict[str, Any]:
        t["s"], t["o"] = o, s
        return t

    s_l = s.lower()
    o_l = o.lower()

    # 1) capital_of:
    # keep canonical direction as (city, capital_of, country)
    # fix only when subject looks like country and object looks like city,
    # or the sentence explicitly says "capital of X is Y"
    country_like = {
        "country", "kingdom", "republic", "state", "nation", "federation", "empire"
    }
    city_like = {
        "city", "capital", "town", "municipality", "metropolis"
    }

    if p == "capital_of":
        if "capital of" in sent and " is " in sent:
            # "The capital of Jordan is Amman" often becomes Jordan capital_of Amman
            # canonicalize to Amman capital_of Jordan
            if s_l in sent and o_l in sent:
                # if the subject appears after "capital of" and object appears after "is", swap
                m = re.search(r"capital(?: city)? of\s+(.+?)\s+is\s+(.+)", sent)
                if m:
                    left = _clean_entity(m.group(1)).lower()
                    right = _clean_entity(m.group(2)).lower()
                    if s_l == left and o_l.startswith(right):
                        return swap()

    # 2) written_by / author_of
    # canonical forms:
    # written_by: (work, written_by, person)
    # author_of:  (person, author_of, work)
    if p == "written_by":
        # if subject looks like person and object looks like title/work, swap
        if any(k in s_l for k in ["mr ", "mrs ", "dr ", "prof "]) and len(o.split()) <= 8:
            return swap()

    if p == "author_of":
        # if subject looks like a work title and object looks like a person, swap
        if (
            ("book" in s_l or "novel" in s_l or "poem" in s_l or "play" in s_l)
            and len(o.split()) <= 4
        ):
            return swap()

    # 3) invented_by
    # canonical form: (item, invented_by, person)
    if p == "invented_by":
        # if subject looks like person and object looks like an item/device, swap
        item_hints = {"telephone", "light bulb", "lamp", "radio", "airplane", "computer", "internet"}
        if o_l in item_hints and len(s.split()) <= 4:
            return swap()

    # 4) ceo_of and language_of
    # canonical forms:
    # ceo_of: (organization, ceo_of, person)
    # language_of: (country/region, language_of, language)
    if p == "ceo_of":
        if any(k in sent for k in ["ceo of", "chief executive officer of", "chief executive of"]):
            if len(s.split()) <= 4 and any(x in o_l for x in ["inc", "corp", "company", "group", "university", "organization", "ltd", "llc"]):
                return swap()

    if p == "language_of":
        if any(x in s_l for x in ["english", "arabic", "french", "german", "spanish", "japanese", "chinese", "urdu", "hindi"]):
            return swap()

    # 5) founder_of / founded_by
    # canonical forms:
    # founded_by: (organization, founded_by, person)
    # founder_of: (person, founder_of, organization)
    if p == "founded_by":
        if _looks_like_year(o):
            return t  # never swap year mistakes here
        # if subject is clearly a person and object looks like an org, swap
        if any(x in o_l for x in ["inc", "corp", "company", "university", "organization"]):
            return swap()

    if p == "founder_of":
        # if subject looks like org and object looks like person, swap
        if any(x in s_l for x in ["inc", "corp", "company", "university", "organization"]):
            return swap()

    return t


# takes generated answer and a limit number of triples to be produced
#and outputs a list of dictionaries representing triplets
def extract_triples_llm(answer_text: str, max_triples: int = 12, prompt_text: str | None = None) -> List[Dict[str, Any]]:
    """
    Extract factual triples from an LLM-generated answer using an LLM and return:
      [{"s": "...", "p": "...", "o": "...", "sentence": "...", "confidence": 0.xx}, ...]
    """

    system_prompt = (
        "You are an information extraction system.\n"
        "Your job is to extract factual claims from text as triples.\n"
        "Return ONLY valid JSON. No markdown, no extra text."
    )

    # NEW: make the LLM extraction thesis-worthy:
    # - enforce KG-verifiable claims only
    # - encourage multi-fact extraction from one sentence
    # - enforce year predicates to have year objects
    # - enforce short clean entities
    user_prompt = {
        "task": "Extract factual triples from the provided answer.",
        "rules": [
            "Output must be a JSON array of objects.",
            "Each object must have keys: s, p, o, sentence, confidence, negated",
            "s, p, o, sentence are strings. confidence is a number between 0 and 1. negated is boolean.",
            "Only include factual, checkable claims (not opinions, not advice).",
            "IMPORTANT: output ONLY knowledge-graph-verifiable claims (avoid 'known for', 'considered', 'inspired', emotions, vague statements).",
            "If a sentence contains multiple facts, extract multiple triples.",
            "Prefer canonical predicates from the allowed list. If none fits exactly, omit the claim.",
            f"Allowed predicates: {ALLOWED_PREDICATES}",
            "Do NOT use predicate 'related_to'. Drop the claim instead.",
            f"Return at most {max_triples} triples.",
            "Keep entities short (e.g., 'Paris', 'Germany', 'Wuthering Heights'). No long clauses.",
            "Do NOT use vague subjects like it, book, novel, work, city, country, person, or name unless a specific entity is explicit.",
            "If prompt_text is given, focus on the entities and propositions the user asked to verify. Do not add side facts that are merely extra background.",
            "Do not output clause fragments such as France, not Germany or Riyadh, not Cairo as entities.",
            "For spelling variants like Shakspere/Shakespeare, use predicate alias_of.",
            "Do NOT turn role-tenure years into birth/publication facts. For example, never extract Tim Cook date_of_birth 2011 from CEO tenure text.",
            "Do NOT use founded_by with a year unless the sentence explicitly says founded on/in that year.",
            "sentence must be the exact sentence (or minimal clause) that expressed the claim.",
            "If p is publication_year, o MUST be a 4-digit year like '1847'.",
            "If p is date_of_birth or date_of_death, o should be a date or year text (not a person).",
            "For human professions like physicist, poet, playwright, writer, use predicate 'occupation', not 'instance_of'.",
            "For non-human types such as planet, chemical element, programming language, river, car brand, fruit, and technology company, use predicate 'instance_of', not 'occupation'.",
            "Use 'alias_of' only for true name or spelling variants of the same entity.",
"Do NOT use 'alias_of' for abbreviations, symbols, acronyms, or codes such as JPY, USD, Hg, or ¥.",
        ],
        "answer_text": answer_text,
        "prompt_text": prompt_text or "",
    }

    # user_prompt convert the dictionary(user_prompt) into a JSON string before sending.
    resp = client.responses.create(
        model=EXTRACTION_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)}
        ],
        temperature=TEMPERATURE,
    )

    # Extract text output ,raw should contain a JSON array string.
    #we only care about output text and concatenate it
    raw = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    raw += c.text

    # NEW: recover JSON array even if model returns extra text
    data = _extract_json_array_from_text(raw)

    # If the model didn't return valid JSON, fail gracefully by returning an empty list:
    if not isinstance(data, list):
        return []

    #Create cleaned list for valid triples.If some entry isn’t a dict, skip it.
    cleaned: List[Dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        
        #Convert to string and strip whitespace.
        s = _clean_entity(obj.get("s", ""))
        p_raw = str(obj.get("p", "")).strip()
        meta = canonicalize_predicate_with_metadata(p_raw, sentence=str(obj.get("sentence", "")), s=s, o=obj.get("o", ""))
        p = meta["p_canonical"]
        o = _clean_entity(obj.get("o", ""))
        p = _maybe_remap_predicate_by_object(p, o)
        sentence = str(obj.get("sentence", "")).strip()
        sentence_l = sentence.lower()
          # Drop bibliography/reference lines from factual claim extraction
        if re.search(r"\(\d{4}\)", sentence) and ('"' in sentence or "doi:" in sentence.lower() or "arxiv:" in sentence.lower()):
            continue

        if re.search(r"\bet al\.?\b", s, flags=re.IGNORECASE):
            continue
        # Drop descriptive/non-KG location phrases
        if p in {"located_in", "located_at"}:
            o_l = o.lower()
            if any(w in o_l for w in ["river", "sea", "coast", "shore", "bank", "banks",
                                    "corner", "northern", "southern", "eastern", "western"]):
                continue
            if any(k in sentence_l for k in ["along the", "on the coast", "by the coast", "near the coast"]):
                continue
        if p in {"publication_year", "publication_date"} and "founded" in sentence_l:
            if _looks_like_role_tenure_sentence(sentence):
                continue
            if "founded by" not in sentence_l and "was founded" not in sentence_l and "founded on" not in sentence_l:
                continue

            if p == "publication_year":
                year = _looks_like_year(o) or _looks_like_year(sentence)
                if not year:
                    continue
                o = year

            p = "founded_on"

        conf = obj.get("confidence", None)
        negated = obj.get("negated", False)
        negated = bool(negated)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5

        # Basic validation,skip if incomplete triples
        if not s or not p or not o or not sentence:
            continue
        if _looks_like_clause_fragment(s) or _looks_like_clause_fragment(o):
            continue
        if prompt_text and not _triple_relevant_to_prompt(s, o, sentence, prompt_text):
            continue
        if conf < 0:
            conf = 0.0
        if conf > 1:
            conf = 1.0

        # NEW: hard filter for obviously unverifiable predicates (reduces NEI noise)
        if _is_unverifiable_predicate(p_raw) or _is_unverifiable_predicate(p):
            continue

        # NEW: fix common extraction failure:
        # if predicate is publication_year, object must be a YEAR (not a person like Emily Brontë)
        if p == "publication_year":
            year = _looks_like_year(sentence) or _looks_like_year(o)
            if year:
                o = year
            else:
                # if we can't find a year, drop this triple (better than wrong triple)
                continue

        repaired = _repair_or_drop_time_role_misextraction(s, p, o, sentence)
        if repaired is None:
            continue
        p, o = repaired
        p = _maybe_remap_predicate_by_object(p, o)

        # NEW: optionally enforce some direction/format consistency
        t = {
            "s": s,
            "p": p,
            "o": o,
            "sentence": sentence,
            "confidence": conf,
            "negated": negated
        }
        t = _direction_fix(t)

        # ✅ FIX MAPPING ISSUE: attach Wikidata property_id + reverse direction at extraction time.
        # This makes verification consistent and prevents "REFUTED" due to wrong direction.
        # If we cannot map predicate -> property_id, mark as related_to (and set property_id None).
        pid = meta.get("property_id")
        reverse = meta.get("reverse", False)
        if not pid and t["p"]:
            pid, reverse = _map_predicate_to_wikidata(t["p"])
        if t["p"] == "related_to":
            pid, reverse = (None, False)

        # If predicate is not mapped, you can either:
        # 1) keep it as related_to with property_id=None (thesis: show as unverifiable), or
        # 2) drop it to reduce noise.
        # Here we keep it BUT clearly mark it as unmapped.
        t["property_id"] = pid
        t["reverse"] = reverse
        t["p_raw"] = meta.get("p_raw", p_raw)
        t["verification_strategy"] = meta.get("strategy", "direct")

        #Add validated triple to output list.
        cleaned.append(t)

    # NEW: light dedup (LLMs sometimes repeat)
    seen = set()
    out: List[Dict[str, Any]] = []
    for t in cleaned:
        key = (
            t["s"].lower(),
            t["p"].lower(),
            t["o"].lower(),
            bool(t.get("negated", False)),
            str(t.get("property_id") or "").lower(),
            bool(t.get("reverse", False))
        )
        if key not in seen:
            seen.add(key)
            out.append(t)

    return out