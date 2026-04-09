import re
from typing import List, Dict, Any, Tuple

import spacy

nlp = spacy.load("en_core_web_sm")

# Conservative spaCy extractor:
# - only emits high precision triples
# - strongly filters clause fragments / meta-discourse
# - avoids mapping every sentence containing "capital" to capital_of
# - prefers fewer, cleaner triples over recall

HUMAN_OCCUPATIONS = {
    "physicist", "theoretical physicist", "playwright", "poet", "writer",
    "author", "scientist", "mathematician", "chemist", "biologist",
    "philosopher", "engineer", "composer", "artist", "painter",
    "musician", "actor", "director", "inventor", "politician",
    "basketball player", "footballer", "businessman", "entrepreneur"
}

GENERIC_ENTITIES = {
    "it", "he", "she", "they", "this", "that", "these", "those",
    "book", "novel", "work", "city", "country", "person", "name",
    "statement", "claim", "conclusion", "analysis", "fact", "details",
    "information", "part", "summary", "implications", "context", "meaning",
    "capital", "capital city", "capital of", "history", "culture", "time"
}

DISCOURSE_CUES = {
    "in conclusion", "to summarize", "summary", "analysis", "implications",
    "therefore", "overall", "this part", "that part", "the statement",
    "the claim", "correct information", "verification"
}

BAD_FRAGMENT_PATTERNS = [
    re.compile(r"\b(is|are|was|were|be|been|being)\b", re.I),
    re.compile(r"\b(and|or|but)\b", re.I),
    re.compile(r"^[-•*]\s*"),
]


def _span_text(tok) -> str:
    span = tok.doc[tok.left_edge.i: tok.right_edge.i + 1]
    return span.text.strip()


def _clean_entity(text: str) -> str:
    t = str(text or "").strip()
    t = re.sub(r"^[-•*]+\s*", "", t)
    t = t.strip("“”\"'`")
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.rstrip(".,;:!?)")
    return t


def _extract_year(text: str) -> str | None:
    m = re.search(r"\b(18|19|20)\d{2}\b", str(text or ""))
    return m.group(0) if m else None


def _is_negated(token) -> bool:
    return any(child.dep_ == "neg" for child in token.children)


def _looks_like_role_tenure_sentence(text: str) -> bool:
    s = (text or "").lower()
    return any(k in s for k in [
        "since ", "has held", "has been in the role", "in the role",
        "succeeding", "appointed", "named", "became", "served as"
    ])


def _is_generic_entity(text: str) -> bool:
    t = _clean_entity(text).lower()
    return t in GENERIC_ENTITIES


def _is_bad_entity_fragment(text: str) -> bool:
    t = _clean_entity(text)
    if not t:
        return True
    tl = t.lower()
    if tl in GENERIC_ENTITIES:
        return True
    if len(t.split()) > 6:
        return True
    if tl.startswith(("capital of ", "capital city of ", "statement", "claim", "conclusion")):
        return True
    if tl.endswith((" is incorrect", " is true", " is false")):
        return True
    if t in {"Inc", "Ltd", "Corp", "Co"}:
        return True
    if ":" in t or "\n" in t:
        return True
    for pat in BAD_FRAGMENT_PATTERNS:
        if pat.search(t) and len(t.split()) > 2:
            return True
    return False


def _extract_quoted(text: str) -> str | None:
    m = re.search(r"[\"“”](.+?)[\"“”]", text)
    return _clean_entity(m.group(1)) if m else None


def _normalize_relation(sent_text: str, verb_lemma: str, prep: str | None) -> Tuple[str, bool]:
    s = (sent_text or "").lower()

    if any(cue in s for cue in DISCOURSE_CUES):
        return ("", False)

    if any(k in s for k in ["alternative spelling", "alternate spelling", "variant spelling", "alternative form", "spelling of"]):
        return ("alias_of", True)

    # Strict capital detection only for explicit constructions.
    if re.search(r"\b[A-Z][^.!?]{0,120}?\bis the capital of\b", sent_text):
        return ("capital_of", True)
    if re.search(r"\bthe capital (?:city )?of\b", s):
        return ("capital_of", True)

    if ("published" in s or "released" in s) and _extract_year(sent_text):
        return ("publication_year", True)

    if verb_lemma in ("bear", "be") and "born" in s and prep == "in":
        return ("place_of_birth", True)
    if verb_lemma == "die" and "died" in s and prep == "in":
        return ("place_of_death", True)

    if prep in ("in", "at", "within", "inside"):
        if "headquartered" in s or "based" in s or "located" in s or "situated" in s:
            return ("located_in", True)

    if verb_lemma in ("be", "become"):
        for job in HUMAN_OCCUPATIONS:
            if job in s:
                return ("occupation", True)

    return ("", False)


def _find_subject(token):
    for child in token.children:
        if child.dep_ in {"nsubj", "nsubjpass", "attr"}:
            return child
    return None


def _find_object(token):
    # direct object / attribute
    for child in token.children:
        if child.dep_ in {"dobj", "attr", "oprd", "pobj"}:
            return child
    # prepositional object
    for child in token.children:
        if child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    return gc
    return None


def _extract_capital_patterns(sent_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # X is the capital of Y
    m1 = re.search(r"\b([A-Z][A-Za-z0-9 .,'’\-]+?)\s+is\s+the\s+capital(?:\s+city)?\s+of\s+([A-Z][A-Za-z0-9 .,'’\-]+)\b", sent_text)
    if m1:
        s = _clean_entity(m1.group(1))
        o = _clean_entity(m1.group(2))
        if not _is_bad_entity_fragment(s) and not _is_bad_entity_fragment(o):
            out.append({"s": s, "p": "capital_of", "o": o, "sentence": sent_text.strip(), "mapped": True, "negated": False})

    # The capital of Y is X
    m2 = re.search(r"\bthe\s+capital(?:\s+city)?\s+of\s+([A-Z][A-Za-z0-9 .,'’\-]+?)\s+is\s+([A-Z][A-Za-z0-9 .,'’\-]+)\b", sent_text, flags=re.I)
    if m2:
        country = _clean_entity(m2.group(1))
        city = _clean_entity(m2.group(2))
        if not _is_bad_entity_fragment(city) and not _is_bad_entity_fragment(country):
            out.append({"s": city, "p": "capital_of", "o": country, "sentence": sent_text.strip(), "mapped": True, "negated": False})

    return out


def _extract_location_patterns(sent_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # X is in Y / located in Y / based in Y / headquartered in Y
    patterns = [
        r"\b([A-Z][A-Za-z0-9 .,'’\-]+?)\s+is\s+in\s+([A-Z][A-Za-z0-9 .,'’\-]+)\b",
        r"\b([A-Z][A-Za-z0-9 .,'’\-]+?)\s+is\s+located\s+in\s+([A-Z][A-Za-z0-9 .,'’\-]+)\b",
        r"\b([A-Z][A-Za-z0-9 .,'’\-]+?)\s+is\s+based\s+in\s+([A-Z][A-Za-z0-9 .,'’\-]+)\b",
        r"\b([A-Z][A-Za-z0-9 .,'’\-]+?)\s+is\s+headquartered\s+in\s+([A-Z][A-Za-z0-9 .,'’\-]+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, sent_text)
        if m:
            s = _clean_entity(m.group(1))
            o = _clean_entity(m.group(2))
            if _is_bad_entity_fragment(s) or _is_bad_entity_fragment(o):
                continue
            pred = "headquarters_in" if "headquartered" in pat else "located_in"
            out.append({"s": s, "p": pred, "o": o, "sentence": sent_text.strip(), "mapped": True, "negated": False})
    return out


def extract_triples_spacy(text: str) -> List[Dict[str, Any]]:
    doc = nlp(text)
    triples: List[Dict[str, Any]] = []
    seen = set()

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        s_lower = sent_text.lower()
        if any(cue in s_lower for cue in DISCOURSE_CUES):
            continue
        if _looks_like_role_tenure_sentence(sent_text):
            # skip noisy tenure/explanatory sentences for spaCy extractor
            continue

        # High precision regex patterns first.
        candidates = []
        candidates.extend(_extract_capital_patterns(sent_text))
        candidates.extend(_extract_location_patterns(sent_text))

        # Small dependency fallback for occupation only.
        if not candidates:
            for token in sent:
                pred, mapped = _normalize_relation(sent_text, token.lemma_.lower(), None)
                if pred != "occupation" or not mapped:
                    continue
                subj_tok = _find_subject(token)
                if not subj_tok:
                    continue
                subj = _clean_entity(_span_text(subj_tok))
                occ = None
                for child in token.children:
                    if child.dep_ in {"attr", "acomp", "dobj"}:
                        occ = _clean_entity(_span_text(child))
                        break
                if not occ:
                    continue
                if _is_bad_entity_fragment(subj) or _is_bad_entity_fragment(occ):
                    continue
                candidates.append({
                    "s": subj,
                    "p": "occupation",
                    "o": occ,
                    "sentence": sent_text,
                    "mapped": True,
                    "negated": _is_negated(token),
                })

        for t in candidates:
            s = _clean_entity(t["s"])
            p = t["p"]
            o = _clean_entity(t["o"])
            if _is_bad_entity_fragment(s) or _is_bad_entity_fragment(o):
                continue
            if _extract_year(o) and p in {"capital_of", "located_in", "occupation"}:
                continue
            key = (s.lower(), p.lower(), o.lower(), sent_text.lower())
            if key in seen:
                continue
            seen.add(key)
            triples.append({
                "s": s,
                "p": p,
                "o": o,
                "sentence": sent_text,
                "mapped": bool(t.get("mapped", False)),
                "negated": bool(t.get("negated", False)),
            })

    return triples
