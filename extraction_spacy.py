#extract triples using spaCy dependency parsing, then applies simple rule-based normalization so predicates become more KG-friendly
import spacy
import re
from typing import List, Dict, Any, Tuple

nlp = spacy.load("en_core_web_sm")


#tok might be only one word,But usually the real entity/phrase is larger,So this function expands a token to include related words.
#If tok is “capital” in “the capital of Turkey”,subtree span becomes “the capital of Turkey”
def _span_text(tok) -> str:
    """Expand token to its subtree span text (rough phrase extraction)."""
    span = tok.doc[tok.left_edge.i : tok.right_edge.i + 1]
    return span.text.strip()

#to detect negation
def _is_negated(token) -> bool:
    # checks if token has a child with dep_ == "neg" (e.g., "not", "n't")
    return any(child.dep_ == "neg" for child in token.children)

#helper to detect if a subject is a pronoun like "It" (usually not useful for KG verification)
def _is_pronoun(tok) -> bool:
    return tok.pos_ == "PRON"
#helper: detect vague subject phrases like "the novel", "this book", "the work",the,he, even if span is longer
def _is_vague_subject(text: str) -> bool:
    t = text.lower().strip()
    # common vague heads that should resolve to last_subject
    vague_heads = ("the novel", "this novel", "novel", "the book", "this book", "book", "the work", "this work", "work", "it", "he", "she", "they", "this", "that", "city", "country", "person", "name")
    return any(t.startswith(v) for v in vague_heads)
#helper: extract a 4-digit year from text (useful for "published in 1847" etc.)
def _extract_year(text: str) -> str | None:
    m = re.search(r"\b(18|19|20)\d{2}\b", text)
    return m.group(0) if m else None

#helper: clean quotes / trailing punctuation from extracted entity spans
def _clean_entity(text: str) -> str:
    # remove surrounding quotes and trailing punctuation artifacts
    t = text.strip()
    t = t.strip("“”\"'`")
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r'[\"“”]+$', "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    t = t.rstrip(".,;:!?)")
    return t

#helper: for phrases like "author of Wuthering Heights" extract the object after "of"
def _extract_of_object(noun_token) -> str | None:
    for child in noun_token.children:
        if child.dep_ == "prep" and child.lemma_.lower() == "of":
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    return _clean_entity(_span_text(pobj))
    return None

#helper: if sentence contains quoted string, prefer that as entity (useful for titles and variants like "Shakspere")
def _extract_quoted(text: str) -> str | None:
    m = re.search(r"[\"“”](.+?)[\"“”]", text)
    return _clean_entity(m.group(1)) if m else None


def _is_generic_entity(text: str) -> bool:
    t = _clean_entity(text).lower()
    return t in {"it", "he", "she", "they", "this", "that", "book", "novel", "work", "city", "country", "person", "name"}


def _looks_like_role_tenure_sentence(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in ["since ", "has held", "has been in the role", "in the role", "succeeding", "chosen as", "chosen to", "appointed", "named", "became"])


def _should_drop_year_role_misextraction(pred: str, sent_text: str, obj: str) -> bool:
    year = _extract_year(obj) or _extract_year(sent_text)
    if not year:
        return False
    s = sent_text.lower()
    if pred in {"publication_year", "publication_date"} and "published" not in s and "released" not in s:
        return True
    if pred in {"date_of_birth", "date_of_death", "place_of_birth", "place_of_death"} and not any(k in s for k in ["born", "died"]):
        return True
    if pred == "founded_by" and "founded" not in s:
        return True
    if _looks_like_role_tenure_sentence(sent_text):
        return True
    return False

HUMAN_OCCUPATIONS = {
    "physicist", "theoretical physicist", "playwright", "poet", "writer",
    "author", "scientist", "mathematician", "chemist", "biologist",
    "philosopher", "engineer", "composer", "artist", "painter", "musician", "actor", "director", "inventor", "politician"
}
#This function converts raw relation text(sentense,verb,prepostion) into a canonical predicate and a boolean to know if it was successfully mapped.
def _normalize_relation(sent_text: str, verb_lemma: str, prep: str | None) -> Tuple[str, bool]:
    """
    Rule-based relation normalization.
    Returns: (predicate, mapped_bool)
    """
    #Lowercasing makes matching easier
    s = sent_text.lower()

    # Alias / spelling-variant patterns
    if any(k in s for k in ["alternative spelling", "alternate spelling", "spelling of", "variant spelling", "alternative form"]):
        return ("alias_of", True)

    # Capital patterns
    if "capital" in s:
        return ("capital_of", True)

    # Publication year pattern (avoid mapping years to located_in)
    if "published" in s and _extract_year(sent_text):
        return ("publication_year", True)

    # Birth / death patterns
    # NOTE: spaCy lemmatizes "born" as "bear" sometimes
    if verb_lemma in ("bear", "be") and "born" in s and prep == "in":
        return ("place_of_birth", True)
    if verb_lemma == "die" and prep == "in":
        return ("place_of_death", True)

    # Location-ish patterns
    if prep in ("in", "at", "within", "inside"):
        if ("published" in s or "released" in s) and _extract_year(sent_text):
            return ("publication_year", True)

        bad_location_context = [
            "world", "history", "culture", "civilization",
            "river", "coast", "shore", "bank", "corner",
            "northern ", "southern ", "eastern ", "western "
        ]
        if any(bad in s for bad in bad_location_context):
            return ("", False)

        if "based" in s or "located" in s or "situated" in s:
            return ("located_in", True)

    # Fallback: keep free-form relation text
    # (good for analysis; verification module can later decide what to do)
    if prep:
        return (f"{verb_lemma}_{prep}", False)
    if verb_lemma in ("be", "become") and any(job in s for job in HUMAN_OCCUPATIONS):
        return ("occupation", True)
    return (verb_lemma, False)


#takes a whole text anf extract triples as dictionaries
def extract_triples_spacy(text: str) -> List[Dict[str, Any]]:
    """
    spaCy-based extraction: outputs triples with a normalized predicate if matched
    or a free-form predicate otherwise (with mapped flag).
    """
    doc = nlp(text)
    triples: List[Dict[str, Any]] = []

    #simple context memory: keep last non-pronoun subject to resolve "It/The novel/The book"
    last_subject: str | None = None

    #helper: anchor subject for "X of Y" patterns (author/capital/CEO/president/etc.)
    def _maybe_reanchor_subject(subj_tok, subj_text: str, sent_text: str) -> str:
        """
        If the subject is an attribute phrase like "The author of",
        try to recover the real entity after "of" (e.g., book/country/company).
        """
        if subj_tok is None:
            return _clean_entity(subj_text)

        # prefer quoted entity if present (often the real title / spelling variant)
        quoted = _extract_quoted(sent_text)
        if quoted:
            # if the subject is vague like "the author of", quoted text is usually the object of "of"
            if subj_tok.lemma_.lower() in ("author", "capital", "ceo", "president", "founder"):
                return quoted

        # if subject is like "author of <WORK>", store <WORK> instead
        if subj_tok.lemma_.lower() in ("author", "capital", "ceo", "president", "founder"):
            obj = _extract_of_object(subj_tok)
            if obj:
                return obj

        # generic: if the subject span ends with "of", also try extracting pobj
        if subj_text.strip().lower().endswith("of"):
            obj = _extract_of_object(subj_tok)
            if obj:
                return obj

        return _clean_entity(subj_text)

    #helper: convert vague subject phrases into last_subject (verifiability-first)
    def _resolve_subject(subject_text: str) -> str:
        """
        If subject is vague like "the novel/the book/it", use last_subject.
        Otherwise keep subject as is.
        """
        nonlocal last_subject
        if last_subject is not None and _is_vague_subject(subject_text):
            return last_subject
        return subject_text

    #helper: drop narrative subjects that are not KG entities (reduces NEI noise)
    def _drop_narrative_subject(subject_text: str) -> bool:
        t = subject_text.lower().strip()
        # subjects that typically introduce narrative/analysis instead of facts
        return any(t.startswith(x) for x in [
            "the story", "their", "his", "her", "its", "these", "those",
            "their intense feelings", "the relationship"
        ])

    for sent in doc.sents:
        sent_text = sent.text.strip()

        #special rule: detect pseudonym statements like
        # "It was originally released under the pseudonym 'Ellis Bell'"
        if "pseudonym" in sent_text.lower():
            pseudo = _extract_quoted(sent_text)

            if pseudo and last_subject:
                triples.append({
                    "s": _clean_entity(last_subject),
                    "p": "published_under_pseudonym",
                    "o": _clean_entity(pseudo),
                    "sentence": sent_text,
                    "mapped": True,
                    "negated": False
                })

            #skip normal extraction for this sentence
            continue


        # Find main verb/root
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root is None:
            continue

        #for negation
        neg_root = _is_negated(root)

        # Find subject (nsubj/nsubjpass)
        subj = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subj is None:
            continue

        #subject phrase
        subject = _span_text(subj)
        #special fix: if subject is "author/capital/ceo/... of X", anchor to X
        subject = _maybe_reanchor_subject(subj, subject, sent_text)

        #resolve pronouns / vague subjects using last_subject (simple coreference-lite)
        subject = _resolve_subject(subject)

        #do not store garbage like narrative subjects; reduce un-verifiable triples
        if _drop_narrative_subject(subject):
            continue
        if _is_generic_entity(subject):
            continue
                  # NEW: handle patterns like:
        # "Albert Einstein is not known as a painter."
        # "She is known as a poet."
        sent_l = sent_text.lower()

        if subj is not None and "known as" in sent_l:
            subject_clean = _clean_entity(subject)

            # try to find the object after "as"
            occ_obj = None
            for tok in sent:
                if tok.dep_ == "pobj" and tok.head.lemma_.lower() == "as":
                    candidate = _clean_entity(_span_text(tok))
                    if candidate.lower() in HUMAN_OCCUPATIONS:
                        occ_obj = candidate
                        break

            if occ_obj and not _is_generic_entity(subject_clean):
                triples.append({
                    "s": subject_clean,
                    "p": "occupation",
                    "o": occ_obj,
                    "sentence": sent_text,
                    "mapped": True,
                    "negated": neg_root
                })
                continue

        #update memory only when subject is not vague
        if not _is_vague_subject(subject):
            last_subject = subject

        # Case 1: Copula "X is Y" => root is often "is"; the attribute is 'attr'
        attr = next((c for c in root.children if c.dep_ == "attr"), None)
        if attr is not None:
            attr_text = _clean_entity(_span_text(attr)).lower()
            attr_text = re.sub(r"^(?:a|an|the)\s+", "", attr_text).strip()
            if attr_text.lower() in HUMAN_OCCUPATIONS:
                triples.append({
                    "s": _clean_entity(subject),
                    "p": "occupation",
                    "o": attr_text,
                    "sentence": sent_text,
                    "mapped": True,
                    "negated": (neg_root or _is_negated(attr))
                })
                continue
            # ✅ FIX: handle "The author of X is Y" where "author" is the SUBJECT (nsubj)
            if subj.lemma_.lower() == "author":
                work = _extract_of_object(subj) or _extract_quoted(sent_text)
                if work:
                    #try to extract PERSON entity as the author name
                    author_name = None
                    for ent in sent.ents:
                        if ent.label_ == "PERSON":
                            author_name = _clean_entity(ent.text)
                            break

                    #fallback: take the attr span as author if no PERSON entity found
                    if author_name is None:
                        author_name = attr_text

                    neg_subj = _is_negated(subj)
                    neg_attr = _is_negated(attr)
                    triples.append({
                        "s": _clean_entity(work),
                        "p": "written_by",
                        "o": _clean_entity(author_name),
                        "sentence": sent_text,
                        "mapped": True,
                        "negated": (neg_root or neg_subj or neg_attr)
                    })

                    #update last_subject to the work, not "the author of"
                    last_subject = _clean_entity(work)
                    continue

            # ✅ NEW: handle spelling/alias patterns like '"Shakspere" is an alternative spelling of William Shakespeare'
            # here subject is often the quoted variant, and attr is the "William Shakespeare ..." phrase
            if "spelling" in sent_text.lower() and "of" in sent_text.lower():
                variant = _extract_quoted(sent_text) or subject
                # try to pull the main PERSON from the sentence
                person = None
                for ent in sent.ents:
                    if ent.label_ == "PERSON":
                        person = _clean_entity(ent.text)
                        break
                if person is None:
                    # fallback: try to cut at first comma for cleaner entity
                    person = _clean_entity(attr_text.split(",")[0])

                triples.append({
                    "s": _clean_entity(variant),
                    "p": "alias_of",
                    "o": _clean_entity(person),
                    "sentence": sent_text,
                    "mapped": True,
                    "negated": neg_root
                })

                # store canonical person as last_subject (helps "He was born..." after)
                last_subject = _clean_entity(person)
                continue

            #special pattern: "author of X is Y"  ->  (X, written_by, Y)
            # (this catches the rarer parse where "author" becomes attr)
            if attr.lemma_.lower() == "author":
                work = _extract_of_object(attr)
                if work:
                    #try to extract PERSON entity as the author name
                    author_name = None
                    for ent in sent.ents:
                        if ent.label_ == "PERSON":
                            author_name = _clean_entity(ent.text)
                            break

                    #fallback: take the right side span as author if no PERSON entity found
                    if author_name is None:
                        author_name = _clean_entity(sent.doc[attr.right_edge.i + 1 : sent.end].text)

                    neg_attr = _is_negated(attr)
                    triples.append({
                        "s": _clean_entity(work),
                        "p": "written_by",
                        "o": _clean_entity(author_name),
                        "sentence": sent_text,
                        "mapped": True,
                        "negated": (neg_root or neg_attr)
                    })

                    #update last_subject to the work
                    last_subject = _clean_entity(work)
                    continue

            # Look for prep under attr: "capital of Turkey"
            prep = next((c for c in attr.children if c.dep_ == "prep"), None)
            if prep is not None:
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is not None:
                    obj = _clean_entity(_span_text(pobj))
                    pred, mapped = _normalize_relation(sent_text, attr.lemma_.lower(), prep.lemma_.lower())
                    neg_attr = _is_negated(attr)

                    # ✅ FIX: if predicate becomes publication_year, force object to the extracted year
                    if pred == "publication_year":
                        year = _extract_year(sent_text)
                        if year:
                            obj = year

                    if _should_drop_year_role_misextraction(pred, sent_text, obj):
                        continue
                    if _is_generic_entity(subject) or _is_generic_entity(obj):
                        continue

                    triples.append({
                        "s": _clean_entity(subject),
                        "p": pred,
                        "o": _clean_entity(obj),
                        "sentence": sent_text,
                        "mapped": mapped,
                        "negated": (neg_root or neg_attr)
                    })
                    continue

            # If no prepositional object, fall back to attr itself
            # 🔥 Added negation handling here as well
            neg_attr = _is_negated(attr)

            # extra rule: if sentence contains published + year, attach year to subject
            year = _extract_year(sent_text)
            # if subject is vague like "the novel", use last_subject
            real_subject = last_subject if (last_subject and _is_vague_subject(subject)) else subject
            if year and "published" in sent_text.lower() and not _is_generic_entity(real_subject):
                triples.append({
                    "s": _clean_entity(real_subject),
                    "p": "publication_year",
                    "o": year,
                    "sentence": sent_text,
                    "mapped": True,
                    "negated": (neg_root or neg_attr)
                })
                continue

            # optional filter: drop very vague "impact/famous/greatest" type copula statements
            if any(k in sent_text.lower() for k in ["lasting impact", "one of the greatest", "renowned", "celebrated for", "considered a classic"]):
                continue
            # optional filter: drop descriptive "known for" / "famous for" statements (usually not KG-verifiable)
            if any(k in sent_text.lower() for k in ["known for", "famous for", "renowned for", "celebrated for"]):
                continue

            triples.append({
                "s": _clean_entity(subject),
                "p": "is",
                "o": _clean_entity(attr_text),
                "sentence": sent_text,
                "mapped": False,
                "negated": (neg_root or neg_attr)
            })
            continue

        # Case 2: Direct object (dobj/obj)
        dobj = next((c for c in root.children if c.dep_ in ("dobj", "obj")), None)
        if dobj is not None:
            obj_text = _clean_entity(_span_text(dobj))
            if _is_generic_entity(subject) or _is_generic_entity(obj_text):
                continue

            triples.append({
                "s": _clean_entity(subject),
                "p": root.lemma_.lower(),
                "o": obj_text,
                "sentence": sent_text,
                "mapped": False,
                "negated": neg_root
            })
            continue

        # Case 3: Prepositional object: "based in Ankara"
        prep = next((c for c in root.children if c.dep_ == "prep"), None)
        if prep is not None:
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj is not None:
                obj = _clean_entity(_span_text(pobj))
                pred, mapped = _normalize_relation(sent_text, root.lemma_.lower(), prep.lemma_.lower())

              # if subject is vague like "the novel", use last_subject
                real_subject = last_subject if (last_subject and _is_vague_subject(subject)) else subject

                # If this looks like "published in 1847", override object to the year
                year = _extract_year(sent_text)
                if year and "published" in sent_text.lower() and not _is_generic_entity(real_subject):
                    pred, mapped = ("publication_year", True)
                    obj = year

                # ✅ FIX: if predicate becomes publication_year, force object to the extracted year
                if pred == "publication_year":
                    year2 = _extract_year(sent_text)
                    if year2:
                        obj = year2

                # If this looks like "born in PLACE", make it place_of_birth not located_in
                if "born" in sent_text.lower() and prep.lemma_.lower() == "in" and not _extract_year(obj):
                    pred, mapped = ("place_of_birth", True)

                # If this looks like "died in PLACE", make it place_of_death not located_in
                if "died" in sent_text.lower() and prep.lemma_.lower() == "in" and not _extract_year(obj):
                    pred, mapped = ("place_of_death", True)

                if _should_drop_year_role_misextraction(pred, sent_text, obj):
                    continue
                if _is_generic_entity(real_subject) or _is_generic_entity(obj):
                    continue

                triples.append({
                    "s": _clean_entity(real_subject),
                    "p": pred,
                    "o": _clean_entity(obj),
                    "sentence": sent_text,
                    "mapped": mapped,
                    "negated": neg_root
                })
                continue

    # Light dedup as Sometimes extraction duplicates the same triple
    seen = set()
    out = []
    for t in triples:
        key = (t["s"].lower(), t["p"].lower(), t["o"].lower(), bool(t.get("negated", False)))
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out