
import re
from typing import Any, Dict, List, Tuple

CANONICAL_PREDICATE_CONFIG: Dict[str, Dict[str, Any]] = {
    "capital_of": {"property_id": "P1376", "reverse": False, "strategy": "capital", "aliases": {
        "capital_of", "is_capital_of", "capital_city_of", "has_capital_city", "capital"
    }},
    "located_in": {"property_id": "P131", "reverse": False, "strategy": "location_union", "aliases": {
        "located_in", "is_in", "in", "situated_in", "based_in", "within", "inside", "lies_in",
        "flows_through", "flow_through", "passes_through", "pass_through", "runs_through", "run_through",
        "travels_through", "extends_through"
    }},
    "part_of": {"property_id": "P361", "reverse": False, "strategy": "direct", "aliases": {
        "part_of", "belongs_to", "member_of", "within_part_of"
    }},
    "instance_of": {"property_id": "P31", "reverse": False, "strategy": "subclass", "aliases": {
        "instance_of", "is_a", "type_of", "kind_of", "class_of", "category_of"
    }},
    "born_in": {"property_id": "P19", "reverse": False, "strategy": "direct", "aliases": {"born_in"}},
    "died_in": {"property_id": "P20", "reverse": False, "strategy": "direct", "aliases": {"died_in"}},
    "place_of_birth": {"property_id": "P19", "reverse": False, "strategy": "direct", "aliases": {
        "place_of_birth", "born_in", "birthplace"
    }},
    "place_of_death": {"property_id": "P20", "reverse": False, "strategy": "direct", "aliases": {
        "place_of_death", "died_in", "deathplace"
    }},
    "date_of_birth": {"property_id": "P569", "reverse": False, "strategy": "year", "aliases": {
        "date_of_birth", "born_on", "birth_date"
    }},
    "date_of_death": {"property_id": "P570", "reverse": False, "strategy": "year", "aliases": {
        "date_of_death", "died_on", "death_date"
    }},
    "founded_by": {"property_id": "P112", "reverse": False, "strategy": "direct", "aliases": {
        "founded_by", "cofounded_by", "started_by", "established_by"
    }},
    "founder_of": {"property_id": "P112", "reverse": True, "strategy": "direct", "aliases": {
        "founder_of", "cofounder_of"
    }},
    "ceo_of": {"property_id": "P169", "reverse": False, "strategy": "direct", "aliases": {
        "ceo_of", "chief_executive_of", "heads_company"
    }},
    "president_of": {"property_id": "P35", "reverse": True, "strategy": "direct", "aliases": {
        "president_of", "head_of_state_of"
    }},
    "author_of": {"property_id": "P50", "reverse": True, "strategy": "direct", "aliases": {
        "author_of", "wrote", "authored", "writer_of"
    }},
    "written_by": {"property_id": "P50", "reverse": False, "strategy": "direct", "aliases": {
        "written_by", "authored_by", "by_author"
    }},
    "invented_by": {"property_id": "P61", "reverse": False, "strategy": "invented_by", "aliases": {
        "invented_by", "discovered_by"
    }},
    "currency_of": {"property_id": "P38", "reverse": False, "strategy": "direct", "aliases": {"currency_of"}},
    "language_of": {"property_id": "P37", "reverse": False, "strategy": "direct", "aliases": {"language_of", "official_language_of"}},
    "publication_year": {"property_id": "P577", "reverse": False, "strategy": "year", "aliases": {
        "publication_year", "published_in", "released_in"
    }},
    "publication_date": {"property_id": "P577", "reverse": False, "strategy": "year", "aliases": {
        "publication_date", "published_on", "released_on"
    }},
    "occupation": {"property_id": "P106", "reverse": False, "strategy": "direct", "aliases": {
        "occupation", "profession", "works_as", "job"
    }},
    "notable_work": {"property_id": "P800", "reverse": False, "strategy": "direct", "aliases": {"notable_work"}},
    "alias_of": {"property_id": "P4970", "reverse": False, "strategy": "direct", "aliases": {
        "alias_of", "also_known_as", "aka", "spelling_variant_of", "alternative_spelling_of"
    }},
    "founded_on": {"property_id": "P571", "reverse": False, "strategy": "year", "aliases": {
        "founded_on", "founded_in", "established_on", "established_in"
    }},
    "headquarters_in": {"property_id": "P159", "reverse": False, "strategy": "direct", "aliases": {
        "headquarters_in", "headquartered_in", "hq_in"
    }},
    "country_of_origin": {"property_id": "P495", "reverse": False, "strategy": "direct", "aliases": {"country_of_origin", "origin_country"}},
    "genre": {"property_id": "P136", "reverse": False, "strategy": "direct", "aliases": {"genre"}},
    "spouse": {"property_id": "P26", "reverse": False, "strategy": "direct", "aliases": {"spouse", "married_to"}},
    "child_of": {"property_id": "P40", "reverse": True, "strategy": "direct", "aliases": {"child_of", "son_of", "daughter_of"}},
    "educated_at": {"property_id": "P69", "reverse": False, "strategy": "direct", "aliases": {
        "educated_at", "studied_at", "attended", "graduated_from", "alumnus_of", "alumni_of"
    }},
    "population_of": {"property_id": "P1082", "reverse": False, "strategy": "direct", "aliases": {"population_of"}},
    "area_of": {"property_id": "P2046", "reverse": False, "strategy": "direct", "aliases": {"area_of"}},
    "located_at": {"property_id": "P276", "reverse": False, "strategy": "direct", "aliases": {"located_at", "at_location"}},
}

ALLOWED_PREDICATES: List[str] = list(CANONICAL_PREDICATE_CONFIG.keys())
REVERSE_PREDICATES = {p for p, cfg in CANONICAL_PREDICATE_CONFIG.items() if cfg.get("reverse")}

NON_HUMAN_INSTANCE_OBJECTS = {
    "planet", "chemical element", "programming language", "river", "car brand",
    "technology company", "fruit", "mountain", "city", "country", "tower",
    "landmark", "novella", "novel", "book", "play", "company", "corporation",
    "software", "species", "big cat", "wild cat", "element", "university",
    "social networking site", "social media platform", "e-commerce platform",
    "electric vehicle manufacturer", "electric car manufacturer"
}

UNVERIFIABLE_PREDICATES = {
    "known_for", "know_for", "famous_for", "renowned_for", "celebrated_for",
    "inspire", "inspired", "captivate", "explore", "lead_to", "contrast",
    "described_as", "considered", "notable_for"
}

# Conservative buckets: only fire when the raw predicate itself strongly suggests the canonical relation.
PREDICATE_KEYWORD_BUCKETS: List[Tuple[Tuple[str, ...], str]] = [
    (("capital",), "capital_of"),
    (("headquarter", "hq"), "headquarters_in"),
    (("study", "graduat", "attend", "alumni", "educat"), "educated_at"),
    (("author_of", "wrote", "authored", "writer_of"), "author_of"),
    (("written_by", "authored_by"), "written_by"),
    (("founded", "cofounded", "established"), "founded_by"),
    (("invent", "discover"), "invented_by"),
    (("publish", "release"), "publication_year"),
    (("birth", "born"), "place_of_birth"),
    (("death", "died"), "place_of_death"),
    (("occup", "profession", "works_as"), "occupation"),
    (("instance", "type", "kind", "class", "category"), "instance_of"),
]

LOCATION_KEYWORDS = ("located", "situated", "based", "within", "inside", "lies_in", "flows", "passes", "runs", "through")


def normalize_relation_text(raw_p: str) -> str:
    p = str(raw_p or "").strip().lower().replace(" ", "_").replace("-", "_")
    p = re.sub(r"_+", "_", p)
    if p == "know_for":
        p = "known_for"
    return p


def is_unverifiable_predicate(p: str) -> bool:
    return normalize_relation_text(p) in UNVERIFIABLE_PREDICATES


def maybe_remap_predicate_by_object(p: str, o: str) -> str:
    p = normalize_relation_text(p)
    o_l = str(o or "").strip().lower()
    if p == "occupation":
        if o_l in NON_HUMAN_INSTANCE_OBJECTS:
            return "instance_of"
        if any(x in o_l for x in [
            "programming language", "chemical element", "technology company", "car brand",
            "social networking site", "social media platform", "e-commerce platform",
            "electric vehicle manufacturer", "electric car manufacturer"
        ]):
            return "instance_of"
    return p


def _alias_to_canonical(p: str) -> str:
    for canon, cfg in CANONICAL_PREDICATE_CONFIG.items():
        if p == canon or p in cfg.get("aliases", set()):
            return canon
    return ""


def canonicalize_predicate(raw_p: str, sentence: str = "", s: str = "", o: str = "") -> str:
    p = normalize_relation_text(raw_p)
    if not p:
        return ""

    canon = _alias_to_canonical(p)
    if canon:
        return maybe_remap_predicate_by_object(canon, o)

    sent_l = str(sentence or "").lower()
    p_l = p.lower()

    # Strong location-like relation handling.
    if any(k in p_l for k in LOCATION_KEYWORDS):
        return "located_in"

    for keys, mapped in PREDICATE_KEYWORD_BUCKETS:
        if any(k in p_l for k in keys):
            if mapped in {"publication_year", "place_of_birth", "place_of_death"}:
                # keep the previous conservative behavior: dates only when the sentence really looks like one.
                if mapped == "publication_year" and not any(k in sent_l for k in ["published", "released", "founded"]):
                    continue
                if mapped == "place_of_birth" and "born" not in sent_l:
                    continue
                if mapped == "place_of_death" and "died" not in sent_l:
                    continue
            return maybe_remap_predicate_by_object(mapped, o)

    return ""


def canonicalize_predicate_with_metadata(raw_p: str, sentence: str = "", s: str = "", o: str = "") -> Dict[str, Any]:
    canon = canonicalize_predicate(raw_p, sentence=sentence, s=s, o=o)
    cfg = CANONICAL_PREDICATE_CONFIG.get(canon, {})
    return {
        "p_raw": normalize_relation_text(raw_p),
        "p_canonical": canon,
        "property_id": cfg.get("property_id"),
        "reverse": bool(cfg.get("reverse", False)),
        "strategy": cfg.get("strategy", "direct"),
    }


def map_predicate_to_wikidata(p: str) -> Tuple[str | None, bool]:
    cfg = CANONICAL_PREDICATE_CONFIG.get(normalize_relation_text(p), {})
    return cfg.get("property_id"), bool(cfg.get("reverse", False))


def get_predicate_strategy(p: str) -> str:
    cfg = CANONICAL_PREDICATE_CONFIG.get(normalize_relation_text(p), {})
    return cfg.get("strategy", "direct")
