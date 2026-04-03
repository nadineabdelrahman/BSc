# reference_linking.py
from typing import List, Dict, Any

def attach_references_to_claims(claims, verified_references):
    enriched = []

    for claim in claims:
        sentence = (claim.get("sentence", "") or "").strip().lower()
        s = (claim.get("s", "") or "").strip().lower()
        o = (claim.get("o", "") or "").strip().lower()

        claim_refs = []

        # 1) citation marker match
        for ref in verified_references:
            marker = ref.get("citation_marker")
            if marker and marker.lower() in sentence:
                claim_refs.append(ref)

        # 2) title match
        if not claim_refs:
            for ref in verified_references:
                title = (ref.get("title") or "").strip().lower()
                if title and title in sentence:
                    claim_refs.append(ref)

        # 3) author/title heuristic
        if not claim_refs:
            for ref in verified_references:
                raw = (ref.get("raw_text") or "").lower()
                authors = [a.lower() for a in ref.get("authors", [])]

                author_hit = any(a.split(",")[0].strip() in s for a in authors if a.strip())
                title_hit = (ref.get("title") or "").strip().lower() == o

                if author_hit or title_hit:
                    claim_refs.append(ref)

        # IMPORTANT: do not attach all references globally
        new_claim = dict(claim)
        new_claim["references"] = claim_refs
        enriched.append(new_claim)

    return enriched