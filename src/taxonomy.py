"""PhytoGATE Canonical Taxonomy Module.

Maintains the controlled list of validated plant hosts and pathologies.
Acts as a normalization boundary: normalizes crop/disease returned by the external vision model
and checks for known treatment profiles in the local knowledge base.

DISEASE-FIRST ARCHITECTURE:
- Disease identification is primary.
- Host identification is optional context.
- A validated canonical disease is accepted even when the host is unknown/null.
"""

from typing import Dict, List, Optional, Tuple

# Canonical controlled taxonomy for PhytoGATE
# Mapping: Host -> List of verified canonical disease names
CANONICAL_TAXONOMY: Dict[str, List[str]] = {
    "Tomato": [
        "Early Blight",
        "Late Blight",
        "Bacterial Spot",
        "Septoria Leaf Spot",
        "Leaf Mold",
        "Target Spot",
        "Spider Mites",
        "Tomato Yellow Leaf Curl Virus",
        "Tomato Mosaic Virus",
        "Healthy",
    ],
    "Potato": [
        "Early Blight",
        "Late Blight",
        "Healthy",
    ],
    "Apple": [
        "Apple Scab",
        "Black Rot",
        "Cedar Apple Rust",
        "Healthy",
    ],
    "Corn": [
        "Cercospora Leaf Spot (Gray Leaf Spot)",
        "Common Rust",
        "Northern Leaf Blight",
        "Healthy",
    ],
    "Grape": [
        "Black Rot",
        "Esca (Black Measles)",
        "Leaf Blight (Isariopsis Leaf Spot)",
        "Healthy",
    ],
    "Pepper": [
        "Bacterial Spot",
        "Frogeye Leaf Spot",
        "Healthy",
    ],
    "Rice": [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Rice Blast",
        "Healthy",
    ],
    "Wheat": [
        "Leaf Rust",
        "Powdery Mildew",
        "Stripe Rust",
        "Healthy",
    ],
    "Soybean": [
        "Frogeye Leaf Spot",
        "Bacterial Blight",
        "Septoria Brown Spot",
        "Healthy",
    ],
}

# Flattened list of all validated canonical diseases across all hosts
ALL_CANONICAL_DISEASES: List[str] = []
for diseases in CANONICAL_TAXONOMY.values():
    for d in diseases:
        if d not in ALL_CANONICAL_DISEASES:
            ALL_CANONICAL_DISEASES.append(d)

# Explicit, agronomically defensible aliases and pathogen scientific names
# Maps specific pathogen names and legitimate naming variants to their canonical disease name.
# NEVER map broad symptom words (e.g., "leaf spot", "brown spots", "necrosis") to specific diseases.
CANONICAL_DISEASE_ALIASES: Dict[str, str] = {
    # Frogeye Leaf Spot variants and pathogen names (Pepper, Soybean)
    "frogeye leaf spot": "Frogeye Leaf Spot",
    "frog eye leaf spot": "Frogeye Leaf Spot",
    "frog-eye leaf spot": "Frogeye Leaf Spot",
    "frogeye spot": "Frogeye Leaf Spot",
    "frog eye spot": "Frogeye Leaf Spot",
    "cercospora capsici": "Frogeye Leaf Spot",
    "cercospora sojina": "Frogeye Leaf Spot",
    "cercospora leaf spot (frogeye leaf spot)": "Frogeye Leaf Spot",

    # Bacterial Spot variants (Tomato, Pepper)
    "bacterial leaf spot": "Bacterial Spot",
    "bacterial spot": "Bacterial Spot",
    "xanthomonas leaf spot": "Bacterial Spot",
    "xanthomonas vesicatoria": "Bacterial Spot",
    "xanthomonas euvesicatoria": "Bacterial Spot",
    "xanthomonas perforans": "Bacterial Spot",

    # Gray Leaf Spot variants (Corn)
    "grey leaf spot": "Cercospora Leaf Spot (Gray Leaf Spot)",
    "gray leaf spot": "Cercospora Leaf Spot (Gray Leaf Spot)",
    "corn gray leaf spot": "Cercospora Leaf Spot (Gray Leaf Spot)",
    "cercospora zeae maydis": "Cercospora Leaf Spot (Gray Leaf Spot)",
    "cercospora zeae-maydis": "Cercospora Leaf Spot (Gray Leaf Spot)",

    # Rust variants (Wheat, Corn)
    "yellow rust": "Stripe Rust",
    "puccinia striiformis": "Stripe Rust",
    "brown rust": "Leaf Rust",
    "puccinia triticina": "Leaf Rust",
    "puccinia sorghi": "Common Rust",

    # Blight variants (Tomato, Potato, Rice, Soybean)
    "alternaria leaf blight": "Early Blight",
    "alternaria solani": "Early Blight",
    "phytophthora infestans": "Late Blight",
    "bipolaris oryzae": "Brown Spot",
    "magnaporthe oryzae": "Rice Blast",
    "xanthomonas oryzae": "Bacterial Leaf Blight",
    "septoria glycines": "Septoria Brown Spot",
    "pseudomonas savastanoi": "Bacterial Blight",

    # Mites
    "two spotted spider mite": "Spider Mites",
    "two-spotted spider mite": "Spider Mites",
    "two-spotted spider mites": "Spider Mites",
    "tetranychus urticae": "Spider Mites",
}

# Broad non-specific symptom terms that MUST NOT be mapped to any specific disease
BROAD_SYMPTOM_TERMS = {
    "leaf spot",
    "leaf spots",
    "brown spots",
    "spot",
    "spots",
    "necrosis",
    "chlorosis",
    "lesion",
    "lesions",
    "leaf lesion",
    "leaf lesions",
    "foliar lesion",
    "foliar lesions",
    "blight",
    "yellowing",
    "wilting",
}


def _normalize_string(val: Optional[str]) -> str:
    """Normalize input string by stripping punctuation, extra spaces, and underscores."""
    if not val:
        return ""
    cleaned = (
        val.replace("___", " ")
        .replace("__", " ")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .lower()
    )
    cleaned = " ".join(cleaned.split())
    return cleaned


def _normalize_compact(val: Optional[str]) -> str:
    """Normalize string removing all whitespace and non-alphanumeric chars for token-agnostic matching."""
    if not val:
        return ""
    return "".join(c for c in val.lower() if c.isalnum())


def match_canonical_host(raw_host: Optional[str]) -> Optional[str]:
    """Matches raw_host against canonical hosts in CANONICAL_TAXONOMY.

    Returns canonical host name if matched, or None if unknown/unsupported/unconfirmed.
    """
    if not raw_host:
        return None

    norm_host = _normalize_string(raw_host)
    if not norm_host or norm_host in {
        "unknown", "unidentified", "null", "none", "crop", "crop specimen",
        "plant", "plant specimen", "leaf", "foliage", "unconfirmed", "undefined"
    }:
        return None

    for host in CANONICAL_TAXONOMY.keys():
        h_norm = _normalize_string(host)
        if norm_host == h_norm or norm_host.startswith(h_norm) or h_norm in norm_host:
            return host

    return None


def match_canonical_disease(
    raw_disease: Optional[str],
    preferred_host: Optional[str] = None
) -> Optional[str]:
    """Matches raw_disease against canonical diseases.

    If preferred_host is valid in CANONICAL_TAXONOMY, we check that host's diseases first.
    """
    if not raw_disease:
        return None

    norm_disease = _normalize_string(raw_disease)
    compact_disease = _normalize_compact(raw_disease)
    if not norm_disease or norm_disease in {"unknown", "unidentified", "unconfirmed", "none", "null"}:
        return None

    # Check for healthy leaf condition
    if "healthy" in norm_disease:
        return "Healthy"

    # Reject broad non-specific symptom words immediately (e.g., "leaf spot", "brown spots", "necrosis")
    if norm_disease in BROAD_SYMPTOM_TERMS or compact_disease in {_normalize_compact(s) for s in BROAD_SYMPTOM_TERMS}:
        return None

    # Check explicit agronomically defensible aliases first
    if norm_disease in CANONICAL_DISEASE_ALIASES:
        return CANONICAL_DISEASE_ALIASES[norm_disease]
    for alias_key, canon_name in CANONICAL_DISEASE_ALIASES.items():
        if compact_disease == _normalize_compact(alias_key):
            return canon_name

    # Prioritize preferred host if provided, then all canonical diseases
    candidate_lists = []
    if preferred_host and preferred_host in CANONICAL_TAXONOMY:
        candidate_lists.append(CANONICAL_TAXONOMY[preferred_host])
    candidate_lists.append(ALL_CANONICAL_DISEASES)

    for candidates in candidate_lists:
        # 1. Exact normalized or compact match
        for d in candidates:
            if norm_disease == _normalize_string(d) or compact_disease == _normalize_compact(d):
                return d

        # 2. Main name without parentheses & alias in parentheses
        for d in candidates:
            if "(" in d and ")" in d:
                main_part = d.split("(")[0].strip()
                alias_part = d.split("(")[1].split(")")[0].strip()
                if (norm_disease == _normalize_string(main_part) or
                    norm_disease == _normalize_string(alias_part) or
                    compact_disease == _normalize_compact(main_part) or
                    compact_disease == _normalize_compact(alias_part)):
                    return d

        # 3. Substring match: only when canonical disease name is a substring of the input query
        # (e.g. "severe frogeye leaf spot infection" matches "Frogeye Leaf Spot")
        # NEVER match when query is merely a substring of the canonical disease (e.g. "spot" in "Septoria Leaf Spot")
        for d in candidates:
            d_norm = _normalize_string(d)
            d_simple = _normalize_string(d.split("(")[0])
            if len(d_simple) >= 5 and d_simple in norm_disease:
                return d
            if len(d_norm) >= 5 and d_norm in norm_disease:
                return d

    return None


def validate_diagnosis(
    raw_host: Optional[str],
    raw_disease: Optional[str]
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validates the diagnosis against canonical taxonomy.

    DISEASE-FIRST ARCHITECTURE:
    - If raw_disease is contained in the canonical validated disease taxonomy,
      the disease diagnosis is valid (is_valid = True).
    - Host is OPTIONAL:
      * If host is identifiable and supports this disease, canonical_host is returned.
      * If host is unknown, null, or outside taxonomy, canonical_host is None.
      * Disease is NEVER rejected solely because the host is unknown or cannot be identified.

    Returns:
        (is_valid, canonical_host, canonical_disease)
        If disease is invalid/unknown, returns (False, canonical_host, None).
    """
    if not raw_disease and not raw_host:
        return False, None, None

    # Handle cases where model returned combined "Tomato___Early_blight" in host or disease
    if raw_host and not raw_disease:
        norm_h = _normalize_string(raw_host)
        if " " in norm_h:
            parts = norm_h.split(" ", 1)
            raw_host = parts[0]
            raw_disease = parts[1]

    canonical_host = match_canonical_host(raw_host)
    canonical_disease = match_canonical_disease(raw_disease, preferred_host=canonical_host)

    if canonical_disease:
        # Check if canonical_host is valid and genuinely supports this disease
        if canonical_host and canonical_disease in CANONICAL_TAXONOMY.get(canonical_host, []):
            return True, canonical_host, canonical_disease
        else:
            # Valid canonical disease, but host is unknown, unconfirmed, or non-matching
            return True, None, canonical_disease

    # Disease is not present in canonical taxonomy -> invalid
    return False, canonical_host, None


def get_supported_taxonomy() -> Dict[str, List[str]]:
    """Returns the full controlled taxonomy."""
    return CANONICAL_TAXONOMY
