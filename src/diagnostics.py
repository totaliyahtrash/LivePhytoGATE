"""PhytoGATE Diagnostic Core Module.

Groq (Qwen 3.8 27B) is the sole multimodal vision inference engine.
This module strictly enforces:
- Visual plant pathology analysis performed solely by external AI
- Zero fake fallbacks, zero filename guesses, zero synthetic confidence scores
- Strict Pydantic structured output validation
- Decoupled local agronomic treatment retrieval
- Deterministic SHA-256 caching of successful valid diagnoses
- Honest diagnosis withholding upon genuine vision service failures or uncertainty
"""

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from src.config import settings
from src.taxonomy import validate_diagnosis
from src.disease_db import get_disease_profile
from src.vision_overlays import normalize_image, generate_cv_overlays
from src.groq_client import GroqClient, GroqResult

logger = logging.getLogger("phytogate.diagnostics")


# Pydantic schema for structured multimodal response from Groq
class DiagnosticResponse(BaseModel):
    is_plant: bool = Field(
        ...,
        description="Whether a plant, leaf, crop, or botanical tissue is visible in the image."
    )
    host: Optional[str] = Field(
        default=None,
        description="Identified host plant or crop (e.g. Tomato, Soybean, Corn, Potato)."
    )
    diagnosis: Optional[str] = Field(
        default=None,
        description="Identified pathology or disease, or 'Healthy' if foliage is asymptomatic."
    )
    assessment: Literal["high", "medium", "low", "unknown"] = Field(
        default="unknown",
        description="Qualitative diagnostic certainty based purely on visible foliar evidence."
    )
    visual_evidence: List[str] = Field(
        default_factory=list,
        description="Visible foliar pathology symptoms (lesions, halos, sporulation, necrotic margins)."
    )
    alternative_diagnosis: Optional[str] = Field(
        default=None,
        alias="alternative",
        description="Reasonable differential diagnosis if symptoms overlap with other conditions."
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Diagnostic limitations or imaging factors preventing definitive identification."
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}


# Unified DiagnosticResult model returned by the diagnostic core
class DiagnosticResult(BaseModel):
    status: Literal["SUCCESS", "WITHHELD"]
    is_confident: bool = False
    confidence_score: Optional[float] = None  # Strictly None: NO fake percentages permitted
    assessment: Literal["high", "medium", "low", "unknown"] = "unknown"
    diagnosis: str
    host: Optional[str] = None
    disease: Optional[str] = None
    visual_evidence: List[str] = Field(default_factory=list)
    alternative_diagnosis: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    treatment: Optional[Dict[str, Any]] = None
    is_cached: bool = False
    cache_key: Optional[str] = None
    visualizations: Optional[Dict[str, Any]] = None
    latencies: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    inference_origin: Optional[str] = None


# Deterministic in-memory cache for valid successful diagnoses
# Key: SHA-256 hex digest of normalized JPEG bytes
_DIAGNOSTIC_CACHE: Dict[str, DiagnosticResult] = {}

# Telemetry counters
_TELEMETRY = {
    "total_live_requests_made": 0,
    "total_cache_hits": 0,
    "total_withheld_results": 0,
    "total_successful_diagnoses": 0,
}


def clear_cache() -> None:
    """Clears the diagnostic cache."""
    _DIAGNOSTIC_CACHE.clear()


def get_cache_size() -> int:
    """Returns number of cached diagnostic entries."""
    return len(_DIAGNOSTIC_CACHE)


def get_telemetry_stats() -> Dict[str, Any]:
    """Returns safe runtime telemetry statistics."""
    return {
        "status": "ONLINE",
        "provider": "groq",
        "model": settings.groq_model,
        "cache": {
            "cached_entries_count": len(_DIAGNOSTIC_CACHE),
            "total_live_requests_made": _TELEMETRY["total_live_requests_made"],
            "total_cache_hits": _TELEMETRY["total_cache_hits"],
            "total_withheld_results": _TELEMETRY["total_withheld_results"],
            "total_successful_diagnoses": _TELEMETRY["total_successful_diagnoses"],
        },
        "config": settings.safe_dict(),
    }


def perform_diagnosis(
    raw_image_bytes: bytes,
    filename: str = "",
    client_override: Optional[Any] = None,
) -> DiagnosticResult:
    """Executes the single production PhytoGATE diagnostic pipeline:

    1. Normalize image to standard RGB JPEG (longest edge <= 1024px)
    2. Check deterministic SHA-256 cache
    3. Query Groq (Qwen 3.8 27B) for structured multimodal visual diagnosis
    4. Parse and validate structured JSON response
    5. Screen non-plant specimens cleanly
    6. Screen uncertain or unidentifiable diagnoses honestly
    7. Display the AI's diagnosis faithfully
    8. Retrieve agronomic treatment protocol deterministically from local DB if available
    9. Cache valid diagnoses and return structured result
    """
    t_pipeline_start = time.perf_counter()

    # 1. Preprocess & Normalize Image
    t0 = time.perf_counter()
    try:
        normalized_jpeg_bytes, width, height, sha256_hash = normalize_image(raw_image_bytes)
    except ValueError as e:
        _TELEMETRY["total_withheld_results"] += 1
        return DiagnosticResult(
            status="WITHHELD",
            is_confident=False,
            confidence_score=None,
            assessment="unknown",
            diagnosis="Diagnosis Withheld — Unprocessable Image Format",
            host=None,
            disease=None,
            error_category="INVALID_IMAGE",
            inference_origin="API_INVALID_REQUEST",
            error_message=str(e),
            limitations=["The uploaded image could not be decoded or normalized as a valid photographic specimen."]
        )
    t_norm = time.perf_counter() - t0

    # 2. Check Deterministic Cache
    if sha256_hash in _DIAGNOSTIC_CACHE:
        _TELEMETRY["total_cache_hits"] += 1
        cached_result = _DIAGNOSTIC_CACHE[sha256_hash].model_copy(deep=True)
        cached_result.is_cached = True
        cached_result.inference_origin = "Deterministic Cache"
        return cached_result

    # 3. Instantiate Groq Client
    if client_override is not None:
        groq_client = client_override if isinstance(client_override, GroqClient) else GroqClient(http_client=client_override)
    else:
        groq_client = GroqClient()

    # 4. Check Provider API Configuration
    if not getattr(groq_client, "api_key", None) or len(groq_client.api_key) <= 5:
        _TELEMETRY["total_withheld_results"] += 1
        return DiagnosticResult(
            status="WITHHELD",
            is_confident=False,
            confidence_score=None,
            assessment="unknown",
            diagnosis="Diagnosis Withheld — Groq API Key Not Configured",
            host="Crop Specimen",
            disease="Unconfirmed",
            error_category="API_AUTH_ERROR",
            inference_origin="API_AUTH_ERROR",
            error_message="Groq API key is not configured. Please supply a valid GROQ_API_KEY in .env.",
            limitations=["The active diagnostic engine requires valid Groq API credentials."]
        )

    # 5. Call Groq Vision Engine
    _TELEMETRY["total_live_requests_made"] += 1

    t_api_start = time.perf_counter()
    groq_res = groq_client.analyze_image(normalized_jpeg_bytes)
    t_api = time.perf_counter() - t_api_start

    # Handle Provider Failure (Honest WITHHELD, Never Cached)
    if groq_res.status != "SUCCESS":
        _TELEMETRY["total_withheld_results"] += 1
        t_total = time.perf_counter() - t_pipeline_start
        api_latencies = {
            "image_normalization_sec": round(t_norm, 4),
            "vision_inference_sec": round(t_api, 4),
            "total_sec": round(t_total, 4),
        }
        logger.warning(
            "Groq API call failed (%s): %s",
            groq_res.error_category,
            groq_res.error_message,
        )
        return DiagnosticResult(
            status="WITHHELD",
            is_confident=False,
            confidence_score=None,
            assessment="unknown",
            diagnosis="Diagnosis Withheld — Vision Service Error",
            host="Crop Specimen",
            disease="Unconfirmed",
            latencies=api_latencies,
            error_category=groq_res.error_category,
            inference_origin=groq_res.error_category,
            error_message=groq_res.error_message or "The vision service could not complete analysis. No diagnosis was generated.",
            limitations=groq_res.limitations or ["Diagnostic inference could not be completed by the multimodal vision engine."]
        )

    # 6. Parse and Validate Structured JSON Response
    t_parse_start = time.perf_counter()
    try:
        if isinstance(groq_res.raw_payload, DiagnosticResponse):
            parsed = groq_res.raw_payload
        elif isinstance(groq_res.raw_payload, dict):
            parsed = DiagnosticResponse.model_validate(groq_res.raw_payload)
        else:
            raise ValueError(f"Unrecognized payload type: {type(groq_res.raw_payload)}")
        t_parse = time.perf_counter() - t_parse_start
    except Exception as e:
        t_parse = time.perf_counter() - t_parse_start
        _TELEMETRY["total_withheld_results"] += 1
        logger.error("Failed to parse structured JSON from Groq: %s", str(e))
        return DiagnosticResult(
            status="WITHHELD",
            is_confident=False,
            confidence_score=None,
            assessment="unknown",
            diagnosis="Diagnosis Withheld — Malformed Vision Response",
            host="Crop Specimen",
            disease="Unconfirmed",
            error_category="API_MALFORMED_RESPONSE",
            inference_origin="API_ERROR",
            error_message="The vision engine returned an unparseable response structure.",
            limitations=["Response from vision engine did not adhere to required pathology schema."]
        )

    # 7. Screen Non-Plant Specimens
    if not parsed.is_plant:
        _TELEMETRY["total_withheld_results"] += 1
        t_total = time.perf_counter() - t_pipeline_start
        latencies = {
            "image_normalization_sec": round(t_norm, 4),
            "vision_inference_sec": round(t_api, 4),
            "structured_json_parsing_sec": round(t_parse, 4),
            "total_sec": round(t_total, 4),
        }
        non_plant_result = DiagnosticResult(
            status="WITHHELD",
            is_confident=False,
            confidence_score=None,
            assessment="unknown",
            diagnosis="Diagnosis Withheld — Non-Plant Specimen Detected",
            host=None,
            disease=None,
            visual_evidence=parsed.visual_evidence,
            alternative_diagnosis=parsed.alternative_diagnosis,
            limitations=parsed.limitations or ["The uploaded image does not contain identifiable crop or plant foliage."],
            error_category="NON_PLANT",
            inference_origin="NON_PLANT",
            error_message="Image does not appear to contain a plant specimen.",
            cache_key=sha256_hash,
            latencies=latencies,
        )
        _DIAGNOSTIC_CACHE[sha256_hash] = non_plant_result.model_copy(deep=True)
        return non_plant_result

    # 8. Screen Uncertain or Null Diagnoses
    raw_diag = (parsed.diagnosis or "").strip()
    if not raw_diag or raw_diag.lower() in ["none", "null", "unknown", "unidentified"] or parsed.assessment == "unknown":
        _TELEMETRY["total_withheld_results"] += 1
        return DiagnosticResult(
            status="WITHHELD",
            is_confident=False,
            confidence_score=None,
            assessment="unknown",
            diagnosis="Diagnosis Withheld — Uncertain Pathology",
            host=parsed.host,
            disease=parsed.diagnosis or "Unconfirmed",
            visual_evidence=parsed.visual_evidence,
            alternative_diagnosis=parsed.alternative_diagnosis,
            limitations=parsed.limitations or ["Visual pathology assessment is uncertain or unconfirmed."],
            error_category="UNCERTAIN_DIAGNOSIS",
            inference_origin="UNCERTAIN_DIAGNOSIS",
            error_message="The visual pathology assessment is uncertain. Diagnosis withheld."
        )

    # 9. External AI is Diagnostic Authority — Taxonomy Used for Normalization & Treatment Lookup
    raw_disease = raw_diag
    raw_host = (parsed.host or "").strip() if parsed.host else None
    is_healthy = "healthy" in raw_disease.lower()

    t_tax_start = time.perf_counter()
    is_valid_taxonomy, canonical_host, canonical_disease = validate_diagnosis(
        raw_host,
        raw_disease
    )
    t_tax = time.perf_counter() - t_tax_start

    final_disease = canonical_disease if (is_valid_taxonomy and canonical_disease) else raw_disease
    final_host = canonical_host if canonical_host else raw_host

    if is_healthy:
        final_disease = "Healthy"
        display_diagnosis = f"{final_host} — Healthy" if final_host else "Healthy"
    else:
        display_diagnosis = f"{final_host} — {final_disease}" if final_host else final_disease

    # 10. Local Disease Database Treatment Lookup (Supplementary, Never Invented)
    t_treat_start = time.perf_counter()
    treatment_profile = None
    lookup_host = canonical_host or raw_host
    lookup_disease = canonical_disease or raw_disease
    if lookup_host and lookup_disease:
        treatment_profile = get_disease_profile(lookup_host, lookup_disease)
    t_treat = time.perf_counter() - t_treat_start

    # 11. Generate Local Computer Vision Overlays (Explicitly Heuristic)
    cv_visualizations = generate_cv_overlays(normalized_jpeg_bytes)

    # 12. Build Final DiagnosticResult
    t_total = time.perf_counter() - t_pipeline_start
    latencies = {
        "image_normalization_sec": round(t_norm, 4),
        "vision_inference_sec": round(t_api, 4),
        "structured_json_parsing_sec": round(t_parse, 4),
        "taxonomy_validation_sec": round(t_tax, 4),
        "treatment_lookup_sec": round(t_treat, 4),
        "total_sec": round(t_total, 4),
    }

    result = DiagnosticResult(
        status="SUCCESS",
        is_confident=(parsed.assessment in ["high", "medium"]),
        confidence_score=None,  # Qualitative assessment only: NO fake percentages
        assessment=parsed.assessment,
        diagnosis=display_diagnosis,
        host=final_host,
        disease=final_disease,
        visual_evidence=parsed.visual_evidence,
        alternative_diagnosis=parsed.alternative_diagnosis,
        limitations=parsed.limitations,
        treatment=treatment_profile,
        is_cached=False,
        cache_key=sha256_hash,
        visualizations=cv_visualizations,
        latencies=latencies,
        error_category="API_SUCCESS",
        inference_origin=groq_res.model,
    )

    # 13. Cache ONLY Successful Valid Diagnoses
    _DIAGNOSTIC_CACHE[sha256_hash] = result.model_copy(deep=True)
    _TELEMETRY["total_successful_diagnoses"] += 1

    return result
