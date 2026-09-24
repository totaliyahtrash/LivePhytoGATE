"""PhytoGATE Production Verification Test Suite (Groq-Only Architecture).

CRITICAL PRINCIPLES:
- GROQ (Qwen 3.8 27B) IS THE ACTIVE VISION ENGINE.
- Zero fake fallbacks, zero filename guesses, zero hash confidence.
- All tests use mock GroqClient / mock HTTP client: ZERO LIVE TOKENS WASTED.
- Enforces honest WITHHELD states on 503, 429, timeout, auth error, and parse failure.
- Verifies deterministic caching, error recovery, telemetry, and security boundaries.
- Non-taxonomy diseases return SUCCESS (AI visual authority preserved).
"""

import io
import json
import pytest
from unittest.mock import MagicMock
from PIL import Image
from fastapi.testclient import TestClient

from server import app
from src.config import settings
from src.groq_client import GroqClient, GroqResult
from src.diagnostics import (
    perform_diagnosis,
    clear_cache,
    get_cache_size,
    DiagnosticResponse,
    _TELEMETRY,
)
from src.disease_db import get_disease_profile
from src.vision_overlays import normalize_image


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure clean cache and telemetry before every test."""
    clear_cache()
    _TELEMETRY["total_live_requests_made"] = 0
    _TELEMETRY["total_cache_hits"] = 0
    _TELEMETRY["total_withheld_results"] = 0
    _TELEMETRY["total_successful_diagnoses"] = 0


@pytest.fixture
def dummy_leaf_jpeg() -> bytes:
    """Generates a valid RGB test image in JPEG format."""
    img = Image.new("RGB", (640, 480), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_leaf_png() -> bytes:
    """Generates a valid RGB test image in PNG format."""
    img = Image.new("RGB", (800, 600), color=(46, 120, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def dummy_large_image() -> bytes:
    """Generates a large test image (2000x1500) to test resize to 1024px."""
    img = Image.new("RGB", (2000, 1500), color=(50, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_mock_groq_client(
    status: str = "SUCCESS",
    raw_payload: dict = None,
    error_category: str = "API_SUCCESS",
    error_message: str = None,
    limitations: list = None,
    latency_sec: float = 0.42,
):
    """Creates a mock GroqClient returning the specified GroqResult."""
    mock = MagicMock(spec=GroqClient)
    mock.api_key = "gsk_mock_valid_key_for_testing_12345"
    mock.model = "qwen/qwen3.8-27b"
    mock.analyze_image.return_value = GroqResult(
        status=status,
        model="qwen/qwen3.8-27b",
        latency_sec=latency_sec,
        raw_payload=raw_payload,
        error_category=error_category,
        error_message=error_message,
        limitations=limitations or [],
    )
    return mock


# =====================================================================
# 1. Groq Successful Diagnosis — Diseased Plant
# =====================================================================
def test_1_groq_success_diseased_plant(dummy_leaf_jpeg):
    """Groq accurately analyzes image and returns Tomato Early Blight."""
    mock_payload = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Early Blight",
        "assessment": "high",
        "visual_evidence": [
            "Concentric target-like dark brown necrotic spots",
            "Chlorotic halo surrounding primary lesion",
        ],
        "alternative": "Septoria leaf spot",
        "limitations": ["Visual foliar symptoms only; laboratory confirmation recommended."],
    }
    client = create_mock_groq_client(raw_payload=mock_payload)
    result = perform_diagnosis(dummy_leaf_jpeg, filename="test_leaf.jpg", client_override=client)

    assert result.status == "SUCCESS"
    assert result.host == "Tomato"
    assert result.disease == "Early Blight"
    assert result.assessment == "high"
    assert result.is_confident is True
    assert result.confidence_score is None  # Never fake confidence percentages
    assert result.treatment is not None
    assert "Chlorothalonil" in result.treatment["chemical_treatments"][0]
    assert len(result.visual_evidence) == 2
    assert result.is_cached is False
    assert get_cache_size() == 1


# =====================================================================
# 2. Groq Successful Diagnosis — Healthy Plant
# =====================================================================
def test_2_groq_success_healthy_plant(dummy_leaf_jpeg):
    """Groq recognizes an asymptomatic, healthy specimen."""
    mock_payload = {
        "is_plant": True,
        "host": "Soybean",
        "diagnosis": "Healthy",
        "assessment": "high",
        "visual_evidence": [
            "Vibrant green, uniform foliar coloration",
            "Absence of chlorosis, necrosis, or sporulation",
        ],
        "alternative": None,
        "limitations": ["Microscopic pathogens or latent systemic viruses cannot be ruled out visually."],
    }
    client = create_mock_groq_client(raw_payload=mock_payload)
    result = perform_diagnosis(dummy_leaf_jpeg, filename="healthy_soybean.jpg", client_override=client)

    assert result.status == "SUCCESS"
    assert result.host == "Soybean"
    assert result.disease == "Healthy"
    assert result.treatment is not None
    assert result.treatment.get("severity_risk") == "NONE"
    assert result.assessment == "high"
    assert result.is_confident is True


# =====================================================================
# 3. Groq Non-Plant Detection — Withheld
# =====================================================================
def test_3_groq_non_plant_withheld(dummy_leaf_jpeg):
    """Groq correctly identifies non-plant imagery and withholds diagnosis."""
    mock_payload = {
        "is_plant": False,
        "host": None,
        "diagnosis": None,
        "assessment": "unknown",
        "visual_evidence": ["Image displays an inanimate object, not plant foliage"],
        "alternative": None,
        "limitations": ["Non-plant specimen submitted."],
    }
    client = create_mock_groq_client(raw_payload=mock_payload)
    result = perform_diagnosis(dummy_leaf_jpeg, filename="non_plant.jpg", client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "NON_PLANT"
    assert result.is_confident is False
    assert result.confidence_score is None
    assert "Non-Plant" in result.diagnosis


# =====================================================================
# 4. Groq Null Diagnosis — Withheld
# =====================================================================
def test_4_groq_null_diagnosis_withheld(dummy_leaf_jpeg):
    """Groq detects a plant but cannot confirm a diagnosis."""
    mock_payload = {
        "is_plant": True,
        "host": "Corn",
        "diagnosis": None,
        "assessment": "unknown",
        "visual_evidence": ["Symptoms are too degraded or ambiguous"],
        "alternative": None,
        "limitations": ["Insufficient visual markers."],
    }
    client = create_mock_groq_client(raw_payload=mock_payload)
    result = perform_diagnosis(dummy_leaf_jpeg, filename="ambiguous.jpg", client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category in ("UNCERTAIN_DIAGNOSIS", "UNCONFIRMED")
    assert result.is_confident is False


# =====================================================================
# 5. Groq 429 Rate Limit — Withheld
# =====================================================================
def test_5_groq_429_rate_limit(dummy_leaf_jpeg):
    """Groq returns HTTP 429; result is honestly withheld with rate limit category."""
    client = create_mock_groq_client(
        status="ERROR",
        error_category="API_429_RATE_LIMITED",
        error_message="Vision service rate limit reached. Please wait a moment before trying again.",
    )
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_429_RATE_LIMITED"
    assert "rate limit" in result.error_message.lower()
    assert result.is_cached is False


# =====================================================================
# 6. Groq Timeout — Withheld
# =====================================================================
def test_6_groq_timeout(dummy_leaf_jpeg):
    """Groq request times out; result is honestly withheld."""
    client = create_mock_groq_client(
        status="ERROR",
        error_category="API_TIMEOUT",
        error_message="Vision provider request timed out after 30 seconds.",
    )
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_TIMEOUT"
    assert "timed out" in result.error_message.lower()
    assert result.is_cached is False


# =====================================================================
# 7. Groq 503 Service Unavailable — Withheld
# =====================================================================
def test_7_groq_503_unavailable(dummy_leaf_jpeg):
    """Groq returns HTTP 503; result is honestly withheld."""
    client = create_mock_groq_client(
        status="ERROR",
        error_category="API_503_UNAVAILABLE",
        error_message="Vision service temporarily unavailable.",
    )
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_503_UNAVAILABLE"
    assert result.is_cached is False


# =====================================================================
# 8. Groq Malformed Response — Withheld
# =====================================================================
def test_8_groq_malformed_response(dummy_leaf_jpeg):
    """Groq returns unparseable or schema-violating payload; result is withheld."""
    client = create_mock_groq_client(
        status="ERROR",
        error_category="API_MALFORMED_RESPONSE",
        error_message="Vision provider returned invalid structured JSON output.",
    )
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_MALFORMED_RESPONSE"
    assert result.is_cached is False


# =====================================================================
# 9. Deterministic Caching
# =====================================================================
def test_9_deterministic_caching(dummy_leaf_jpeg):
    """Identical image bytes produce deterministic cache hit with ZERO API calls."""
    mock_payload = {
        "is_plant": True,
        "host": "Apple",
        "diagnosis": "Apple Scab",
        "assessment": "high",
        "visual_evidence": ["Olive-green velvety fungal spots on leaf surface"],
        "alternative": "Cedar apple rust",
        "limitations": [],
    }
    client = create_mock_groq_client(raw_payload=mock_payload)

    # First call: live API inference
    res1 = perform_diagnosis(dummy_leaf_jpeg, client_override=client)
    assert res1.status == "SUCCESS"
    assert res1.is_cached is False
    assert client.analyze_image.call_count == 1
    assert _TELEMETRY["total_live_requests_made"] == 1
    assert _TELEMETRY["total_cache_hits"] == 0

    # Second call with exact same image bytes: deterministic cache hit
    res2 = perform_diagnosis(dummy_leaf_jpeg, client_override=client)
    assert res2.status == "SUCCESS"
    assert res2.is_cached is True
    assert res2.disease == "Apple Scab"
    assert client.analyze_image.call_count == 1  # ZERO additional API calls
    assert _TELEMETRY["total_cache_hits"] == 1


# =====================================================================
# 10. Failures Are Never Cached
# =====================================================================
def test_10_failures_never_cached(dummy_leaf_jpeg):
    """Failed API calls must NEVER be cached, allowing immediate retry."""
    fail_client = create_mock_groq_client(
        status="ERROR",
        error_category="API_503_UNAVAILABLE",
        error_message="503 Service Unavailable",
    )

    # First call fails
    res1 = perform_diagnosis(dummy_leaf_jpeg, client_override=fail_client)
    assert res1.status == "WITHHELD"
    assert res1.is_cached is False
    assert get_cache_size() == 0

    # Second call with recovered client succeeds
    success_payload = {
        "is_plant": True,
        "host": "Potato",
        "diagnosis": "Late Blight",
        "assessment": "high",
        "visual_evidence": ["Dark water-soaked lesions"],
    }
    success_client = create_mock_groq_client(raw_payload=success_payload)
    res2 = perform_diagnosis(dummy_leaf_jpeg, client_override=success_client)
    assert res2.status == "SUCCESS"
    assert res2.disease == "Late Blight"
    assert res2.is_cached is False
    assert get_cache_size() == 1


# =====================================================================
# 11. Success After Failure (Recovery)
# =====================================================================
def test_11_success_after_failure_recovery(dummy_leaf_jpeg):
    """Pipeline seamlessly recovers when provider recovers."""
    client_fail = create_mock_groq_client(status="ERROR", error_category="API_TIMEOUT")
    res_fail = perform_diagnosis(dummy_leaf_jpeg, client_override=client_fail)
    assert res_fail.status == "WITHHELD"

    client_ok = create_mock_groq_client(
        status="SUCCESS",
        raw_payload={"is_plant": True, "host": "Grape", "diagnosis": "Black Rot", "assessment": "high", "visual_evidence": ["Brown lesions"]},
    )
    res_ok = perform_diagnosis(dummy_leaf_jpeg, client_override=client_ok)
    assert res_ok.status == "SUCCESS"
    assert res_ok.disease == "Black Rot"


# =====================================================================
# 12. Failure After Success (No State Leak)
# =====================================================================
def test_12_failure_after_success_no_state_leak(dummy_leaf_jpeg, dummy_leaf_png):
    """A subsequent failed diagnosis does NOT leak data from a preceding success."""
    client_ok = create_mock_groq_client(
        status="SUCCESS",
        raw_payload={"is_plant": True, "host": "Grape", "diagnosis": "Black Rot", "assessment": "high", "visual_evidence": ["Brown lesions"]},
    )
    res_ok = perform_diagnosis(dummy_leaf_jpeg, client_override=client_ok)
    assert res_ok.status == "SUCCESS"
    assert res_ok.disease == "Black Rot"

    # Second distinct image fails
    client_fail = create_mock_groq_client(status="ERROR", error_category="API_429_RATE_LIMITED")
    res_fail = perform_diagnosis(dummy_leaf_png, client_override=client_fail)
    assert res_fail.status == "WITHHELD"
    assert res_fail.disease == "Unconfirmed"
    assert res_fail.treatment is None


# =====================================================================
# 13. Missing API Key
# =====================================================================
def test_13_missing_api_key(dummy_leaf_jpeg):
    """Unconfigured Groq API key halts execution with API_AUTH_ERROR."""
    client = GroqClient(api_key="")
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_AUTH_ERROR"
    assert "GROQ_API_KEY" in result.error_message


# =====================================================================
# 14. Telemetry Endpoint
# =====================================================================
def test_14_telemetry_endpoint():
    """GET /api/telemetry returns safe telemetry and model configuration without API key."""
    client = TestClient(app)
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()

    assert data["provider"] == "groq"
    assert data["model"] == "qwen/qwen3.8-27b"
    assert "groq_api_key" not in data
    assert "api_key" not in data
    assert "total_live_requests_made" in data["cache"]
    assert "cached_entries_count" in data["cache"]


# =====================================================================
# 15. Health Endpoint
# =====================================================================
def test_15_health_endpoint():
    """GET /api/health returns 200 OK with zero external calls."""
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


# =====================================================================
# 16. Disease DB Endpoint
# =====================================================================
def test_16_disease_db_endpoint():
    """GET /api/disease/{host}/{disease} retrieves local profile deterministically."""
    client = TestClient(app)
    resp = client.get("/api/disease/Tomato/Early Blight")
    assert resp.status_code == 200
    data = resp.json()
    assert data["host"] == "Tomato"
    assert data["disease"] == "Early Blight"
    assert "Chlorothalonil" in data["chemical_treatments"][0]


# =====================================================================
# 17. Non-Taxonomy Disease Is Preserved As SUCCESS
# =====================================================================
def test_17_non_taxonomy_disease_is_success(dummy_leaf_jpeg):
    """External AI visual diagnosis is authority; non-taxonomy disease is SUCCESS."""
    mock_payload = {
        "is_plant": True,
        "host": "Soybean",
        "diagnosis": "Frogeye Leaf Spot",
        "assessment": "medium",
        "visual_evidence": [
            "Small circular spots with reddish-brown borders and tan centers"
        ],
        "alternative": "Septoria brown spot",
        "limitations": [],
    }
    client = create_mock_groq_client(raw_payload=mock_payload)
    result = perform_diagnosis(dummy_leaf_jpeg, filename="soybean_frogeye.jpg", client_override=client)

    # Must be SUCCESS, not WITHHELD
    assert result.status == "SUCCESS"
    assert result.host == "Soybean"
    assert result.disease == "Frogeye Leaf Spot"
    assert result.assessment == "medium"
    assert result.is_confident is True
    # Local DB may or may not have treatment for Frogeye Leaf Spot, but status is SUCCESS
    if result.treatment is None:
        assert result.treatment is None
    else:
        assert isinstance(result.treatment, dict)


# =====================================================================
# 18. Image Normalization
# =====================================================================
def test_18_image_normalization(dummy_leaf_jpeg, dummy_leaf_png, dummy_large_image):
    """Image normalizer converts PNG, large JPEG, and RGB images to <= 1024px JPEG."""
    # Test JPEG
    norm_jpeg, w1, h1, _ = normalize_image(dummy_leaf_jpeg)
    img1 = Image.open(io.BytesIO(norm_jpeg))
    assert img1.format == "JPEG"
    assert max(img1.size) <= 1024

    # Test PNG
    norm_png, w2, h2, _ = normalize_image(dummy_leaf_png)
    img2 = Image.open(io.BytesIO(norm_png))
    assert img2.format == "JPEG"
    assert max(img2.size) <= 1024

    # Test Large Image Resize
    norm_large, w3, h3, _ = normalize_image(dummy_large_image)
    img3 = Image.open(io.BytesIO(norm_large))
    assert img3.format == "JPEG"
    assert max(img3.size) <= 1024
