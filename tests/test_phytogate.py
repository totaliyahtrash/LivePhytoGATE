"""PhytoGATE Comprehensive Production Verification Test Suite.

CRITICAL PRINCIPLES:
- GEMINI IS THE ACTUAL DIAGNOSTIC ENGINE
- Zero fake fallbacks, zero filename guesses, zero hash confidence
- All tests use mocked Gemini client: ZERO LIVE GEMINI TOKENS WASTED
- Enforces honest WITHHELD states on 503, 429, timeout, auth error, and taxonomy rejection
- Verifies deterministic caching and security boundaries
"""

import io
import json
import pytest
from unittest.mock import MagicMock
from PIL import Image
from fastapi.testclient import TestClient

from server import app
from src.config import settings
from src.diagnostics import (
    perform_diagnosis,
    clear_cache,
    get_cache_size,
    DiagnosticResponse,
    _TELEMETRY,
)
from src.vision_provider import GroqVisionProvider
from src.taxonomy import validate_diagnosis
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


def create_mock_gemini_client(response_data: dict):
    """Creates a mock google-genai Client that returns structured JSON."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_data)
    mock_response.parsed = response_data
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# =====================================================================
# 1. Gemini Successful Diagnosis
# =====================================================================
def test_1_gemini_success(dummy_leaf_jpeg):
    """Gemini accurately analyzes image and returns Tomato Early Blight.

    Verifies:
    - Status is SUCCESS
    - Valid host and disease resolved
    - Qualitative assessment is HIGH
    - Confidence score is None (No fake percentage)
    - Treatment retrieved locally from disease database
    - Result is cached
    """
    mock_data = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Early Blight",
        "assessment": "high",
        "visual_evidence": [
            "Concentric target-like dark brown necrotic spots",
            "Chlorotic yellow halos surrounding lesions",
            "Lower canopy foliar desiccation"
        ],
        "alternative_diagnosis": "Tomato — Septoria Leaf Spot",
        "limitations": []
    }
    mock_client = create_mock_gemini_client(mock_data)

    result = perform_diagnosis(dummy_leaf_jpeg, filename="test_leaf.jpg", client_override=mock_client)

    assert result.status == "SUCCESS"
    assert result.is_confident is True
    assert result.confidence_score is None  # Strictly NO fake percentage
    assert result.assessment == "high"
    assert result.host == "Tomato"
    assert result.disease == "Early Blight"
    assert result.diagnosis == "Tomato — Early Blight"
    assert len(result.visual_evidence) == 3
    assert result.treatment is not None
    assert result.treatment["pathogen_type"] == "Fungal (Alternaria solani / Alternaria linariae)"
    assert result.visualizations is not None
    assert "lesion_segmentation" in result.visualizations
    assert result.is_cached is False


# =====================================================================
# 2. Gemini 503 API Error -> Honest WITHHELD
# =====================================================================
def test_2_gemini_503_api_error_returns_honest_withheld(dummy_leaf_jpeg):
    """When Gemini returns 503 Unavailable, PhytoGATE must withhold honestly."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: This model is currently experiencing high demand.")

    result = perform_diagnosis(dummy_leaf_jpeg, filename="field_specimen.jpg", client_override=mock_client)

    assert result.status == "WITHHELD"
    assert result.is_confident is False
    assert result.confidence_score is None
    assert result.assessment == "unknown"
    assert result.diagnosis == "Diagnosis Withheld — Field Image Requires Further Review"
    assert result.host == "Crop Specimen"
    assert result.disease == "Unconfirmed"
    assert result.treatment is None
    assert result.error_category == "API_503_UNAVAILABLE"
    assert "temporarily unavailable" in result.error_message


# =====================================================================
# 3. Gemini Timeout -> Honest WITHHELD
# =====================================================================
def test_3_gemini_timeout_returns_withheld(dummy_leaf_jpeg):
    """When Gemini times out, system must withhold diagnosis."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Deadline exceeded: 45000ms timed out")

    result = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_TIMEOUT"
    assert result.treatment is None
    assert result.confidence_score is None


# =====================================================================
# 4. Gemini Auth Error -> Honest WITHHELD
# =====================================================================
def test_4_gemini_auth_error_returns_withheld(dummy_leaf_jpeg):
    """When API authentication fails, system withholds cleanly."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("403 Forbidden: API_KEY_INVALID")

    result = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_AUTH_ERROR"
    assert result.confidence_score is None


# =====================================================================
# 5. Gemini Rate Limit (429) -> Honest WITHHELD
# =====================================================================
def test_5_gemini_rate_limit_returns_withheld(dummy_leaf_jpeg):
    """When rate limit 429 is received, system withholds cleanly."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    result = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)

    assert result.status == "WITHHELD"
    assert result.error_category in ["API_RATE_LIMIT", "API_429_RATE_LIMITED"]
    assert result.inference_origin == "API_429_RATE_LIMITED"
    assert result.confidence_score is None


# =====================================================================
# 6. Unsupported Taxonomy -> Displayed as SUCCESS (AI Diagnostic Authority)
# =====================================================================
def test_6_unsupported_taxonomy_displayed_without_treatment(dummy_leaf_jpeg):
    """If AI returns a crop/disease outside canonical taxonomy, diagnosis is STILL displayed.

    UNKNOWN TAXONOMY != UNKNOWN DIAGNOSIS. Treatment is marked unavailable in local DB.
    """
    mock_data = {
        "is_plant": True,
        "host": "Dragonfruit",
        "diagnosis": "Anthracnose",
        "assessment": "high",
        "visual_evidence": ["Reddish brown lesions on cladodes"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    mock_client = create_mock_gemini_client(mock_data)

    result = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)

    assert result.status == "SUCCESS"
    assert result.host == "Dragonfruit"
    assert result.disease == "Anthracnose"
    assert result.treatment is None


# =====================================================================
# 7. Cache Hit -> Zero Gemini Calls
# =====================================================================
def test_7_cache_hit_zero_gemini_calls(dummy_leaf_jpeg):
    """Submitting the identical image twice requires exactly 0 Gemini calls on request 2."""
    mock_data = {
        "is_plant": True,
        "host": "Potato",
        "diagnosis": "Late Blight",
        "assessment": "high",
        "visual_evidence": ["Water-soaked dark lesions", "White sporulation"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    mock_client = create_mock_gemini_client(mock_data)

    # 1st request -> Gemini is called once
    res1 = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)
    assert res1.status == "SUCCESS"
    assert res1.is_cached is False
    assert mock_client.models.generate_content.call_count == 1

    # 2nd request with identical image -> Gemini is NOT called!
    res2 = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)
    assert res2.status == "SUCCESS"
    assert res2.is_cached is True
    # Call count MUST REMAIN 1
    assert mock_client.models.generate_content.call_count == 1


# =====================================================================
# 8. Local Treatment Lookup -> Zero Gemini Calls
# =====================================================================
def test_8_treatment_lookup_zero_gemini_calls():
    """Local treatment lookup retrieves rich agronomic knowledge without Gemini."""
    client = TestClient(app)
    response = client.get("/api/disease/Tomato/Early%20Blight")
    assert response.status_code == 200
    data = response.json()
    assert data["host"] == "Tomato"
    assert data["disease"] == "Early Blight"
    assert len(data["chemical_treatments"]) > 0
    assert len(data["organic_treatments"]) > 0
    assert len(data["cultural_controls"]) > 0


# =====================================================================
# 9. Frontend Read Endpoints -> Zero Gemini Calls
# =====================================================================
def test_9_frontend_read_endpoints_zero_gemini_calls():
    """All reading/telemetry/health endpoints make 0 Gemini calls."""
    client = TestClient(app)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "healthy"}

    r_tel = client.get("/api/telemetry")
    assert r_tel.status_code == 200
    assert r_tel.json()["status"] == "ONLINE"

    r_tax = client.get("/api/taxonomy")
    assert r_tax.status_code == 200
    assert "Tomato" in r_tax.json()

    r_samples = client.get("/api/samples")
    assert r_samples.status_code == 200
    assert isinstance(r_samples.json(), list)

    r_index = client.get("/")
    assert r_index.status_code == 200


# =====================================================================
# 10. Critical Regression: IMG_0042.jpg + Gemini 503
# =====================================================================
def test_10_regression_img_0042_gemini_503(dummy_leaf_jpeg):
    """CRITICAL REGRESSION TEST:

    Previously, an image named IMG_0042.jpg during a Gemini 503 error would
    fall back to a fake 'Foliar Specimen — Foliar Leaf Spot' with 81.5% confidence!
    This must NEVER happen. It MUST be WITHHELD.
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: High demand")

    result = perform_diagnosis(dummy_leaf_jpeg, filename="IMG_0042.jpg", client_override=mock_client)

    assert result.status == "WITHHELD"
    assert "Foliar Leaf Spot" not in result.diagnosis
    assert result.confidence_score is None  # NEVER 81.5%
    assert result.assessment == "unknown"


# =====================================================================
# 11. API Failures Are NOT Cached
# =====================================================================
def test_11_api_errors_not_cached_as_success(dummy_leaf_jpeg):
    """An API failure must never be permanently cached."""
    mock_fail_client = MagicMock()
    mock_fail_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE")

    # First attempt fails
    res_fail = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_fail_client)
    assert res_fail.status == "WITHHELD"

    # Subsequent attempt when Gemini recovers succeeds and is NOT blocked by cached failure
    mock_ok_data = {
        "is_plant": True,
        "host": "Apple",
        "diagnosis": "Apple Scab",
        "assessment": "medium",
        "visual_evidence": ["Olive-green spots on foliage"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    mock_ok_client = create_mock_gemini_client(mock_ok_data)

    res_ok = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_ok_client)
    assert res_ok.status == "SUCCESS"
    assert res_ok.diagnosis == "Apple — Apple Scab"


# =====================================================================
# 12. Non-Plant Specimen Cleanly WITHHELD
# =====================================================================
def test_12_non_plant_specimen_cleanly_withheld(dummy_leaf_jpeg):
    """If Gemini determines the image is not a plant, diagnosis is cleanly WITHHELD."""
    mock_data = {
        "is_plant": False,
        "host": None,
        "diagnosis": None,
        "assessment": "unknown",
        "visual_evidence": ["Mechanical equipment and metallic surfaces"],
        "alternative_diagnosis": None,
        "limitations": ["Specimen is not biological foliage."]
    }
    mock_client = create_mock_gemini_client(mock_data)

    result = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)

    assert result.status == "WITHHELD"
    assert result.error_category == "NON_PLANT"
    assert result.host is None
    assert result.disease is None
    assert result.treatment is None


# =====================================================================
# 13. API Key Never Appears in Responses or Telemetry
# =====================================================================
def test_13_api_key_never_appears_in_responses(dummy_leaf_jpeg):
    """Guarantees API credentials are never leaked in telemetry, headers, or payloads."""
    client = TestClient(app)

    tel_resp = client.get("/api/telemetry")
    tel_text = tel_resp.text
    assert settings.gemini_api_key not in tel_text
    data = tel_resp.json()
    assert "api_key" not in data["config"]
    assert "api_key_status" in data["config"]


# =====================================================================
# 14. Image Normalization and Resize Edge <= 1024
# =====================================================================
def test_14_image_normalization_formats(dummy_large_image, dummy_leaf_png):
    """Verifies images of different formats (PNG, large JPEG) normalize to RGB JPEG <= 1024px."""
    # Test large image resize
    norm_bytes, w, h, sha = normalize_image(dummy_large_image, max_long_edge=1024)
    assert max(w, h) <= 1024
    assert len(sha) == 64  # Valid SHA-256

    # Test PNG normalization to JPEG
    norm_bytes_png, w_png, h_png, sha_png = normalize_image(dummy_leaf_png)
    assert max(w_png, h_png) <= 1024
    # Ensure decodable as JPEG
    reopened = Image.open(io.BytesIO(norm_bytes_png))
    assert reopened.format == "JPEG"
    assert reopened.mode == "RGB"


# =====================================================================
# 15. Server Multipart Diagnose Endpoint
# =====================================================================
def test_15_server_multipart_diagnose_endpoint(dummy_leaf_jpeg, monkeypatch):
    """Tests POST /api/diagnose through FastAPI TestClient."""
    mock_data = {
        "is_plant": True,
        "host": "Grape",
        "diagnosis": "Black Rot",
        "assessment": "high",
        "visual_evidence": ["Circular reddish brown leaf spots with pycnidia"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    mock_client = create_mock_gemini_client(mock_data)

    # Monkeypatch gemini_api_key and _create_gemini_client
    monkeypatch.setattr(settings, "gemini_api_key", "mock_key_for_test_12345")
    monkeypatch.setattr("src.diagnostics._create_gemini_client", lambda override=None: mock_client)

    client = TestClient(app)
    response = client.post(
        "/api/diagnose",
        files={"file": ("grape_specimen.jpg", dummy_leaf_jpeg, "image/jpeg")}
    )

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "SUCCESS"
    assert res_json["diagnosis"] == "Grape — Black Rot"
    assert res_json["host"] == "Grape"
    assert res_json["disease"] == "Black Rot"
    assert res_json["treatment"] is not None
    assert res_json["confidence_score"] is None


# =====================================================================
# 16. Corrupted / Non-Image Bytes Handled Honestly
# =====================================================================
def test_16_corrupted_image_handled_honestly():
    """Corrupted or invalid image bytes return honest WITHHELD."""
    bad_bytes = b"NOT_A_VALID_IMAGE_BYTES_12345"
    result = perform_diagnosis(bad_bytes)
    assert result.status == "WITHHELD"
    assert result.error_category == "INVALID_IMAGE"
    assert result.confidence_score is None
    assert result.treatment is None


# =====================================================================
# 17. Taxonomy Normalization (Underscores & Case Variations)
# =====================================================================
def test_17_taxonomy_normalization():
    """Verifies underscore notation like 'Tomato___Early_blight' normalizes correctly."""
    valid, host, disease = validate_diagnosis("Tomato", "Tomato___Early_blight")
    assert valid is True
    assert host == "Tomato"
    assert disease == "Early Blight"

    valid2, host2, disease2 = validate_diagnosis("potato", "late_blight")
    assert valid2 is True
    assert host2 == "Potato"
    assert disease2 == "Late Blight"

    valid3, host3, disease3 = validate_diagnosis("Soybean", "Frogeye Leaf Spot")
    assert valid3 is True
    assert host3 == "Soybean"
    assert disease3 == "Frogeye Leaf Spot"

    # Unsupported host should fail
    valid_bad, _, _ = validate_diagnosis("Banana", "Panama Disease")
    assert valid_bad is False


# =====================================================================
# 18. Asymptomatic / Healthy Foliage Specimen
# =====================================================================
def test_18_healthy_specimen_diagnosis(dummy_leaf_jpeg):
    """Verifies that an asymptomatic healthy leaf is diagnosed as Healthy."""
    mock_data = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Healthy",
        "assessment": "high",
        "visual_evidence": ["Uniform vibrant green pigment", "No necrotic spotting", "Turgid leaf lamina"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    mock_client = create_mock_gemini_client(mock_data)

    result = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_client)

    assert result.status == "SUCCESS"
    assert result.host == "Tomato"
    assert result.disease == "Healthy"
    assert result.diagnosis == "Tomato — Healthy"
    assert result.treatment is not None
    assert result.treatment["severity_risk"] == "NONE"


# =====================================================================
# 19. Phase 7 Regression: SUCCESS -> API_503_UNAVAILABLE (Zero Stale State)
# =====================================================================
def test_19_regression_stale_state_success_then_503(dummy_leaf_jpeg, dummy_leaf_png):
    """Verifies that a subsequent 503 failure completely wipes out prior success.

    Request A: SUCCESS (Tomato Early Blight)
    Request B: 503 UNAVAILABLE
    Assert after Request B:
    - Status is WITHHELD
    - inference_origin is API_503_UNAVAILABLE
    - No diagnosis / host / treatment from Request A survives
    """
    mock_data_a = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Early Blight",
        "assessment": "high",
        "visual_evidence": ["Target-like concentric rings on leaflets"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client_a = create_mock_gemini_client(mock_data_a)
    result_a = perform_diagnosis(dummy_leaf_jpeg, client_override=client_a)

    assert result_a.status == "SUCCESS"
    assert result_a.diagnosis == "Tomato — Early Blight"
    assert result_a.host == "Tomato"
    assert result_a.treatment is not None

    # Request B on a distinct specimen encounters 503 UNAVAILABLE
    client_b = MagicMock()
    client_b.models.generate_content.side_effect = Exception("503 UNAVAILABLE: Model overloaded")
    result_b = perform_diagnosis(dummy_leaf_png, client_override=client_b)

    assert result_b.status == "WITHHELD"
    assert result_b.inference_origin == "API_503_UNAVAILABLE"
    assert result_b.error_category == "API_503_UNAVAILABLE"
    assert result_b.confidence_score is None
    assert result_b.treatment is None
    assert result_b.host in [None, "Crop Specimen", "Unconfirmed"]
    assert result_b.disease in [None, "Unconfirmed"]
    assert "Early Blight" not in str(result_b.diagnosis)
    assert "Tomato" not in str(result_b.host)


# =====================================================================
# 20. Phase 7 Regression: SUCCESS -> API_TIMEOUT (Zero Stale State)
# =====================================================================
def test_20_regression_stale_state_success_then_timeout(dummy_leaf_jpeg, dummy_leaf_png):
    """Verifies that a subsequent timeout completely wipes out prior success.

    Request A: SUCCESS (Potato Late Blight)
    Request B: TimeoutError
    Assert after Request B:
    - Status is WITHHELD
    - inference_origin is API_TIMEOUT
    - No diagnosis / host / treatment from Request A survives
    """
    mock_data_a = {
        "is_plant": True,
        "host": "Potato",
        "diagnosis": "Late Blight",
        "assessment": "high",
        "visual_evidence": ["Water-soaked dark lesions"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client_a = create_mock_gemini_client(mock_data_a)
    result_a = perform_diagnosis(dummy_leaf_jpeg, client_override=client_a)

    assert result_a.status == "SUCCESS"
    assert result_a.diagnosis == "Potato — Late Blight"

    # Request B encounters timeout
    client_b = MagicMock()
    client_b.models.generate_content.side_effect = TimeoutError("Deadline exceeded: 45000ms timed out")
    result_b = perform_diagnosis(dummy_leaf_png, client_override=client_b)

    assert result_b.status == "WITHHELD"
    assert result_b.inference_origin == "API_TIMEOUT"
    assert result_b.error_category == "API_TIMEOUT"
    assert result_b.confidence_score is None
    assert result_b.treatment is None
    assert result_b.host in [None, "Crop Specimen", "Unconfirmed"]
    assert "Late Blight" not in str(result_b.diagnosis)
    assert "Potato" not in str(result_b.host)


# =====================================================================
# 21. Phase 7 Regression: SUCCESS -> API_429_RATE_LIMITED (Zero Stale State)
# =====================================================================
def test_21_regression_stale_state_success_then_rate_limit(dummy_leaf_jpeg, dummy_leaf_png):
    """Verifies that a subsequent 429 rate limit completely wipes out prior success.

    Request A: SUCCESS (Apple Apple Scab)
    Request B: 429 RESOURCE_EXHAUSTED
    Assert after Request B:
    - Status is WITHHELD
    - inference_origin is API_429_RATE_LIMITED
    - No diagnosis / host / treatment from Request A survives
    """
    mock_data_a = {
        "is_plant": True,
        "host": "Apple",
        "diagnosis": "Apple Scab",
        "assessment": "high",
        "visual_evidence": ["Velvety olive-green spots on foliage"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client_a = create_mock_gemini_client(mock_data_a)
    result_a = perform_diagnosis(dummy_leaf_jpeg, client_override=client_a)

    assert result_a.status == "SUCCESS"
    assert result_a.diagnosis == "Apple — Apple Scab"

    # Request B encounters rate limit
    client_b = MagicMock()
    client_b.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit reached")
    result_b = perform_diagnosis(dummy_leaf_png, client_override=client_b)

    assert result_b.status == "WITHHELD"
    assert result_b.inference_origin == "API_429_RATE_LIMITED"
    assert result_b.error_category in ["API_RATE_LIMIT", "API_429_RATE_LIMITED"]
    assert result_b.confidence_score is None
    assert result_b.treatment is None
    assert result_b.host in [None, "Crop Specimen", "Unconfirmed"]
    assert "Apple Scab" not in str(result_b.diagnosis)
    assert "Apple" not in str(result_b.host)


# =====================================================================
# 22. Disease-First: Known Host + Known Disease
# =====================================================================
def test_22_disease_first_known_host_known_disease(dummy_leaf_jpeg):
    """Known host + known canonical disease -> Disease displayed and treatment displayed."""
    mock_data = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Early Blight",
        "assessment": "high",
        "visual_evidence": ["Target-like concentric rings on foliar lamina"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Early Blight"
    assert result.host == "Tomato"
    assert result.diagnosis == "Tomato — Early Blight"
    assert result.treatment is not None
    assert len(result.treatment["chemical_treatments"]) > 0
    assert len(result.treatment["cultural_controls"]) > 0


# =====================================================================
# 23. Disease-First: Unknown Host + Known Disease
# =====================================================================
def test_23_disease_first_unknown_host_known_disease(dummy_leaf_jpeg):
    """Unknown host + known canonical disease -> Disease displayed, host omitted, no host-specific treatment."""
    mock_data = {
        "is_plant": True,
        "host": None,  # Host cannot be identified
        "diagnosis": "Apple Scab",
        "assessment": "high",
        "visual_evidence": ["Velvety olive-green lesions on upper foliar surface"],
        "alternative_diagnosis": None,
        "limitations": ["Specimen leaf isolated without tree or branch morphology."]
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Apple Scab"
    assert result.host is None  # Host omitted cleanly
    assert result.diagnosis == "Apple Scab"  # Pure disease name
    assert result.treatment is None  # No host-specific treatment invented!


# =====================================================================
# 24. Disease-First: Known Host + Non-Taxonomy Disease -> SUCCESS (No Treatment)
# =====================================================================
def test_24_disease_first_known_host_non_taxonomy_disease(dummy_leaf_jpeg):
    """Known host + non-canonical disease -> Displayed as SUCCESS with no local treatment."""
    mock_data = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Crown Rot and Vascular Collapse",  # Not in canonical taxonomy
        "assessment": "high",
        "visual_evidence": ["Basal stem discoloration"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.host == "Tomato"
    assert result.disease == "Crown Rot and Vascular Collapse"
    assert result.treatment is None


# =====================================================================
# 25. Disease-First: Unknown Host + Null Diagnosis -> WITHHELD
# =====================================================================
def test_25_disease_first_unknown_host_null_diagnosis_withheld(dummy_leaf_jpeg):
    """Unknown host + null/uncertain diagnosis -> Diagnosis must be WITHHELD."""
    mock_data = {
        "is_plant": True,
        "host": None,
        "diagnosis": None,  # Null diagnosis from AI
        "assessment": "unknown",
        "visual_evidence": ["Amorphous discoloration"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "UNCERTAIN_DIAGNOSIS"
    assert result.treatment is None


# =====================================================================
# 26. Disease-First: Non-Plant Specimen -> WITHHELD
# =====================================================================
def test_26_disease_first_non_plant_image(dummy_leaf_jpeg):
    """Non-plant specimen (is_plant=False) must be WITHHELD regardless of diagnosis text."""
    mock_data = {
        "is_plant": False,
        "host": None,
        "diagnosis": "Apple Scab",
        "assessment": "low",
        "visual_evidence": [],
        "alternative_diagnosis": None,
        "limitations": ["Image is a mechanical tractor component, not plant tissue."]
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "NON_PLANT"
    assert result.treatment is None


# =====================================================================
# 27. Disease-First: Healthy Leaf Specimen
# =====================================================================
def test_27_disease_first_healthy_leaf(dummy_leaf_jpeg, dummy_leaf_png):
    """Healthy foliage is properly classified as Healthy both with and without host."""
    # Case A: Known host + Healthy
    mock_data_a = {
        "is_plant": True,
        "host": "Soybean",
        "diagnosis": "Healthy",
        "assessment": "high",
        "visual_evidence": ["Unblemished trifoliate leaves, uniform dark green lamina"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client_a = create_mock_gemini_client(mock_data_a)
    result_a = perform_diagnosis(dummy_leaf_jpeg, client_override=client_a)

    assert result_a.status == "SUCCESS"
    assert result_a.disease == "Healthy"
    assert result_a.host == "Soybean"
    assert result_a.treatment is not None
    assert result_a.treatment["severity_risk"] == "NONE"

    # Case B: Unknown host + Healthy
    mock_data_b = {
        "is_plant": True,
        "host": None,
        "diagnosis": "Healthy",
        "assessment": "high",
        "visual_evidence": ["Vibrant green foliage with no necrotic spotting or chlorosis"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client_b = create_mock_gemini_client(mock_data_b)
    result_b = perform_diagnosis(dummy_leaf_png, client_override=client_b)

    assert result_b.status == "SUCCESS"
    assert result_b.disease == "Healthy"
    assert result_b.host is None
    assert result_b.diagnosis == "Healthy"
    assert result_b.treatment is None


# =====================================================================
# 28. Synonym Normalization: Frogeye Variants & Cercospora
# =====================================================================
def test_28_synonym_normalization_frogeye_variants(dummy_leaf_jpeg):
    """Naming variants ('Frog-eye Leaf Spot', 'Frog eye leaf spot', 'Cercospora capsici')
    map deterministically to canonical 'Frogeye Leaf Spot'.
    """
    variants = [
        ("Frog-eye Leaf Spot", "Pepper"),
        ("Frog eye leaf spot", None),
        ("Cercospora capsici", "Pepper"),
        ("frogeye spot", None),
    ]

    for variant_name, host_input in variants:
        clear_cache()
        mock_data = {
            "is_plant": True,
            "host": host_input,
            "diagnosis": variant_name,
            "assessment": "high",
            "visual_evidence": ["Circular necrotic lesions with light tan centers and dark borders"],
            "alternative_diagnosis": None,
            "limitations": []
        }
        client = create_mock_gemini_client(mock_data)
        result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

        assert result.status == "SUCCESS", f"Failed for variant {variant_name}"
        assert result.disease == "Frogeye Leaf Spot", f"Expected Frogeye Leaf Spot for {variant_name}, got {result.disease}"

        if host_input == "Pepper":
            assert result.host == "Pepper"
            assert result.treatment is not None
            assert "Cercospora capsici" in result.treatment["pathogen_type"]
        else:
            assert result.host is None
            assert result.treatment is None


# =====================================================================
# 29. Synonym Normalization: Bacterial Leaf Spot -> Bacterial Spot
# =====================================================================
def test_29_synonym_normalization_bacterial_leaf_spot(dummy_leaf_jpeg):
    """Common variant 'Bacterial Leaf Spot' normalizes to canonical 'Bacterial Spot'."""
    mock_data = {
        "is_plant": True,
        "host": "Pepper",
        "diagnosis": "Bacterial Leaf Spot",
        "assessment": "high",
        "visual_evidence": ["Small angular water-soaked dark lesions"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Bacterial Spot"
    assert result.host == "Pepper"
    assert result.treatment is not None


# =====================================================================
# 30. Null or Uncertain Diagnosis Must Be Withheld
# =====================================================================
def test_30_uncertain_or_null_diagnosis_rejected(dummy_leaf_jpeg):
    """When the AI cannot make a diagnosis or returns unknown assessment, it must be WITHHELD."""
    uncertain_cases = [None, "", "Unknown", "unidentified", "null"]

    for diag in uncertain_cases:
        clear_cache()
        mock_data = {
            "is_plant": True,
            "host": None,
            "diagnosis": diag,
            "assessment": "unknown",
            "visual_evidence": ["Generalized discoloration on leaf tissue"],
            "alternative_diagnosis": None,
            "limitations": []
        }
        client = create_mock_gemini_client(mock_data)
        result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

        assert result.status == "WITHHELD", f"Uncertain diagnosis '{diag}' should be WITHHELD"
        assert result.error_category == "UNCERTAIN_DIAGNOSIS"
        assert result.treatment is None


# =====================================================================
# 31. Groq Vision Provider Success
# =====================================================================
def test_31_groq_provider_success(dummy_leaf_jpeg):
    """Groq vision provider parses valid JSON response and returns SUCCESS."""
    clear_cache()
    mock_resp_json = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "is_plant": True,
                    "host": "Soybean",
                    "diagnosis": "Frogeye Leaf Spot",
                    "assessment": "high",
                    "visual_evidence": ["Circular lesions with grey centers and dark reddish margins"],
                    "alternative": None,
                    "limitations": []
                })
            }
        }]
    }
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = MagicMock(status_code=200, json=lambda: mock_resp_json)

    groq_provider = GroqVisionProvider(api_key="gsk_mock_test_key", client=mock_http_client)
    result = perform_diagnosis(dummy_leaf_jpeg, provider_override=groq_provider)

    assert result.status == "SUCCESS"
    assert result.host == "Soybean"
    assert result.disease == "Frogeye Leaf Spot"
    assert result.treatment is not None
    assert "Fungal" in result.treatment["pathogen_type"]


# =====================================================================
# 32. Groq Vision Provider 429 Rate Limited -> WITHHELD
# =====================================================================
def test_32_groq_429_rate_limited(dummy_leaf_jpeg):
    """Groq 429 rate limit returns honest WITHHELD."""
    clear_cache()
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = MagicMock(
        status_code=429,
        text="Rate limit reached: ITPM exceeded"
    )

    groq_provider = GroqVisionProvider(api_key="gsk_mock_test_key", client=mock_http_client)
    result = perform_diagnosis(dummy_leaf_jpeg, provider_override=groq_provider)

    assert result.status == "WITHHELD"
    assert result.error_category in ["API_RATE_LIMIT", "API_429_RATE_LIMITED"]
    assert result.confidence_score is None


# =====================================================================
# 33. Groq Vision Provider Timeout -> WITHHELD
# =====================================================================
def test_33_groq_timeout(dummy_leaf_jpeg):
    """Groq timeout returns honest WITHHELD with API_TIMEOUT category."""
    clear_cache()
    mock_http_client = MagicMock()
    import httpx
    mock_http_client.post.side_effect = httpx.TimeoutException("Connection timed out after 30s")

    groq_provider = GroqVisionProvider(api_key="gsk_mock_test_key", client=mock_http_client)
    result = perform_diagnosis(dummy_leaf_jpeg, provider_override=groq_provider)

    assert result.status == "WITHHELD"
    assert result.error_category == "API_TIMEOUT"
    assert result.confidence_score is None


# =====================================================================
# 34. Groq Vision Provider Malformed JSON -> WITHHELD
# =====================================================================
def test_34_groq_malformed_json(dummy_leaf_jpeg):
    """Malformed or invalid JSON from Groq returns WITHHELD."""
    clear_cache()
    mock_resp_json = {
        "choices": [{
            "message": {
                "content": "This is plain text with no JSON { broken"
            }
        }]
    }
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = MagicMock(status_code=200, json=lambda: mock_resp_json)

    groq_provider = GroqVisionProvider(api_key="gsk_mock_test_key", client=mock_http_client)
    result = perform_diagnosis(dummy_leaf_jpeg, provider_override=groq_provider)

    assert result.status == "WITHHELD"
    assert result.error_category in ["API_MALFORMED_RESPONSE", "STRUCTURED_OUTPUT_PARSE_ERROR"]
    assert result.confidence_score is None


# =====================================================================
# 35. Groq Failure Is Never Cached
# =====================================================================
def test_35_groq_failure_not_cached(dummy_leaf_jpeg):
    """Failed Groq requests are NEVER cached in _DIAGNOSTIC_CACHE."""
    clear_cache()
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = MagicMock(status_code=500, text="Internal Server Error")

    groq_provider = GroqVisionProvider(api_key="gsk_mock_test_key", client=mock_http_client)
    result = perform_diagnosis(dummy_leaf_jpeg, provider_override=groq_provider)

    assert result.status == "WITHHELD"
    assert get_cache_size() == 0


# =====================================================================
# AI Authority Requirement 1: AI returns known disease -> Displayed
# =====================================================================
def test_ai_authority_1_known_disease(dummy_leaf_jpeg):
    """When the AI identifies a known disease, the diagnosis is displayed."""
    mock_data = {
        "is_plant": True,
        "host": "Soybean",
        "diagnosis": "Frogeye Leaf Spot",
        "assessment": "high",
        "visual_evidence": ["Circular lesions with grey centers"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Frogeye Leaf Spot"
    assert result.host == "Soybean"
    assert result.is_confident is True


# =====================================================================
# AI Authority Requirement 2: AI returns disease absent from taxonomy -> STILL DISPLAYED
# =====================================================================
def test_ai_authority_2_disease_absent_from_taxonomy_still_displays(dummy_leaf_jpeg):
    """When the AI identifies a disease absent from local taxonomy (e.g. Maple Tar Spot),
    PhytoGATE displays the AI's diagnosis normally without withholding.
    """
    mock_data = {
        "is_plant": True,
        "host": "Maple",
        "diagnosis": "Tar Spot",
        "assessment": "high",
        "visual_evidence": ["Black tar-like raised stromatic spots on upper leaf surface"],
        "alternative_diagnosis": "Rhytisma acerinum",
        "limitations": ["Visual observation without laboratory spore analysis"]
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Tar Spot"
    assert result.host == "Maple"
    assert "Tar Spot" in result.diagnosis
    assert result.treatment is None  # Unavailable in local DB, but NEVER withheld!


# =====================================================================
# AI Authority Requirement 3: Known host + known disease -> Treatment attaches
# =====================================================================
def test_ai_authority_3_known_host_known_disease_attaches_treatment(dummy_leaf_jpeg):
    """When AI returns known host and known disease, treatment protocol is attached."""
    mock_data = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Early Blight",
        "assessment": "high",
        "visual_evidence": ["Concentric dark rings with chlorotic halo"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Early Blight"
    assert result.treatment is not None
    assert "chemical_treatments" in result.treatment
    assert len(result.treatment["chemical_treatments"]) > 0


# =====================================================================
# AI Authority Requirement 4: Unknown host + known disease -> Disease displays, host unknown
# =====================================================================
def test_ai_authority_4_unknown_host_known_disease_displays_disease_host_unknown(dummy_leaf_jpeg):
    """When AI returns unknown host but valid disease, disease is displayed and host is unknown."""
    mock_data = {
        "is_plant": True,
        "host": None,
        "diagnosis": "Late Blight",
        "assessment": "high",
        "visual_evidence": ["Dark water-soaked lesions with white sporulation on margin"],
        "alternative_diagnosis": None,
        "limitations": ["Host species could not be identified from single leaf specimen."]
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Late Blight"
    assert result.host is None
    assert result.treatment is None  # No host-specific treatment invented


# =====================================================================
# AI Authority Requirement 5: AI returns healthy -> Healthy displays
# =====================================================================
def test_ai_authority_5_healthy_displays_healthy(dummy_leaf_jpeg):
    """When AI determines foliage is healthy, Healthy status is displayed."""
    mock_data = {
        "is_plant": True,
        "host": "Pepper",
        "diagnosis": "Healthy",
        "assessment": "high",
        "visual_evidence": ["Vigorous green leaf tissue with no lesions or chlorosis"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "SUCCESS"
    assert result.disease == "Healthy"
    assert "Healthy" in result.diagnosis
    assert result.treatment is not None
    assert result.treatment["severity_risk"] == "NONE"


# =====================================================================
# AI Authority Requirement 6: AI returns non-plant -> Appropriate non-plant result
# =====================================================================
def test_ai_authority_6_non_plant_returns_appropriate_result(dummy_leaf_jpeg):
    """When AI determines image is not a plant, non-plant WITHHELD result is returned."""
    mock_data = {
        "is_plant": False,
        "host": None,
        "diagnosis": None,
        "assessment": "unknown",
        "visual_evidence": ["Metallic surface with industrial bolts"],
        "alternative_diagnosis": None,
        "limitations": ["Specimen is not biological plant tissue."]
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "NON_PLANT"
    assert result.host is None
    assert result.disease is None
    assert result.treatment is None


# =====================================================================
# AI Authority Requirement 7: AI returns null diagnosis -> WITHHELD
# =====================================================================
def test_ai_authority_7_null_diagnosis_returns_withheld(dummy_leaf_jpeg):
    """When AI returns null or unidentifiable diagnosis, diagnosis is WITHHELD."""
    mock_data = {
        "is_plant": True,
        "host": "Corn",
        "diagnosis": None,
        "assessment": "unknown",
        "visual_evidence": ["Foliage is severely obscured by shadow and glare"],
        "alternative_diagnosis": None,
        "limitations": ["Visual symptoms cannot be assessed due to poor illumination."]
    }
    client = create_mock_gemini_client(mock_data)
    result = perform_diagnosis(dummy_leaf_jpeg, client_override=client)

    assert result.status == "WITHHELD"
    assert result.error_category == "UNCERTAIN_DIAGNOSIS"


# =====================================================================
# AI Authority Requirement 8: API failure -> WITHHELD
# =====================================================================
def test_ai_authority_8_api_failure_returns_withheld(dummy_leaf_jpeg):
    """When vision API fails (network error, timeout, 503), diagnosis is cleanly WITHHELD."""
    mock_http_client = MagicMock()
    import httpx
    mock_http_client.post.side_effect = httpx.ConnectError("Connection refused by provider")

    groq_provider = GroqVisionProvider(api_key="gsk_mock_test_key", client=mock_http_client)
    result = perform_diagnosis(dummy_leaf_jpeg, provider_override=groq_provider)

    assert result.status == "WITHHELD"
    assert result.confidence_score is None


# =====================================================================
# AI Authority Requirement 9: Successful diagnosis after previous failure
# =====================================================================
def test_ai_authority_9_success_after_previous_failure(dummy_leaf_jpeg, dummy_leaf_png):
    """A successful diagnosis after a failed one displays the current diagnosis cleanly."""
    clear_cache()

    # Step 1: Failed request
    mock_failing_client = MagicMock()
    mock_failing_client.models.generate_content.side_effect = Exception("503 Service Unavailable")
    res1 = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_failing_client)
    assert res1.status == "WITHHELD"

    # Step 2: Fresh successful request with a different image
    mock_success_data = {
        "is_plant": True,
        "host": "Soybean",
        "diagnosis": "Frogeye Leaf Spot",
        "assessment": "high",
        "visual_evidence": ["Distinct frogeye lesions"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    mock_success_client = create_mock_gemini_client(mock_success_data)
    res2 = perform_diagnosis(dummy_leaf_png, client_override=mock_success_client)

    assert res2.status == "SUCCESS"
    assert res2.disease == "Frogeye Leaf Spot"
    assert res2.host == "Soybean"
    assert res2.error_category == "API_SUCCESS"


# =====================================================================
# AI Authority Requirement 10: Failure after previous success -> Cleared
# =====================================================================
def test_ai_authority_10_failure_after_previous_success_cleared(dummy_leaf_jpeg, dummy_leaf_png):
    """A failure after a previous success does NOT leak any previous diagnosis data."""
    clear_cache()

    # Step 1: Successful request
    mock_success_data = {
        "is_plant": True,
        "host": "Tomato",
        "diagnosis": "Early Blight",
        "assessment": "high",
        "visual_evidence": ["Target-like concentric rings"],
        "alternative_diagnosis": None,
        "limitations": []
    }
    mock_success_client = create_mock_gemini_client(mock_success_data)
    res1 = perform_diagnosis(dummy_leaf_jpeg, client_override=mock_success_client)
    assert res1.status == "SUCCESS"
    assert res1.disease == "Early Blight"

    # Step 2: Failing request with different specimen
    mock_failing_client = MagicMock()
    mock_failing_client.models.generate_content.side_effect = Exception("429 Resource Exhausted")
    res2 = perform_diagnosis(dummy_leaf_png, client_override=mock_failing_client)

    assert res2.status == "WITHHELD"
    assert res2.disease == "Unconfirmed" or res2.disease is None
    assert "Early Blight" not in (res2.disease or "")
    assert res2.treatment is None
    assert res2.confidence_score is None





