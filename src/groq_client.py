"""PhytoGATE Groq Client Module.

Unified multimodal vision client connecting directly to Groq (Qwen 3.8 27B).
Performs visual plant disease diagnosis with structured JSON schema output.
"""

import base64
import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger("phytogate.groq")

DIAGNOSTIC_PROMPT = (
    "You are an expert plant pathologist performing a complete visual diagnostic evaluation on an uploaded specimen image.\n\n"
    "Your instructions:\n"
    "1. Inspect the actual image pixels and determine whether a biological plant, leaf, crop, or botanical tissue is visible.\n"
    "2. If it is a plant, identify the host plant or crop when possible (or null if unidentifiable).\n"
    "3. Identify the specific disease or pathology visibly present on the foliage, or indicate 'Healthy' if the foliage is asymptomatic, or null if uncertain or not a plant.\n"
    "4. Provide a qualitative assessment of diagnostic certainty: 'high', 'medium', 'low', or 'unknown'.\n"
    "5. List all visible foliar symptoms and morphological evidence supporting your diagnosis (e.g. lesion shape, color, concentric rings, chlorotic halos, necrotic margins, sporulation).\n"
    "6. Provide a reasonable alternative differential diagnosis if foliar symptoms overlap with other conditions (or null).\n"
    "7. List diagnostic limitations or imaging factors preventing definitive identification.\n\n"
    "Critical constraints:\n"
    "- Perform all diagnosis solely from the visual evidence in the image.\n"
    "- Do NOT use or guess from any filename or client metadata.\n"
    "- Do NOT invent laboratory, culturing, or microscopic confirmation.\n"
    "- Do NOT claim certainty that the image cannot support.\n\n"
    "Return ONLY a valid JSON object conforming strictly to this format:\n"
    "{\n"
    '  "is_plant": true,\n'
    '  "host": "identified plant/crop or null",\n'
    '  "diagnosis": "disease identified from image or Healthy or null",\n'
    '  "assessment": "high | medium | low | unknown",\n'
    '  "visual_evidence": [\n'
    '    "everything visually relevant that supports the diagnosis"\n'
    '  ],\n'
    '  "alternative": "alternative diagnosis or null",\n'
    '  "limitations": [\n'
    '    "limitations of image-based diagnosis"\n'
    '  ]\n'
    "}"
)


@dataclass
class GroqResult:
    """Standardized response from the Groq vision model."""
    status: str  # "SUCCESS" or "ERROR"
    model: str
    latency_sec: float
    raw_payload: Optional[Dict[str, Any]] = None
    error_category: str = "API_SUCCESS"
    error_message: Optional[str] = None
    limitations: Optional[List[str]] = None


class GroqClient:
    """Production Groq multimodal vision client for Qwen 3.8 27B."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        http_client: Optional[Any] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model if model is not None else settings.groq_model
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.groq_timeout_seconds
        self.http_client = http_client
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def analyze_image(self, normalized_jpeg_bytes: bytes) -> GroqResult:
        """Submits normalized leaf image to Groq for multimodal diagnostic analysis."""
        # 1. Dependency injection for testing
        if self.http_client is not None:
            if hasattr(self.http_client, "post"):
                t0 = time.perf_counter()
                try:
                    resp = self.http_client.post(self.api_url, headers={}, json={})
                    t1 = time.perf_counter()
                    latency = round(t1 - t0, 4)
                    if hasattr(resp, "status_code") and resp.status_code != 200:
                        code = resp.status_code
                        err_cat = "API_429_RATE_LIMITED" if code == 429 else ("API_503_UNAVAILABLE" if code == 503 else "API_ERROR")
                        user_msg = "Vision service rate limit reached. Please wait a moment before trying again." if code == 429 else f"Vision service returned HTTP error {code}."
                        return GroqResult(
                            status="ERROR",
                            model=self.model,
                            latency_sec=latency,
                            error_category=err_cat,
                            error_message=user_msg,
                            limitations=[f"Groq API returned HTTP {code}"]
                        )
                    res_json = resp.json() if callable(resp.json) else resp.json
                    raw_text = res_json["choices"][0]["message"]["content"]
                    try:
                        parsed = json.loads(raw_text)
                    except Exception:
                        return GroqResult(
                            status="ERROR",
                            model=self.model,
                            latency_sec=latency,
                            error_category="API_MALFORMED_RESPONSE",
                            error_message="Vision provider returned invalid structured JSON output.",
                            limitations=["Structured response decoding failed."]
                        )
                    return GroqResult(
                        status="SUCCESS",
                        model=res_json.get("model", self.model),
                        latency_sec=latency,
                        raw_payload=parsed,
                        error_category="API_SUCCESS"
                    )
                except Exception as e:
                    t1 = time.perf_counter()
                    latency = round(t1 - t0, 4)
                    err_msg = str(e)
                    err_cat = "API_TIMEOUT" if ("timeout" in err_msg.lower() or "timed out" in err_msg.lower()) else "API_SERVER_ERROR"
                    return GroqResult(
                        status="ERROR",
                        model=self.model,
                        latency_sec=latency,
                        error_category=err_cat,
                        error_message=err_msg,
                        limitations=[err_msg]
                    )

        # 2. Check credentials
        if not self.api_key or len(self.api_key) <= 5:
            return GroqResult(
                status="ERROR",
                model=self.model,
                latency_sec=0.0,
                error_category="API_AUTH_ERROR",
                error_message="Groq API key is not configured. Please supply a valid GROQ_API_KEY in .env.",
                limitations=["The active diagnostic engine requires valid Groq API credentials."]
            )

        # 3. Build payload
        b64_img = base64.b64encode(normalized_jpeg_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_img}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DIAGNOSTIC_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PhytoGATE/2.0 (Plant Pathology Core)",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.api_url, data=data, headers=headers, method="POST")

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                t1 = time.perf_counter()
                latency = round(t1 - t0, 4)
                body = resp.read().decode("utf-8")
                res_json = json.loads(body)

                model_returned = res_json.get("model", self.model)
                raw_text = res_json["choices"][0]["message"]["content"]

                cleaned = raw_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                try:
                    parsed = json.loads(cleaned)
                except Exception:
                    return GroqResult(
                        status="ERROR",
                        model=model_returned,
                        latency_sec=latency,
                        error_category="API_MALFORMED_RESPONSE",
                        error_message="Vision provider returned invalid structured JSON output.",
                        limitations=["Structured response decoding failed."]
                    )

                return GroqResult(
                    status="SUCCESS",
                    model=model_returned,
                    latency_sec=latency,
                    raw_payload=parsed,
                    error_category="API_SUCCESS"
                )

        except urllib.error.HTTPError as e:
            t1 = time.perf_counter()
            latency = round(t1 - t0, 4)
            status_code = e.code

            if status_code == 429:
                category = "API_429_RATE_LIMITED"
                user_msg = "Vision service rate limit reached. Please wait a moment before trying again."
            elif status_code == 503:
                category = "API_503_UNAVAILABLE"
                user_msg = "The vision service is temporarily unavailable due to high demand. No diagnosis was generated. Please try again later."
            elif status_code in (401, 403):
                category = "API_AUTH_ERROR"
                user_msg = "Vision service authentication failed. Please verify API key configuration."
            else:
                category = "API_SERVER_ERROR"
                user_msg = f"Vision provider returned HTTP error {status_code}."

            return GroqResult(
                status="ERROR",
                model=self.model,
                latency_sec=latency,
                error_category=category,
                error_message=user_msg,
                limitations=["Diagnostic inference could not be completed by the multimodal vision engine."]
            )

        except (TimeoutError, urllib.error.URLError) as e:
            t1 = time.perf_counter()
            latency = round(t1 - t0, 4)
            err_str = str(e).lower()
            is_timeout = isinstance(e, TimeoutError) or "timed out" in err_str or "timeout" in err_str

            return GroqResult(
                status="ERROR",
                model=self.model,
                latency_sec=latency,
                error_category="API_TIMEOUT" if is_timeout else "API_ERROR",
                error_message="Vision service connection timed out." if is_timeout else f"Network communication error: {e}",
                limitations=["Request timed out while awaiting vision inference."] if is_timeout else ["Network error communicating with vision provider."]
            )
        except Exception as e:
            t1 = time.perf_counter()
            return GroqResult(
                status="ERROR",
                model=self.model,
                latency_sec=round(t1 - t0, 4),
                error_category="API_ERROR",
                error_message=str(e),
                limitations=[str(e)]
            )
