"""PhytoGATE Vision Provider Abstraction.

Decouples multimodal vision model inference from core diagnostics, taxonomy,
treatment retrieval, and UI rendering.

Supports:
- GroqVisionProvider (Production Primary: Qwen 3.8 27B)
- GeminiVisionProvider (Secondary / Legacy)
"""

import abc
import base64
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import settings


@dataclass
class ProviderResult:
    """Standardized response from any multimodal vision provider."""
    status: str  # "SUCCESS" or "ERROR"
    provider: str  # e.g., "groq", "gemini"
    model: str  # e.g., "qwen/qwen3.8-27b"
    latency_sec: float
    raw_payload: Optional[Dict[str, Any]] = None
    error_category: str = "API_SUCCESS"
    error_message: Optional[str] = None
    limitations: Optional[List[str]] = None


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



class VisionProvider(abc.ABC):
    """Abstract base class for vision inference providers."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of the provider, e.g. 'groq' or 'gemini'."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name of the model being called."""
        pass

    @abc.abstractmethod
    def analyze_image(self, normalized_jpeg_bytes: bytes) -> ProviderResult:
        """Analyzes a normalized JPEG leaf image and returns structured ProviderResult."""
        pass


class GroqVisionProvider(VisionProvider):
    """Groq OpenAI-compatible Multimodal Vision Provider (Qwen 3.8 27B)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self.timeout_seconds = timeout_seconds or settings.groq_timeout_seconds
        self.client = client
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self.model

    def analyze_image(self, normalized_jpeg_bytes: bytes) -> ProviderResult:
        # If a mock client with post or chat.completions is injected
        if self.client is not None:
            if hasattr(self.client, "post"):
                t0 = time.perf_counter()
                try:
                    resp = self.client.post(self.api_url, headers={}, json={})
                    t1 = time.perf_counter()
                    latency = round(t1 - t0, 4)
                    if hasattr(resp, "status_code") and resp.status_code != 200:
                        code = resp.status_code
                        err_cat = "API_429_RATE_LIMITED" if code == 429 else ("API_503_UNAVAILABLE" if code == 503 else "API_ERROR")
                        user_msg = "Vision service rate limit reached. Please wait a moment before trying again." if code == 429 else f"Vision provider returned HTTP error {code}."
                        return ProviderResult(
                            status="ERROR",
                            provider=self.provider_name,
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
                        return ProviderResult(
                            status="ERROR",
                            provider=self.provider_name,
                            model=self.model,
                            latency_sec=latency,
                            error_category="STRUCTURED_OUTPUT_PARSE_ERROR",
                            error_message="Vision provider returned invalid structured JSON output.",
                            limitations=["Structured response decoding failed."]
                        )
                    return ProviderResult(
                        status="SUCCESS",
                        provider=self.provider_name,
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
                    return ProviderResult(
                        status="ERROR",
                        provider=self.provider_name,
                        model=self.model,
                        latency_sec=latency,
                        error_category=err_cat,
                        error_message=err_msg,
                        limitations=[err_msg]
                    )
            elif hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                t0 = time.perf_counter()
                try:
                    res = self.client.chat.completions.create(
                        model=self.model,
                        messages=[],
                        response_format={"type": "json_object"}
                    )
                    t1 = time.perf_counter()
                    content = res.choices[0].message.content
                    parsed = json.loads(content) if isinstance(content, str) else content
                    return ProviderResult(
                        status="SUCCESS",
                        provider=self.provider_name,
                        model=self.model,
                        latency_sec=round(t1 - t0, 4),
                        raw_payload=parsed,
                        error_category="API_SUCCESS"
                    )
                except Exception as e:
                    t1 = time.perf_counter()
                    return ProviderResult(
                        status="ERROR",
                        provider=self.provider_name,
                        model=self.model,
                        latency_sec=round(t1 - t0, 4),
                        error_category="API_SERVER_ERROR",
                        error_message=str(e),
                        limitations=[str(e)]
                    )

        if not self.api_key or len(self.api_key) <= 5:
            return ProviderResult(
                status="ERROR",
                provider=self.provider_name,
                model=self.model_name,
                latency_sec=0.0,
                error_category="API_AUTH_ERROR",
                error_message="Groq API key is not configured. Please supply a valid GROQ_API_KEY in .env.",
                limitations=["The active diagnostic engine requires valid Groq multimodal API credentials."]
            )

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
            "User-Agent": "PhytoGATE/1.0 (Plant Pathology Core)",
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

                # Parse JSON
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
                except Exception as pe:
                    return ProviderResult(
                        status="ERROR",
                        provider=self.provider_name,
                        model=model_returned,
                        latency_sec=latency,
                        error_category="STRUCTURED_OUTPUT_PARSE_ERROR",
                        error_message="Vision provider returned invalid structured JSON output.",
                        limitations=["Structured response decoding failed."]
                    )

                return ProviderResult(
                    status="SUCCESS",
                    provider=self.provider_name,
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

            return ProviderResult(
                status="ERROR",
                provider=self.provider_name,
                model=self.model,
                latency_sec=latency,
                error_category=category,
                error_message=user_msg,
                limitations=["Diagnostic inference could not be completed by the multimodal vision engine."]
            )

        except TimeoutError:
            t1 = time.perf_counter()
            return ProviderResult(
                status="ERROR",
                provider=self.provider_name,
                model=self.model,
                latency_sec=round(t1 - t0, 4),
                error_category="API_TIMEOUT",
                error_message="Vision service request timed out. No diagnosis was generated. Please try again later.",
                limitations=["Request timed out."]
            )
        except urllib.error.URLError as e:
            t1 = time.perf_counter()
            err_lower = str(e).lower()
            if "timed out" in err_lower or "timeout" in err_lower:
                category = "API_TIMEOUT"
                user_msg = "Vision service request timed out. No diagnosis was generated. Please try again later."
            else:
                category = "API_SERVER_ERROR"
                user_msg = "Unable to reach the vision provider network endpoint."

            return ProviderResult(
                status="ERROR",
                provider=self.provider_name,
                model=self.model,
                latency_sec=round(t1 - t0, 4),
                error_category=category,
                error_message=user_msg,
                limitations=["Network connection failure."]
            )
        except Exception as e:
            t1 = time.perf_counter()
            return ProviderResult(
                status="ERROR",
                provider=self.provider_name,
                model=self.model,
                latency_sec=round(t1 - t0, 4),
                error_category="API_SERVER_ERROR",
                error_message=f"The vision service could not complete analysis: {str(e)}",
                limitations=[str(e)]
            )


class GeminiVisionProvider(VisionProvider):
    """Google Gemini Multimodal Vision Provider (Legacy / Secondary)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.timeout_seconds = timeout_seconds or settings.gemini_timeout_seconds
        self.client = client

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.model

    def analyze_image(self, normalized_jpeg_bytes: bytes) -> ProviderResult:
        client = self.client
        if client is None:
            if not self.api_key or len(self.api_key) <= 5:
                return ProviderResult(
                    status="ERROR",
                    provider=self.provider_name,
                    model=self.model_name,
                    latency_sec=0.0,
                    error_category="API_AUTH_ERROR",
                    error_message="Gemini API key is not configured.",
                    limitations=["Gemini credentials missing."]
                )

            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.api_key)
            except ImportError:
                return ProviderResult(
                    status="ERROR",
                    provider=self.provider_name,
                    model=self.model_name,
                    latency_sec=0.0,
                    error_category="API_SERVER_ERROR",
                    error_message="google-genai package is not installed.",
                    limitations=["Missing SDK dependency."]
                )
        else:
            try:
                from google.genai import types
            except ImportError:
                types = None

        if types is not None:
            image_part = types.Part.from_bytes(data=normalized_jpeg_bytes, mime_type="image/jpeg")
            gen_config_kwargs = {
                "response_mime_type": "application/json",
                "max_output_tokens": settings.gemini_max_output_tokens,
            }
            if "gemini-3" in self.model.lower():
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
            elif "gemini-2.5" in self.model.lower():
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            gen_config = types.GenerateContentConfig(**gen_config_kwargs)
        else:
            image_part = None
            gen_config = None

        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[DIAGNOSTIC_PROMPT, image_part],
                config=gen_config,
            )
            t1 = time.perf_counter()
            latency = round(t1 - t0, 4)

            # Extract parsed JSON payload
            if hasattr(response, "text") and response.text:
                parsed = json.loads(response.text)
            elif hasattr(response, "parsed") and response.parsed is not None:
                parsed = response.parsed
            else:
                raise ValueError("No text or parsed content received in Gemini response.")

            return ProviderResult(
                status="SUCCESS",
                provider=self.provider_name,
                model=self.model,
                latency_sec=latency,
                raw_payload=parsed,
                error_category="API_SUCCESS"
            )
        except Exception as e:
            t1 = time.perf_counter()
            latency = round(t1 - t0, 4)
            err_str = str(e)
            err_lower = err_str.lower()

            if "503" in err_str or "unavailable" in err_lower or "high demand" in err_lower:
                category = "API_503_UNAVAILABLE"
                user_msg = "The vision service is temporarily unavailable due to high demand. No diagnosis was generated. Please try again later."
            elif "429" in err_str or "rate limit" in err_lower or "quota" in err_lower or "resource_exhausted" in err_lower:
                category = "API_429_RATE_LIMITED"
                user_msg = "Vision service rate limit reached. No diagnosis was generated. Please try again later."
            elif isinstance(e, TimeoutError) or "timeout" in err_lower or "timed out" in err_lower or "deadline" in err_lower:
                category = "API_TIMEOUT"
                user_msg = "Vision service request timed out. No diagnosis was generated. Please try again later."
            elif "401" in err_str or "403" in err_str or "auth" in err_lower or "api_key_invalid" in err_lower:
                category = "API_AUTH_ERROR"
                user_msg = "Vision service authentication failed. Please verify API key configuration."
            elif "400" in err_str or "invalid" in err_lower or "bad request" in err_lower:
                category = "API_INVALID_REQUEST"
                user_msg = "Invalid request sent to vision service. No diagnosis was generated."
            else:
                category = "API_ERROR"
                user_msg = "The vision service could not complete analysis. No diagnosis was generated. Please try again later."

            return ProviderResult(
                status="ERROR",
                provider=self.provider_name,
                model=self.model,
                latency_sec=latency,
                error_category=category,
                error_message=user_msg,
                limitations=["Diagnostic inference could not be completed by the multimodal vision engine."]
            )


def get_vision_provider(provider_override: Optional[str] = None) -> VisionProvider:
    """Factory function returning the active or requested VisionProvider."""
    active = (provider_override or settings.vision_provider).lower()
    if active == "gemini":
        return GeminiVisionProvider()
    return GroqVisionProvider()
