"""PhytoGATE Configuration Module.

Loads and sanitizes environment configuration for the diagnostic core.
Ensures API keys are never leaked to logs, representations, or frontend responses.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    def __init__(self):
        # Active Vision Provider selection: 'groq' or 'gemini'
        self.vision_provider: str = os.getenv("VISION_PROVIDER", "groq").strip().lower()

        # Groq configuration (Production Primary)
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
        try:
            self.groq_timeout_seconds: int = int(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
        except ValueError:
            self.groq_timeout_seconds: int = 30

        # Gemini configuration (Legacy / Secondary)
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        try:
            self.gemini_max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "500"))
        except ValueError:
            self.gemini_max_output_tokens: int = 500

        try:
            self.gemini_timeout_seconds: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "45"))
        except ValueError:
            self.gemini_timeout_seconds: int = 45

        try:
            self.gemini_max_retries: int = int(os.getenv("GEMINI_MAX_RETRIES", "0"))
        except ValueError:
            self.gemini_max_retries: int = 0

        self.host: str = os.getenv("HOST", "127.0.0.1").strip()
        try:
            self.port: int = int(os.getenv("PORT", "8000"))
        except ValueError:
            self.port: int = 8000

    @property
    def is_groq_configured(self) -> bool:
        """Checks if a valid, non-placeholder Groq API key is configured."""
        if not self.groq_api_key:
            return False
        if self.groq_api_key.upper() in ["YOUR_KEY_HERE", "YOUR_API_KEY", "NONE", "NULL", ""]:
            return False
        return len(self.groq_api_key) > 5

    @property
    def is_gemini_configured(self) -> bool:
        """Checks if a valid, non-placeholder Gemini API key is configured."""
        if not self.gemini_api_key:
            return False
        if self.gemini_api_key.upper() in ["YOUR_KEY_HERE", "YOUR_API_KEY", "NONE", "NULL", ""]:
            return False
        return len(self.gemini_api_key) > 5

    @property
    def is_active_provider_configured(self) -> bool:
        """Returns True if the actively selected provider has valid credentials."""
        if self.vision_provider == "groq":
            return self.is_groq_configured
        return self.is_gemini_configured

    @property
    def active_model(self) -> str:
        """Returns the model string for the actively selected provider."""
        if self.vision_provider == "groq":
            return self.groq_model
        return self.gemini_model

    @property
    def api_key_status(self) -> str:
        """Returns safe status descriptor for telemetry (PRESENT / MISSING)."""
        return "PRESENT" if self.is_active_provider_configured else "MISSING"

    def safe_dict(self) -> dict:
        """Returns safe configuration dict excluding all secrets."""
        return {
            "provider": self.vision_provider,
            "model": self.active_model,
            "is_configured": self.is_active_provider_configured,
            "api_key_status": self.api_key_status,
            "groq": {
                "model": self.groq_model,
                "is_configured": self.is_groq_configured,
                "timeout_seconds": self.groq_timeout_seconds,
            },
            "gemini": {
                "model": self.gemini_model,
                "is_configured": self.is_gemini_configured,
                "timeout_seconds": self.gemini_timeout_seconds,
            },
            # Backward compatibility fields for telemetry consumers
            "gemini_model": self.active_model,
            "gemini_max_output_tokens": self.gemini_max_output_tokens,
            "gemini_timeout_seconds": self.gemini_timeout_seconds,
            "gemini_max_retries": self.gemini_max_retries,
            "is_gemini_configured": self.is_active_provider_configured,
        }

    def __repr__(self) -> str:
        # Redact API keys from any debug printing
        return (
            f"Settings(provider={self.vision_provider!r}, "
            f"model={self.active_model!r}, "
            f"is_configured={self.is_active_provider_configured})"
        )


settings = Settings()
