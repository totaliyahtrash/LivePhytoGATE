"""PhytoGATE Configuration Module.

Single-point configuration for Groq multimodal vision inference and runtime settings.
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
        # Groq configuration (Sole Vision Engine)
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        raw_model = os.getenv("GROQ_MODEL", "").strip()
        self.groq_model: str = raw_model if raw_model else "qwen/qwen3.8-27b"
        try:
            self.groq_timeout_seconds: int = int(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
        except ValueError:
            self.groq_timeout_seconds: int = 30

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
    def is_configured(self) -> bool:
        return self.is_groq_configured

    @property
    def is_active_provider_configured(self) -> bool:
        return self.is_groq_configured

    @property
    def vision_provider(self) -> str:
        return "groq"

    @property
    def active_model(self) -> str:
        return self.groq_model

    @property
    def api_key_status(self) -> str:
        """Returns safe status descriptor for telemetry (PRESENT / MISSING)."""
        return "PRESENT" if self.is_groq_configured else "MISSING"

    def safe_dict(self) -> dict:
        """Returns safe configuration dict excluding all secrets."""
        return {
            "provider": "groq",
            "model": self.groq_model,
            "is_configured": self.is_groq_configured,
            "api_key_status": self.api_key_status,
            "timeout_seconds": self.groq_timeout_seconds,
        }

    def __repr__(self) -> str:
        return f"Settings(provider='groq', model={self.groq_model!r}, is_configured={self.is_groq_configured})"


settings = Settings()
