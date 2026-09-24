"""PhytoGATE Server Application.

Exposes REST API and serves the diagnostic web console:
- GET  /
- POST /api/diagnose
- GET  /api/taxonomy
- GET  /api/disease/{host}/{disease}
- GET  /api/telemetry
- GET  /api/samples
- GET  /api/health
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.taxonomy import get_supported_taxonomy
from src.disease_db import get_disease_profile
from src.diagnostics import perform_diagnosis, get_telemetry_stats, DiagnosticResult

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "samples"

app = FastAPI(
    title="PhytoGATE Diagnostic Core",
    description="Multimodal Plant Pathology Vision Diagnosis System",
    version="2.0.0",
)

# Enable CORS for local testing/development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and samples directories
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the primary PhytoGATE diagnostic dashboard UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index HTML not found.")
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check():
    """Purely local readiness health check.

    CRITICAL ARCHITECTURAL GUARANTEE:
    This endpoint makes ZERO external AI provider calls.
    """
    return {"status": "healthy"}


@app.get("/api/telemetry")
async def get_telemetry():
    """Returns safe runtime telemetry and model configuration.

    Guarantees API keys are never exposed.
    """
    return get_telemetry_stats()


@app.get("/api/taxonomy")
async def get_taxonomy():
    """Returns the controlled canonical crop/disease taxonomy."""
    return get_supported_taxonomy()


@app.get("/api/disease/{host}/{disease}")
async def get_disease(host: str, disease: str):
    """Deterministically retrieves local agronomic profile for host and disease.

    Makes ZERO external API calls.
    """
    profile = get_disease_profile(host, disease)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for host '{host}' and disease '{disease}'."
        )
    return profile


@app.get("/api/samples")
async def list_sample_images() -> List[Dict[str, Any]]:
    """Returns metadata for built-in sample images available for rapid demonstration."""
    samples = []
    if SAMPLES_DIR.exists():
        for file in sorted(SAMPLES_DIR.glob("*.*")):
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                samples.append({
                    "filename": file.name,
                    "url": f"/samples/{file.name}",
                    "size_bytes": file.stat().st_size,
                })
    return samples


@app.post("/api/diagnose", response_model=DiagnosticResult)
async def diagnose_leaf(file: UploadFile = File(...)):
    """Primary plant disease diagnosis endpoint.

    Workflow:
    - Normalizes image to RGB JPEG
    - Checks deterministic SHA-256 cache
    - Evaluates via Groq multimodal vision
    - Enforces canonical taxonomy boundary
    - Retrieves agronomic management locally
    - Withholds diagnosis honestly upon any error
    """
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Execute diagnosis pipeline
        result = perform_diagnosis(
            raw_image_bytes=raw_bytes,
            filename=file.filename or "uploaded_specimen.jpg"
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        # Sanitize unexpected server exceptions to avoid leaking internals or credentials
        return DiagnosticResult(
            status="WITHHELD",
            is_confident=False,
            confidence_score=None,
            assessment="unknown",
            diagnosis="Diagnosis Withheld — Server Execution Error",
            host="Crop Specimen",
            disease="Unconfirmed",
            error_category="API_ERROR",
            inference_origin="API_ERROR",
            error_message="An internal server error occurred while processing the specimen.",
            limitations=["Internal server exception occurred during pipeline execution."]
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.host, port=settings.port, reload=False)
