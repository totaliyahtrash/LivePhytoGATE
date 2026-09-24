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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.taxonomy import get_supported_taxonomy
from src.disease_db import get_disease_profile
from src.diagnostics import perform_diagnosis, get_telemetry_stats, DiagnosticResult

BASE_DIR = Path(__file__).resolve().parent


def get_static_dir() -> Path:
    """Finds static assets directory across local and serverless environments."""
    candidates = [
        BASE_DIR / "static",
        Path.cwd() / "static",
        Path("/var/task/static"),
        Path(__file__).resolve().parent.parent / "static",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return BASE_DIR / "static"


def get_samples_dir() -> Path:
    """Finds samples directory across local and serverless environments."""
    candidates = [
        BASE_DIR / "samples",
        Path.cwd() / "samples",
        Path("/var/task/samples"),
        Path(__file__).resolve().parent.parent / "samples",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return BASE_DIR / "samples"


STATIC_DIR = get_static_dir()
SAMPLES_DIR = get_samples_dir()

app = FastAPI(
    title="PhytoGATE Diagnostic Core",
    description="Multimodal Plant Pathology Vision Diagnosis System",
    version="2.0.0",
)

# Enable CORS for local testing and Vercel cloud deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def vercel_path_rewrite_middleware(request, call_next):
    """Restores the original request URL path from Vercel proxy headers."""
    matched_path = request.headers.get("x-matched-path") or request.headers.get("x-invoke-path")
    if matched_path and not matched_path.endswith("index.py"):
        request.scope["path"] = matched_path.split("?")[0]
    elif request.scope.get("path") in ["/api/index.py", "/api/index"]:
        request.scope["path"] = "/"
    return await call_next(request)


# Mount static and samples directories if available
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_index():
    """Serves the primary PhytoGATE diagnostic dashboard UI."""
    s_dir = get_static_dir()
    index_path = s_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Index HTML not found.")


@app.get("/api")
@app.get("/api/")
async def api_root_info():
    """API informational root endpoint."""
    return {
        "status": "online",
        "service": "PhytoGATE Diagnostic Core",
        "provider": "groq",
        "model": settings.groq_model,
        "is_configured": settings.is_groq_configured,
    }


from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response


@app.get("/static/css/style.css")
@app.get("/css/style.css")
async def serve_style_css():
    target = get_static_dir() / "css" / "style.css"
    if target.exists():
        return Response(content=target.read_text(encoding="utf-8"), media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS not found")


@app.get("/static/js/app.js")
@app.get("/js/app.js")
async def serve_app_js():
    target = get_static_dir() / "js" / "app.js"
    if target.exists():
        return Response(content=target.read_text(encoding="utf-8"), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS not found")


@app.get("/static/{file_path:path}")
async def serve_static_file(file_path: str):
    """Explicit fallback handler for static assets on serverless runtimes."""
    s_dir = get_static_dir()
    target = s_dir / file_path
    if target.exists() and target.is_file():
        media_type = "text/css" if file_path.endswith(".css") else ("application/javascript" if file_path.endswith(".js") else None)
        return FileResponse(target, media_type=media_type)
    raise HTTPException(status_code=404, detail=f"Static asset '{file_path}' not found.")


@app.get("/api/health")
@app.get("/health")
async def health_check():
    """Purely local readiness health check.

    CRITICAL ARCHITECTURAL GUARANTEE:
    This endpoint makes ZERO external AI provider calls.
    """
    return {"status": "healthy"}


@app.get("/api/telemetry")
@app.get("/telemetry")
async def get_telemetry():
    """Returns safe runtime telemetry and model configuration.

    Guarantees API keys are never exposed.
    """
    return get_telemetry_stats()


@app.get("/api/taxonomy")
@app.get("/taxonomy")
async def get_taxonomy():
    """Returns the controlled canonical crop/disease taxonomy."""
    return get_supported_taxonomy()


@app.get("/api/disease/{host}/{disease}")
@app.get("/disease/{host}/{disease}")
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
@app.get("/samples-list")
async def list_sample_images() -> List[Dict[str, Any]]:
    """Returns metadata for built-in sample images available for rapid demonstration."""
    samples = []
    s_dir = get_samples_dir()
    if s_dir.exists():
        for file in sorted(s_dir.glob("*.*")):
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                samples.append({
                    "filename": file.name,
                    "url": f"/samples/{file.name}",
                    "size_bytes": file.stat().st_size,
                })
    return samples


@app.post("/api/diagnose", response_model=DiagnosticResult)
@app.post("/diagnose", response_model=DiagnosticResult)
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
