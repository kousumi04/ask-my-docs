"""
FastAPI application entrypoint.

Phase 1 goal: prove the skeleton runs. Ingestion, retrieval, and
generation routers get added in later phases and included here.

Phase 7: /query and /upload are now wired in via app.api.routes.
"""

from fastapi import FastAPI
from app.api.routes import router as api_router
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Hybrid-retrieval RAG system with citation enforcement.",
)
app.include_router(api_router)


@app.get("/health")
def health_check() -> dict:
    """Liveness probe — used by Docker/CI to confirm the service is up."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/")
def root() -> dict:
    return {"message": f"{settings.app_name} API — see /docs for the interactive Swagger UI."}