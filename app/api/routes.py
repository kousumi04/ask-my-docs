"""
The two real endpoints of the service: ask a question, add a document.
Route handlers stay thin -- all real logic lives in app.core.pipeline
(and everything it delegates to), so these functions are barely more
than request validation and response shaping.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import QueryRequest, QueryResponse, UploadResponse
from app.config import settings
from app.core import pipeline
from app.ingestion.pipeline import SUPPORTED_SUFFIXES

router = APIRouter()

RAW_DATA_DIR = pipeline.PROJECT_ROOT / "data" / "raw"
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    result = pipeline.query(request.question, top_k_rerank=request.top_k)
    return QueryResponse(**result)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RAW_DATA_DIR / file.filename
    bytes_written = 0
    with dest_path.open("wb") as f:
        while chunk := file.file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                f.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum supported upload size is {settings.max_upload_mb} MB.",
                )
            f.write(chunk)

    result = pipeline.add_document(dest_path)
    return UploadResponse(**result)
