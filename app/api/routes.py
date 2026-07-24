"""
The two real endpoints of the service: ask a question, add a document.
Route handlers stay thin -- all real logic lives in app.core.pipeline
(and everything it delegates to), so these functions are barely more
than request validation and response shaping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import QueryRequest, QueryResponse, UploadResponse
from app.core import pipeline
from app.ingestion.pipeline import SUPPORTED_SUFFIXES

router = APIRouter()

RAW_DATA_DIR = pipeline.PROJECT_ROOT / "data" / "raw"


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
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = pipeline.add_document(dest_path)
    return UploadResponse(**result)