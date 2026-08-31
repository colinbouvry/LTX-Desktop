"""Route handlers for /api/outputs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api_types import OutputsListResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["outputs"])


@router.get("/outputs", response_model=OutputsListResponse)
def route_list_outputs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    handler: AppHandler = Depends(get_state_service),
) -> OutputsListResponse:
    return handler.outputs.list_outputs(limit=limit, offset=offset)
