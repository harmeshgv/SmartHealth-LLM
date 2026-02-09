from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional

from app.core.metrics_tracker import metrics_tracker

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/summary")
async def metrics_summary():
    return metrics_tracker.summary()


@router.get("/runs")
async def metrics_runs(limit: int = Query(default=50, ge=1, le=500)):
    return {"runs": metrics_tracker.get_recent_runs(limit=limit)}


class SaveMetricsRequest(BaseModel):
    filepath: Optional[str] = None
    limit: int = 500


@router.post("/save-local")
async def metrics_save_local(data: SaveMetricsRequest):
    saved_path = metrics_tracker.save_to_local(filepath=data.filepath, limit=data.limit)
    return {"message": "Metrics saved locally", "filepath": saved_path}


@router.post("/reset")
async def metrics_reset():
    metrics_tracker.reset()
    return {"message": "Metrics reset complete"}
