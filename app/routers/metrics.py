from fastapi import APIRouter, HTTPException, Query
from app.services.metrics_service import compute_metrics
from app.schemas.responses import MetricsResponse
from app.ml.pipeline import SUPPORTED_MODELS

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/evaluate", response_model=MetricsResponse)
def evaluate(model_name: str = Query(default="random_forest", enum=list(SUPPORTED_MODELS))):
    try:
        return compute_metrics(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
