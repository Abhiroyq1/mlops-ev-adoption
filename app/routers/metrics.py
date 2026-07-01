from fastapi import APIRouter, HTTPException, Query
from app.services.metrics_service import compute_metrics
from app.schemas.responses import MetricsResponse
from app.ml.pipeline import SUPPORTED_MODELS

router = APIRouter(prefix="/metrics", tags=["Metrics"])


def _check_model_name(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"model_name '{model_name}' is not supported. Choose from: {list(SUPPORTED_MODELS)}",
        )


@router.get("/evaluate", response_model=MetricsResponse)
def evaluate(model_name: str = Query(default="random_forest", enum=list(SUPPORTED_MODELS))):
    _check_model_name(model_name)
    try:
        return compute_metrics(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
