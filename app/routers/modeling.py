from fastapi import APIRouter, HTTPException, Query
from app.services.model_service import train_model, predict, list_trained_models, cross_validate_model
from app.schemas.requests import PredictRequest
from app.schemas.responses import TrainResponse, PredictResponse, CrossValidationResponse
from app.ml.pipeline import SUPPORTED_MODELS

router = APIRouter(prefix="/model", tags=["Modeling"])


def _check_model_name(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"model_name '{model_name}' is not supported. Choose from: {list(SUPPORTED_MODELS)}",
        )


@router.post("/train", response_model=TrainResponse)
def train(model_name: str = Query(default="random_forest", enum=list(SUPPORTED_MODELS))):
    _check_model_name(model_name)
    try:
        return train_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
def run_predict(
    body: PredictRequest,
    model_name: str = Query(default="random_forest", enum=list(SUPPORTED_MODELS)),
):
    _check_model_name(model_name)
    try:
        return predict(model_name, body.features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
def get_trained_models() -> list[str]:
    return list_trained_models()


@router.post("/cross-validate", response_model=CrossValidationResponse)
def cross_validate(
    model_name: str = Query(default="random_forest", enum=list(SUPPORTED_MODELS)),
    cv: int = Query(default=5, ge=2, le=10, description="Number of stratified CV folds"),
):
    _check_model_name(model_name)
    try:
        return cross_validate_model(model_name, cv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
