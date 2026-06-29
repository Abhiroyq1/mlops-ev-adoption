from fastapi import APIRouter, HTTPException, Query
from app.services.model_service import train_model, predict, list_trained_models
from app.schemas.requests import PredictRequest
from app.schemas.responses import TrainResponse, PredictResponse
from app.ml.pipeline import SUPPORTED_MODELS

router = APIRouter(prefix="/model", tags=["Modeling"])


@router.post("/train", response_model=TrainResponse)
def train(model_name: str = Query(default="random_forest", enum=list(SUPPORTED_MODELS))):
    try:
        return train_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
def run_predict(
    body: PredictRequest,
    model_name: str = Query(default="random_forest", enum=list(SUPPORTED_MODELS)),
):
    try:
        return predict(model_name, body.features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
def get_trained_models() -> list[str]:
    return list_trained_models()
