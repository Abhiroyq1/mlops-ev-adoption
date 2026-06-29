from fastapi import APIRouter, HTTPException
from app.services.preprocessing import get_preprocessing_info
from app.schemas.responses import PreprocessingResponse

router = APIRouter(prefix="/data-eng", tags=["Data Engineering"])


@router.get("/splits", response_model=PreprocessingResponse)
def preprocessing_info():
    try:
        return get_preprocessing_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
