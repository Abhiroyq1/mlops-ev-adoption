from fastapi import APIRouter, HTTPException
from app.services.data_service import get_data_summary
from app.services.eda_service import get_eda
from app.schemas.responses import DataSummaryResponse, EDAResponse

router = APIRouter(prefix="/eda", tags=["EDA"])


@router.get("/summary", response_model=DataSummaryResponse)
def data_summary():
    try:
        return get_data_summary()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/analysis", response_model=EDAResponse)
def eda_analysis():
    try:
        return get_eda()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
