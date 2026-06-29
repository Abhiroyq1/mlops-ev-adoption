from pydantic import BaseModel
from typing import Any


class PredictRequest(BaseModel):
    features: dict[str, Any]
