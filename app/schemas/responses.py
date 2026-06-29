from pydantic import BaseModel
from typing import Any


class DataSummaryResponse(BaseModel):
    num_rows: int
    num_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    target_distribution: dict[str, int]


class EDAResponse(BaseModel):
    describe: dict[str, Any]
    correlation: dict[str, Any]
    categorical_counts: dict[str, dict[str, int]]


class PreprocessingResponse(BaseModel):
    train_size: int
    test_size: int
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


class TrainResponse(BaseModel):
    model_name: str
    artifact_path: str
    train_accuracy: float
    test_accuracy: float
    classes: list[str]


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class MetricsResponse(BaseModel):
    model_name: str
    accuracy: float
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    classes: list[str]
