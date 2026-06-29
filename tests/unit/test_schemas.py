import pytest
from pydantic import ValidationError

from app.schemas.requests import PredictRequest
from app.schemas.responses import (
    DataSummaryResponse,
    EDAResponse,
    PreprocessingResponse,
    TrainResponse,
    PredictResponse,
    MetricsResponse,
)


class TestDataSummaryResponse:
    def test_valid(self):
        r = DataSummaryResponse(
            num_rows=1000, num_columns=23, columns=["age", "city"],
            dtypes={"age": "float64"}, missing_values={"age": 0},
            target_distribution={"High": 400, "Medium": 350, "Low": 250},
        )
        assert r.num_rows == 1000
        assert r.num_columns == 23

    def test_target_distribution_keys(self):
        r = DataSummaryResponse(
            num_rows=100, num_columns=5, columns=[],
            dtypes={}, missing_values={},
            target_distribution={"High": 40, "Medium": 30, "Low": 30},
        )
        assert set(r.target_distribution.keys()) == {"High", "Medium", "Low"}

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DataSummaryResponse(num_rows=100)


class TestEDAResponse:
    def test_valid(self):
        r = EDAResponse(
            describe={"age": {"mean": 45.0, "std": 12.0}},
            correlation={"age": {"age": 1.0}},
            categorical_counts={"city_type": {"Urban": 500, "Rural": 200}},
        )
        assert "age" in r.describe

    def test_empty_dicts_allowed(self):
        r = EDAResponse(describe={}, correlation={}, categorical_counts={})
        assert r.describe == {}


class TestPreprocessingResponse:
    def test_valid(self):
        r = PreprocessingResponse(
            train_size=800, test_size=200,
            feature_columns=["age", "city"],
            numeric_features=["age"],
            categorical_features=["city"],
        )
        assert r.train_size == 800
        assert r.test_size == 200

    def test_feature_lists_preserved(self):
        cols = ["age", "income"]
        r = PreprocessingResponse(
            train_size=800, test_size=200,
            feature_columns=cols, numeric_features=cols, categorical_features=[],
        )
        assert r.feature_columns == cols


class TestTrainResponse:
    def test_valid(self):
        r = TrainResponse(
            model_name="random_forest",
            artifact_path="models/random_forest.joblib",
            train_accuracy=0.95,
            test_accuracy=0.85,
            classes=["High", "Low", "Medium"],
        )
        assert r.model_name == "random_forest"
        assert r.test_accuracy == 0.85

    def test_invalid_accuracy_type_raises(self):
        with pytest.raises(ValidationError):
            TrainResponse(
                model_name="x", artifact_path="x",
                train_accuracy="not_a_float", test_accuracy=0.8,
                classes=["High"],
            )


class TestPredictRequest:
    def test_valid_with_features(self):
        r = PredictRequest(features={"age": 30, "city": "Urban"})
        assert r.features["age"] == 30

    def test_empty_features_allowed(self):
        r = PredictRequest(features={})
        assert r.features == {}

    def test_nested_values_preserved(self):
        r = PredictRequest(features={"nested": {"key": "value"}})
        assert r.features["nested"] == {"key": "value"}


class TestPredictResponse:
    def test_valid(self):
        r = PredictResponse(
            prediction="High",
            probabilities={"High": 0.8, "Medium": 0.1, "Low": 0.1},
        )
        assert r.prediction == "High"

    def test_probabilities_type(self):
        r = PredictResponse(
            prediction="Low",
            probabilities={"High": 0.1, "Medium": 0.2, "Low": 0.7},
        )
        assert all(isinstance(v, float) for v in r.probabilities.values())


class TestMetricsResponse:
    def test_valid(self):
        r = MetricsResponse(
            model_name="random_forest",
            accuracy=0.85,
            classification_report={"High": {"precision": 0.9, "recall": 0.88}},
            confusion_matrix=[[10, 1, 0], [0, 8, 2], [1, 0, 9]],
            classes=["High", "Low", "Medium"],
        )
        assert r.accuracy == 0.85

    def test_confusion_matrix_is_list_of_lists(self):
        r = MetricsResponse(
            model_name="rf", accuracy=0.9,
            classification_report={},
            confusion_matrix=[[5, 0], [1, 4]],
            classes=["A", "B"],
        )
        assert isinstance(r.confusion_matrix, list)
        assert isinstance(r.confusion_matrix[0], list)
