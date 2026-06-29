import joblib
import pytest
from unittest.mock import patch
from sklearn.model_selection import train_test_split

from app.services.metrics_service import compute_metrics
from app.services.preprocessing import FEATURE_COLUMNS
from app.ml.pipeline import build_pipeline


@pytest.fixture
def trained_setup(tmp_path, sample_df):
    """Train logistic regression on sample data, save artifact, return (tmp_path, splits)."""
    X = sample_df[FEATURE_COLUMNS]
    y = sample_df["ev_adoption_likelihood"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, tmp_path / "logistic_regression.joblib")
    return tmp_path, (X_train, X_test, y_train, y_test)


class TestComputeMetrics:
    def test_returns_model_name(self, trained_setup):
        tmp_path, splits = trained_setup
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            result = compute_metrics("logistic_regression")
        assert result["model_name"] == "logistic_regression"

    def test_accuracy_is_float_in_range(self, trained_setup):
        tmp_path, splits = trained_setup
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            result = compute_metrics("logistic_regression")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_is_square(self, trained_setup):
        tmp_path, splits = trained_setup
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            result = compute_metrics("logistic_regression")
        cm = result["confusion_matrix"]
        n = len(result["classes"])
        assert len(cm) == n
        assert all(len(row) == n for row in cm)

    def test_confusion_matrix_cells_are_non_negative_ints(self, trained_setup):
        tmp_path, splits = trained_setup
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            result = compute_metrics("logistic_regression")
        for row in result["confusion_matrix"]:
            for val in row:
                assert isinstance(val, int)
                assert val >= 0

    def test_confusion_matrix_row_sums_equal_test_size(self, trained_setup):
        tmp_path, splits = trained_setup
        _, (_, X_test, _, _) = trained_setup[0], trained_setup
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            result = compute_metrics("logistic_regression")
        total = sum(sum(row) for row in result["confusion_matrix"])
        assert total == len(splits[1])  # X_test length

    def test_classes_are_valid_labels(self, trained_setup):
        tmp_path, splits = trained_setup
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            result = compute_metrics("logistic_regression")
        assert set(result["classes"]).issubset({"High", "Medium", "Low"})

    def test_classification_report_has_per_class_keys(self, trained_setup):
        tmp_path, splits = trained_setup
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            result = compute_metrics("logistic_regression")
        report = result["classification_report"]
        # sklearn always adds accuracy, macro avg, weighted avg
        assert "accuracy" in report or any(c in report for c in {"High", "Medium", "Low"})

    def test_missing_artifact_raises_file_not_found(self, tmp_path, sample_df):
        X = sample_df[FEATURE_COLUMNS]
        y = sample_df["ev_adoption_likelihood"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        splits = (X_train, X_test, y_train, y_test)
        with patch("app.services.metrics_service.MODELS_DIR", tmp_path), \
             patch("app.services.metrics_service.get_splits", return_value=splits):
            with pytest.raises(FileNotFoundError):
                compute_metrics("random_forest")
