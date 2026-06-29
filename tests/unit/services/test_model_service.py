import joblib
import pytest
from pathlib import Path
from unittest.mock import patch
from sklearn.model_selection import train_test_split

from app.services.model_service import train_model, predict, list_trained_models
from app.services.preprocessing import FEATURE_COLUMNS
from app.ml.pipeline import build_pipeline


@pytest.fixture
def splits(sample_df):
    X = sample_df[FEATURE_COLUMNS]
    y = sample_df["ev_adoption_likelihood"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@pytest.fixture
def saved_artifact(tmp_path, sample_df):
    """Train a logistic regression on sample data and persist it to tmp_path."""
    X = sample_df[FEATURE_COLUMNS]
    y = sample_df["ev_adoption_likelihood"]
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(X, y)
    artifact_path = tmp_path / "logistic_regression.joblib"
    joblib.dump(pipeline, artifact_path)
    return tmp_path, pipeline


class TestTrainModel:
    def test_returns_model_name(self, splits, tmp_path):
        with patch("app.services.model_service.get_splits", return_value=splits), \
             patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = train_model("logistic_regression")
        assert result["model_name"] == "logistic_regression"

    def test_returns_artifact_path(self, splits, tmp_path):
        with patch("app.services.model_service.get_splits", return_value=splits), \
             patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = train_model("logistic_regression")
        assert "artifact_path" in result
        assert result["artifact_path"].endswith(".joblib")

    def test_artifact_file_exists_after_training(self, splits, tmp_path):
        with patch("app.services.model_service.get_splits", return_value=splits), \
             patch("app.services.model_service.MODELS_DIR", tmp_path):
            train_model("logistic_regression")
        assert (tmp_path / "logistic_regression.joblib").exists()

    def test_train_accuracy_is_float_in_range(self, splits, tmp_path):
        with patch("app.services.model_service.get_splits", return_value=splits), \
             patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = train_model("logistic_regression")
        assert 0.0 <= result["train_accuracy"] <= 1.0

    def test_test_accuracy_is_float_in_range(self, splits, tmp_path):
        with patch("app.services.model_service.get_splits", return_value=splits), \
             patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = train_model("logistic_regression")
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_classes_are_valid_labels(self, splits, tmp_path):
        with patch("app.services.model_service.get_splits", return_value=splits), \
             patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = train_model("logistic_regression")
        assert set(result["classes"]).issubset({"High", "Medium", "Low"})

    def test_classes_are_strings(self, splits, tmp_path):
        with patch("app.services.model_service.get_splits", return_value=splits), \
             patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = train_model("logistic_regression")
        assert all(isinstance(c, str) for c in result["classes"])


class TestPredict:
    def test_prediction_is_valid_class(self, saved_artifact, sample_features):
        tmp_path, _ = saved_artifact
        with patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = predict("logistic_regression", sample_features)
        assert result["prediction"] in {"High", "Medium", "Low"}

    def test_probabilities_sum_to_one(self, saved_artifact, sample_features):
        tmp_path, _ = saved_artifact
        with patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = predict("logistic_regression", sample_features)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_probabilities_have_all_classes(self, saved_artifact, sample_features):
        tmp_path, _ = saved_artifact
        with patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = predict("logistic_regression", sample_features)
        assert set(result["probabilities"].keys()) == {"High", "Medium", "Low"}

    def test_missing_artifact_raises_file_not_found(self, tmp_path, sample_features):
        with patch("app.services.model_service.MODELS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="No trained artifact"):
                predict("random_forest", sample_features)


class TestListTrainedModels:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        with patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = list_trained_models()
        assert result == []

    def test_returns_model_stem_after_training(self, saved_artifact):
        tmp_path, _ = saved_artifact
        with patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = list_trained_models()
        assert "logistic_regression" in result

    def test_only_joblib_files_listed(self, tmp_path):
        (tmp_path / "model.txt").write_text("not a model")
        (tmp_path / "random_forest.joblib").write_bytes(b"fake")
        with patch("app.services.model_service.MODELS_DIR", tmp_path):
            result = list_trained_models()
        assert result == ["random_forest"]
        assert "model" not in result
