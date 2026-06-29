import pytest
from unittest.mock import patch

_MOCK_METRICS = {
    "model_name": "random_forest",
    "accuracy": 0.8525,
    "classification_report": {
        "High": {"precision": 0.88, "recall": 0.90, "f1-score": 0.89, "support": 3400},
        "Medium": {"precision": 0.82, "recall": 0.81, "f1-score": 0.81, "support": 3300},
        "Low": {"precision": 0.84, "recall": 0.83, "f1-score": 0.83, "support": 3300},
        "accuracy": 0.8525,
        "macro avg": {"precision": 0.85, "recall": 0.85, "f1-score": 0.84, "support": 10000},
        "weighted avg": {"precision": 0.85, "recall": 0.85, "f1-score": 0.85, "support": 10000},
    },
    "confusion_matrix": [[3060, 180, 160], [200, 2673, 427], [150, 410, 2740]],
    "classes": ["High", "Low", "Medium"],
}


class TestMetricsEvaluateEndpoint:
    def test_returns_200_default_model(self, client):
        with patch("app.routers.metrics.compute_metrics", return_value=_MOCK_METRICS):
            response = client.get("/metrics/evaluate")
        assert response.status_code == 200

    def test_returns_200_explicit_model(self, client):
        with patch("app.routers.metrics.compute_metrics", return_value=_MOCK_METRICS):
            response = client.get("/metrics/evaluate?model_name=random_forest")
        assert response.status_code == 200

    def test_accuracy_in_response(self, client):
        with patch("app.routers.metrics.compute_metrics", return_value=_MOCK_METRICS):
            data = client.get("/metrics/evaluate").json()
        assert data["accuracy"] == 0.8525

    def test_model_name_in_response(self, client):
        with patch("app.routers.metrics.compute_metrics", return_value=_MOCK_METRICS):
            data = client.get("/metrics/evaluate").json()
        assert data["model_name"] == "random_forest"

    def test_confusion_matrix_present(self, client):
        with patch("app.routers.metrics.compute_metrics", return_value=_MOCK_METRICS):
            data = client.get("/metrics/evaluate").json()
        assert "confusion_matrix" in data
        assert len(data["confusion_matrix"]) == 3

    def test_classification_report_present(self, client):
        with patch("app.routers.metrics.compute_metrics", return_value=_MOCK_METRICS):
            data = client.get("/metrics/evaluate").json()
        assert "classification_report" in data
        assert "High" in data["classification_report"]

    def test_classes_present(self, client):
        with patch("app.routers.metrics.compute_metrics", return_value=_MOCK_METRICS):
            data = client.get("/metrics/evaluate").json()
        assert set(data["classes"]) == {"High", "Low", "Medium"}

    def test_no_artifact_returns_404(self, client):
        with patch("app.routers.metrics.compute_metrics", side_effect=FileNotFoundError("no model")):
            response = client.get("/metrics/evaluate")
        assert response.status_code == 404
        assert "no model" in response.json()["detail"]

    def test_invalid_model_name_returns_422(self, client):
        response = client.get("/metrics/evaluate?model_name=xgboost")
        assert response.status_code == 422

    def test_internal_error_returns_500(self, client):
        with patch("app.routers.metrics.compute_metrics", side_effect=RuntimeError("crash")):
            response = client.get("/metrics/evaluate")
        assert response.status_code == 500

    @pytest.mark.parametrize("model_name", ["random_forest", "gradient_boosting", "logistic_regression"])
    def test_all_supported_models_accepted(self, client, model_name):
        mock = {**_MOCK_METRICS, "model_name": model_name}
        with patch("app.routers.metrics.compute_metrics", return_value=mock):
            response = client.get(f"/metrics/evaluate?model_name={model_name}")
        assert response.status_code == 200
