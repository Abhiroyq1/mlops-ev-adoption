"""
Integration tests — exercise the full stack against the real CSV.

These tests are slow (they train real models on 50k rows) and require
the dataset to be present at data/global_ev_adoption_behavior_2026.csv.

Run selectively:
    pytest tests/integration -v
    pytest -m integration -v

Skip in CI where the CSV is absent:
    pytest -m "not integration"
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.data_service import load_data


@pytest.fixture(scope="module", autouse=True)
def clear_data_cache():
    load_data.cache_clear()
    yield
    load_data.cache_clear()


@pytest.fixture(scope="module")
def integration_client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_eda_summary_loads_real_data(integration_client):
    response = integration_client.get("/eda/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["num_rows"] == 50000
    assert data["num_columns"] >= 20
    assert set(data["target_distribution"].keys()) == {"High", "Medium", "Low"}


@pytest.mark.integration
def test_eda_analysis_real_data(integration_client):
    response = integration_client.get("/eda/analysis")
    assert response.status_code == 200
    data = response.json()
    assert "age" in data["describe"]
    assert "city_type" in data["categorical_counts"]


@pytest.mark.integration
def test_data_eng_splits_real_data(integration_client):
    response = integration_client.get("/data-eng/splits")
    assert response.status_code == 200
    data = response.json()
    assert data["train_size"] == 40000
    assert data["test_size"] == 10000
    assert len(data["feature_columns"]) == 22


# ---------------------------------------------------------------------------
# Train → List → Predict → Metrics pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_train_random_forest(integration_client, tmp_path):
    with patch("app.services.model_service.MODELS_DIR", tmp_path), \
         patch("app.services.metrics_service.MODELS_DIR", tmp_path):
        response = integration_client.post("/model/train?model_name=random_forest")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "random_forest"
    assert data["test_accuracy"] > 0.5
    assert set(data["classes"]) == {"High", "Medium", "Low"}


@pytest.mark.integration
def test_train_logistic_regression(integration_client, tmp_path):
    with patch("app.services.model_service.MODELS_DIR", tmp_path):
        response = integration_client.post("/model/train?model_name=logistic_regression")
    assert response.status_code == 200
    assert response.json()["test_accuracy"] > 0.3


@pytest.mark.integration
def test_predict_after_training(integration_client, tmp_path, sample_features):
    # Train first into the tmp models dir
    with patch("app.services.model_service.MODELS_DIR", tmp_path):
        integration_client.post("/model/train?model_name=logistic_regression")

    with patch("app.services.model_service.MODELS_DIR", tmp_path):
        response = integration_client.post(
            "/model/predict?model_name=logistic_regression",
            json={"features": sample_features},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"High", "Medium", "Low"}
    assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01


@pytest.mark.integration
def test_predict_without_training_returns_404(integration_client, tmp_path, sample_features):
    with patch("app.services.model_service.MODELS_DIR", tmp_path):
        response = integration_client.post(
            "/model/predict?model_name=gradient_boosting",
            json={"features": sample_features},
        )
    assert response.status_code == 404


@pytest.mark.integration
def test_metrics_after_training(integration_client, tmp_path):
    with patch("app.services.model_service.MODELS_DIR", tmp_path):
        integration_client.post("/model/train?model_name=logistic_regression")

    with patch("app.services.model_service.MODELS_DIR", tmp_path), \
         patch("app.services.metrics_service.MODELS_DIR", tmp_path):
        response = integration_client.get("/metrics/evaluate?model_name=logistic_regression")

    assert response.status_code == 200
    data = response.json()
    assert 0.0 < data["accuracy"] <= 1.0
    assert len(data["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in data["confusion_matrix"])


@pytest.mark.integration
def test_metrics_without_training_returns_404(integration_client, tmp_path):
    with patch("app.services.metrics_service.MODELS_DIR", tmp_path):
        response = integration_client.get("/metrics/evaluate?model_name=gradient_boosting")
    assert response.status_code == 404


@pytest.mark.integration
def test_list_models_after_training(integration_client, tmp_path):
    with patch("app.services.model_service.MODELS_DIR", tmp_path):
        integration_client.post("/model/train?model_name=logistic_regression")

    with patch("app.services.model_service.MODELS_DIR", tmp_path):
        response = integration_client.get("/model/list")
        with patch("app.routers.modeling.list_trained_models",
                   side_effect=lambda: [p.stem for p in tmp_path.glob("*.joblib")]):
            response = integration_client.get("/model/list")

    assert response.status_code == 200
