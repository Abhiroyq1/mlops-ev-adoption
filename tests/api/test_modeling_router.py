import pytest
from unittest.mock import patch

_MOCK_TRAIN_RESULT = {
    "model_name": "random_forest",
    "artifact_path": "models/random_forest.joblib",
    "train_accuracy": 0.9852,
    "test_accuracy": 0.8525,
    "classes": ["High", "Low", "Medium"],
}

_MOCK_PREDICT_RESULT = {
    "prediction": "High",
    "probabilities": {"High": 0.75, "Medium": 0.15, "Low": 0.10},
}

_MOCK_MODEL_LIST = ["random_forest", "logistic_regression"]


class TestTrainEndpoint:
    def test_returns_200_default_model(self, client):
        with patch("app.routers.modeling.train_model", return_value=_MOCK_TRAIN_RESULT):
            response = client.post("/model/train")
        assert response.status_code == 200

    def test_returns_200_explicit_model(self, client):
        with patch("app.routers.modeling.train_model", return_value=_MOCK_TRAIN_RESULT):
            response = client.post("/model/train?model_name=random_forest")
        assert response.status_code == 200

    def test_response_contains_model_name(self, client):
        with patch("app.routers.modeling.train_model", return_value=_MOCK_TRAIN_RESULT):
            data = client.post("/model/train").json()
        assert data["model_name"] == "random_forest"

    def test_response_contains_accuracies(self, client):
        with patch("app.routers.modeling.train_model", return_value=_MOCK_TRAIN_RESULT):
            data = client.post("/model/train").json()
        assert "train_accuracy" in data
        assert "test_accuracy" in data
        assert data["test_accuracy"] == 0.8525

    def test_response_contains_classes(self, client):
        with patch("app.routers.modeling.train_model", return_value=_MOCK_TRAIN_RESULT):
            data = client.post("/model/train").json()
        assert "classes" in data
        assert set(data["classes"]) == {"High", "Low", "Medium"}

    def test_invalid_model_name_returns_422(self, client):
        response = client.post("/model/train?model_name=xgboost")
        assert response.status_code == 422

    def test_training_error_returns_500(self, client):
        with patch("app.routers.modeling.train_model", side_effect=RuntimeError("training failed")):
            response = client.post("/model/train")
        assert response.status_code == 500

    @pytest.mark.parametrize("model_name", ["random_forest", "gradient_boosting", "logistic_regression"])
    def test_all_supported_model_names_accepted(self, client, model_name):
        with patch("app.routers.modeling.train_model", return_value={**_MOCK_TRAIN_RESULT, "model_name": model_name}):
            response = client.post(f"/model/train?model_name={model_name}")
        assert response.status_code == 200


class TestPredictEndpoint:
    def test_returns_200(self, client, sample_features):
        with patch("app.routers.modeling.predict", return_value=_MOCK_PREDICT_RESULT):
            response = client.post("/model/predict", json={"features": sample_features})
        assert response.status_code == 200

    def test_prediction_in_response(self, client, sample_features):
        with patch("app.routers.modeling.predict", return_value=_MOCK_PREDICT_RESULT):
            data = client.post("/model/predict", json={"features": sample_features}).json()
        assert data["prediction"] == "High"

    def test_probabilities_in_response(self, client, sample_features):
        with patch("app.routers.modeling.predict", return_value=_MOCK_PREDICT_RESULT):
            data = client.post("/model/predict", json={"features": sample_features}).json()
        assert "probabilities" in data
        assert "High" in data["probabilities"]

    def test_missing_body_returns_422(self, client):
        response = client.post("/model/predict")
        assert response.status_code == 422

    def test_no_artifact_returns_404(self, client, sample_features):
        with patch("app.routers.modeling.predict", side_effect=FileNotFoundError("no artifact")):
            response = client.post("/model/predict", json={"features": sample_features})
        assert response.status_code == 404

    def test_invalid_model_name_returns_422(self, client, sample_features):
        response = client.post("/model/predict?model_name=xgboost", json={"features": sample_features})
        assert response.status_code == 422

    def test_explicit_model_name_passed_through(self, client, sample_features):
        captured = {}

        def fake_predict(model_name, features):
            captured["model_name"] = model_name
            return _MOCK_PREDICT_RESULT

        with patch("app.routers.modeling.predict", side_effect=fake_predict):
            client.post("/model/predict?model_name=logistic_regression", json={"features": sample_features})
        assert captured["model_name"] == "logistic_regression"


class TestListModelsEndpoint:
    def test_returns_200(self, client):
        with patch("app.routers.modeling.list_trained_models", return_value=_MOCK_MODEL_LIST):
            response = client.get("/model/list")
        assert response.status_code == 200

    def test_returns_list(self, client):
        with patch("app.routers.modeling.list_trained_models", return_value=_MOCK_MODEL_LIST):
            data = client.get("/model/list").json()
        assert isinstance(data, list)

    def test_returns_model_names(self, client):
        with patch("app.routers.modeling.list_trained_models", return_value=_MOCK_MODEL_LIST):
            data = client.get("/model/list").json()
        assert "random_forest" in data
        assert "logistic_regression" in data

    def test_empty_models_dir_returns_empty_list(self, client):
        with patch("app.routers.modeling.list_trained_models", return_value=[]):
            data = client.get("/model/list").json()
        assert data == []
