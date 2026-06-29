from unittest.mock import patch

from app.services.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS

_MOCK_SPLIT_INFO = {
    "train_size": 40000,
    "test_size": 10000,
    "feature_columns": FEATURE_COLUMNS,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
}


class TestDataEngSplitsEndpoint:
    def test_returns_200(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", return_value=_MOCK_SPLIT_INFO):
            response = client.get("/data-eng/splits")
        assert response.status_code == 200

    def test_train_size_in_response(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", return_value=_MOCK_SPLIT_INFO):
            data = client.get("/data-eng/splits").json()
        assert data["train_size"] == 40000

    def test_test_size_in_response(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", return_value=_MOCK_SPLIT_INFO):
            data = client.get("/data-eng/splits").json()
        assert data["test_size"] == 10000

    def test_feature_columns_count(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", return_value=_MOCK_SPLIT_INFO):
            data = client.get("/data-eng/splits").json()
        assert len(data["feature_columns"]) == 22

    def test_numeric_features_count(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", return_value=_MOCK_SPLIT_INFO):
            data = client.get("/data-eng/splits").json()
        assert len(data["numeric_features"]) == 17

    def test_categorical_features_count(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", return_value=_MOCK_SPLIT_INFO):
            data = client.get("/data-eng/splits").json()
        assert len(data["categorical_features"]) == 5

    def test_all_schema_fields_present(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", return_value=_MOCK_SPLIT_INFO):
            data = client.get("/data-eng/splits").json()
        for field in ("train_size", "test_size", "feature_columns", "numeric_features", "categorical_features"):
            assert field in data

    def test_internal_error_returns_500(self, client):
        with patch("app.routers.data_engineering.get_preprocessing_info", side_effect=RuntimeError("split failed")):
            response = client.get("/data-eng/splits")
        assert response.status_code == 500
