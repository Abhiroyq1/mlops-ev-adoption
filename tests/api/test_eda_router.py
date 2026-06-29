from unittest.mock import patch

_MOCK_SUMMARY = {
    "num_rows": 50000, "num_columns": 23,
    "columns": ["age", "ev_adoption_likelihood"],
    "dtypes": {"age": "float64", "ev_adoption_likelihood": "object"},
    "missing_values": {"age": 0},
    "target_distribution": {"High": 20000, "Medium": 17000, "Low": 13000},
}

_MOCK_EDA = {
    "describe": {"age": {"mean": 45.2, "std": 12.1, "min": 18.0, "max": 80.0}},
    "correlation": {"age": {"age": 1.0, "annual_income": 0.12}},
    "categorical_counts": {
        "city_type": {"Urban": 25000, "Suburban": 15000, "Rural": 10000},
        "education_level": {"Bachelor": 18000, "Master": 12000, "High School": 10000},
    },
}


class TestEDASummaryEndpoint:
    def test_returns_200(self, client):
        with patch("app.routers.eda.get_data_summary", return_value=_MOCK_SUMMARY):
            response = client.get("/eda/summary")
        assert response.status_code == 200

    def test_num_rows_in_response(self, client):
        with patch("app.routers.eda.get_data_summary", return_value=_MOCK_SUMMARY):
            data = client.get("/eda/summary").json()
        assert data["num_rows"] == 50000

    def test_num_columns_in_response(self, client):
        with patch("app.routers.eda.get_data_summary", return_value=_MOCK_SUMMARY):
            data = client.get("/eda/summary").json()
        assert data["num_columns"] == 23

    def test_target_distribution_in_response(self, client):
        with patch("app.routers.eda.get_data_summary", return_value=_MOCK_SUMMARY):
            data = client.get("/eda/summary").json()
        assert "High" in data["target_distribution"]
        assert "Medium" in data["target_distribution"]
        assert "Low" in data["target_distribution"]

    def test_file_not_found_returns_404(self, client):
        with patch("app.routers.eda.get_data_summary", side_effect=FileNotFoundError("no CSV")):
            response = client.get("/eda/summary")
        assert response.status_code == 404
        assert "no CSV" in response.json()["detail"]

    def test_response_schema_fields(self, client):
        with patch("app.routers.eda.get_data_summary", return_value=_MOCK_SUMMARY):
            data = client.get("/eda/summary").json()
        for field in ("num_rows", "num_columns", "columns", "dtypes", "missing_values", "target_distribution"):
            assert field in data


class TestEDAAnalysisEndpoint:
    def test_returns_200(self, client):
        with patch("app.routers.eda.get_eda", return_value=_MOCK_EDA):
            response = client.get("/eda/analysis")
        assert response.status_code == 200

    def test_describe_field_present(self, client):
        with patch("app.routers.eda.get_eda", return_value=_MOCK_EDA):
            data = client.get("/eda/analysis").json()
        assert "describe" in data
        assert "age" in data["describe"]

    def test_correlation_field_present(self, client):
        with patch("app.routers.eda.get_eda", return_value=_MOCK_EDA):
            data = client.get("/eda/analysis").json()
        assert "correlation" in data

    def test_categorical_counts_field_present(self, client):
        with patch("app.routers.eda.get_eda", return_value=_MOCK_EDA):
            data = client.get("/eda/analysis").json()
        assert "categorical_counts" in data
        assert "city_type" in data["categorical_counts"]

    def test_internal_error_returns_500(self, client):
        with patch("app.routers.eda.get_eda", side_effect=RuntimeError("computation failed")):
            response = client.get("/eda/analysis")
        assert response.status_code == 500
