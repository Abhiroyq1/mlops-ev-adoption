from unittest.mock import patch

from app.services.eda_service import get_eda


class TestGetEDA:
    def test_returns_three_top_level_keys(self, sample_df):
        with patch("app.services.eda_service.load_data", return_value=sample_df):
            result = get_eda()
        assert set(result.keys()) == {"describe", "correlation", "categorical_counts"}

    def test_describe_contains_numeric_columns(self, sample_df):
        with patch("app.services.eda_service.load_data", return_value=sample_df):
            result = get_eda()
        assert "age" in result["describe"]
        assert "annual_income" in result["describe"]

    def test_describe_has_stat_keys_per_column(self, sample_df):
        with patch("app.services.eda_service.load_data", return_value=sample_df):
            result = get_eda()
        age_stats = result["describe"]["age"]
        assert "mean" in age_stats
        assert "std" in age_stats

    def test_correlation_is_symmetric(self, sample_df):
        with patch("app.services.eda_service.load_data", return_value=sample_df):
            result = get_eda()
        corr = result["correlation"]
        assert "age" in corr
        # self-correlation must be 1
        assert abs(corr["age"]["age"] - 1.0) < 1e-4

    def test_categorical_counts_contains_expected_columns(self, sample_df):
        with patch("app.services.eda_service.load_data", return_value=sample_df):
            result = get_eda()
        assert "education_level" in result["categorical_counts"]
        assert "city_type" in result["categorical_counts"]
        assert "current_vehicle_type" in result["categorical_counts"]

    def test_categorical_counts_values_are_numeric(self, sample_df):
        with patch("app.services.eda_service.load_data", return_value=sample_df):
            result = get_eda()
        for col, counts in result["categorical_counts"].items():
            for val, count in counts.items():
                assert isinstance(count, (int, float)), f"{col}[{val}] is not numeric"

    def test_education_level_categories_present(self, sample_df):
        with patch("app.services.eda_service.load_data", return_value=sample_df):
            result = get_eda()
        edu_counts = result["categorical_counts"]["education_level"]
        assert "Bachelor" in edu_counts
        assert "Master" in edu_counts
