import pytest
from unittest.mock import patch

from app.services.data_service import load_data, get_data_summary


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear load_data cache before and after each test to prevent cross-test pollution."""
    load_data.cache_clear()
    yield
    load_data.cache_clear()


class TestLoadData:
    def test_raises_file_not_found_for_missing_csv(self, tmp_path):
        with patch("app.services.data_service.DATA_PATH", tmp_path / "nonexistent.csv"):
            with pytest.raises(FileNotFoundError, match="Dataset not found"):
                load_data()

    def test_returns_dataframe(self, sample_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)
        with patch("app.services.data_service.DATA_PATH", csv_path):
            df = load_data()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_csv(self, sample_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)
        with patch("app.services.data_service.DATA_PATH", csv_path):
            df = load_data()
        assert len(df) == len(sample_df)

    def test_columns_are_lowercased(self, sample_df, tmp_path):
        df_upper = sample_df.rename(columns=str.upper)
        csv_path = tmp_path / "upper.csv"
        df_upper.to_csv(csv_path, index=False)
        with patch("app.services.data_service.DATA_PATH", csv_path):
            df = load_data()
        assert all(col == col.lower() for col in df.columns)

    def test_columns_are_stripped(self, sample_df, tmp_path):
        df_spaced = sample_df.rename(columns=lambda c: f"  {c}  ")
        csv_path = tmp_path / "spaced.csv"
        df_spaced.to_csv(csv_path, index=False)
        with patch("app.services.data_service.DATA_PATH", csv_path):
            df = load_data()
        assert all(col == col.strip() for col in df.columns)


class TestGetDataSummary:
    def test_all_expected_keys_present(self, sample_df):
        with patch("app.services.data_service.load_data", return_value=sample_df):
            summary = get_data_summary()
        expected_keys = {"num_rows", "num_columns", "columns", "dtypes", "missing_values", "target_distribution"}
        assert expected_keys.issubset(summary.keys())

    def test_num_rows_correct(self, sample_df):
        with patch("app.services.data_service.load_data", return_value=sample_df):
            summary = get_data_summary()
        assert summary["num_rows"] == len(sample_df)

    def test_num_columns_correct(self, sample_df):
        with patch("app.services.data_service.load_data", return_value=sample_df):
            summary = get_data_summary()
        assert summary["num_columns"] == len(sample_df.columns)

    def test_target_distribution_classes(self, sample_df):
        with patch("app.services.data_service.load_data", return_value=sample_df):
            summary = get_data_summary()
        assert set(summary["target_distribution"].keys()) == {"High", "Medium", "Low"}

    def test_dtypes_are_strings(self, sample_df):
        with patch("app.services.data_service.load_data", return_value=sample_df):
            summary = get_data_summary()
        assert all(isinstance(v, str) for v in summary["dtypes"].values())

    def test_missing_values_are_ints(self, sample_df):
        with patch("app.services.data_service.load_data", return_value=sample_df):
            summary = get_data_summary()
        assert all(isinstance(v, (int, float)) for v in summary["missing_values"].values())
