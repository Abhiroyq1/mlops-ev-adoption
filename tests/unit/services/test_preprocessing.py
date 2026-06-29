from unittest.mock import patch

from app.services.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    get_splits,
    get_preprocessing_info,
)


class TestFeatureDefinitions:
    def test_numeric_features_count(self):
        assert len(NUMERIC_FEATURES) == 17

    def test_categorical_features_count(self):
        assert len(CATEGORICAL_FEATURES) == 5

    def test_feature_columns_is_concatenation(self):
        assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def test_total_feature_count(self):
        assert len(FEATURE_COLUMNS) == 22

    def test_no_duplicate_features(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_target_not_in_features(self):
        assert "ev_adoption_likelihood" not in FEATURE_COLUMNS

    def test_key_numeric_features_present(self):
        assert "age" in NUMERIC_FEATURES
        assert "annual_income" in NUMERIC_FEATURES
        assert "ev_knowledge_score" in NUMERIC_FEATURES

    def test_key_categorical_features_present(self):
        assert "education_level" in CATEGORICAL_FEATURES
        assert "city_type" in CATEGORICAL_FEATURES
        assert "home_charging_available" in CATEGORICAL_FEATURES


class TestGetSplits:
    def test_split_sizes_80_20(self, sample_df):
        # sample_df has 150 rows; 80% = 120 train, 20% = 30 test
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            X_train, X_test, y_train, y_test = get_splits()
        assert len(X_train) == 120
        assert len(X_test) == 30

    def test_y_sizes_match_x(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            X_train, X_test, y_train, y_test = get_splits()
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_X_columns_are_feature_columns(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            X_train, X_test, _, _ = get_splits()
        assert list(X_train.columns) == FEATURE_COLUMNS
        assert list(X_test.columns) == FEATURE_COLUMNS

    def test_stratification_preserves_all_classes_in_test(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            _, _, _, y_test = get_splits()
        assert set(y_test.unique()) == {"High", "Medium", "Low"}

    def test_stratification_preserves_all_classes_in_train(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            _, _, y_train, _ = get_splits()
        assert set(y_train.unique()) == {"High", "Medium", "Low"}

    def test_no_row_overlap_between_splits(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            X_train, X_test, _, _ = get_splits()
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert train_indices.isdisjoint(test_indices)


class TestGetPreprocessingInfo:
    def test_all_keys_present(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            info = get_preprocessing_info()
        expected = {"train_size", "test_size", "feature_columns", "numeric_features", "categorical_features"}
        assert expected.issubset(info.keys())

    def test_split_sizes(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            info = get_preprocessing_info()
        assert info["train_size"] == 120
        assert info["test_size"] == 30

    def test_feature_column_lists_match_definitions(self, sample_df):
        with patch("app.services.preprocessing.load_data", return_value=sample_df):
            info = get_preprocessing_info()
        assert info["numeric_features"] == NUMERIC_FEATURES
        assert info["categorical_features"] == CATEGORICAL_FEATURES
