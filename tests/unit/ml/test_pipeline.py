import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from app.ml.pipeline import build_pipeline, SUPPORTED_MODELS
from app.services.preprocessing import FEATURE_COLUMNS

_ALL_MODEL_NAMES = list(SUPPORTED_MODELS.keys())


class TestSupportedModels:
    def test_registry_has_seven_models(self):
        assert len(SUPPORTED_MODELS) == 7

    def test_all_expected_keys_present(self):
        expected = {
            "random_forest", "gradient_boosting", "logistic_regression",
            "extra_trees", "adaboost", "svm", "voting_soft",
        }
        assert set(SUPPORTED_MODELS.keys()) == expected

    def test_model_types(self):
        assert isinstance(SUPPORTED_MODELS["random_forest"], RandomForestClassifier)
        assert isinstance(SUPPORTED_MODELS["gradient_boosting"], GradientBoostingClassifier)
        assert isinstance(SUPPORTED_MODELS["logistic_regression"], LogisticRegression)
        assert isinstance(SUPPORTED_MODELS["extra_trees"], ExtraTreesClassifier)
        assert isinstance(SUPPORTED_MODELS["adaboost"], AdaBoostClassifier)
        assert isinstance(SUPPORTED_MODELS["svm"], SVC)
        assert isinstance(SUPPORTED_MODELS["voting_soft"], VotingClassifier)

    def test_svm_has_probability_enabled(self):
        assert SUPPORTED_MODELS["svm"].probability is True

    def test_voting_soft_uses_soft_voting(self):
        assert SUPPORTED_MODELS["voting_soft"].voting == "soft"


class TestBuildPipeline:
    def test_returns_sklearn_pipeline(self):
        pipeline = build_pipeline("random_forest")
        assert isinstance(pipeline, Pipeline)

    @pytest.mark.parametrize("model_name", _ALL_MODEL_NAMES)
    def test_all_models_build_successfully(self, model_name):
        pipeline = build_pipeline(model_name)
        assert isinstance(pipeline, Pipeline)

    def test_build_pipeline_returns_fresh_clone(self):
        p1 = build_pipeline("random_forest")
        p2 = build_pipeline("random_forest")
        assert p1["classifier"] is not p2["classifier"]

    def test_pipeline_has_preprocessor_step(self):
        pipeline = build_pipeline("random_forest")
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_has_classifier_step(self):
        pipeline = build_pipeline("random_forest")
        assert "classifier" in pipeline.named_steps

    def test_pipeline_step_order(self):
        pipeline = build_pipeline("logistic_regression")
        assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]

    def test_unsupported_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            build_pipeline("xgboost")

    def test_unsupported_model_error_lists_valid_names(self):
        with pytest.raises(ValueError) as exc_info:
            build_pipeline("bert")
        assert "random_forest" in str(exc_info.value)


class TestPipelineFitPredict:
    def test_fit_predict_shape(self, sample_df):
        X = sample_df[FEATURE_COLUMNS]
        y = sample_df["ev_adoption_likelihood"]
        pipeline = build_pipeline("logistic_regression")
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(X)

    def test_predictions_are_valid_classes(self, sample_df):
        X = sample_df[FEATURE_COLUMNS]
        y = sample_df["ev_adoption_likelihood"]
        pipeline = build_pipeline("logistic_regression")
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert set(preds).issubset({"High", "Medium", "Low"})

    def test_predict_proba_sums_to_one(self, sample_df):
        X = sample_df[FEATURE_COLUMNS]
        y = sample_df["ev_adoption_likelihood"]
        pipeline = build_pipeline("logistic_regression")
        pipeline.fit(X, y)
        probas = pipeline.predict_proba(X)
        assert probas.shape == (len(X), 3)
        row_sums = probas.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-6 for s in row_sums)

    def test_classes_attribute_after_fit(self, sample_df):
        X = sample_df[FEATURE_COLUMNS]
        y = sample_df["ev_adoption_likelihood"]
        pipeline = build_pipeline("logistic_regression")
        pipeline.fit(X, y)
        assert set(pipeline.classes_) == {"High", "Medium", "Low"}
