import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_validate as sklearn_cv, StratifiedKFold
from app.ml.pipeline import build_pipeline, SUPPORTED_MODELS
from app.services.preprocessing import get_splits, FEATURE_COLUMNS
from app.config import MODELS_DIR


def _artifact_path(model_name: str) -> Path:
    return MODELS_DIR / f"{model_name}.joblib"


def train_model(model_name: str = "random_forest") -> dict:
    X_train, X_test, y_train, y_test = get_splits()
    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    classes = list(pipeline.classes_)

    artifact_path = _artifact_path(model_name)
    joblib.dump(pipeline, artifact_path)

    return {
        "model_name": model_name,
        "artifact_path": str(artifact_path),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "classes": classes,
    }


def predict(model_name: str, features: dict) -> dict:
    artifact_path = _artifact_path(model_name)
    if not artifact_path.exists():
        raise FileNotFoundError(f"No trained artifact for '{model_name}'. Train first.")

    pipeline = joblib.load(artifact_path)
    input_df = pd.DataFrame([features])[FEATURE_COLUMNS]
    prediction = pipeline.predict(input_df)[0]
    probas = pipeline.predict_proba(input_df)[0]
    classes = list(pipeline.classes_)

    return {
        "prediction": prediction,
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probas)},
    }


def list_trained_models() -> list[str]:
    return [p.stem for p in MODELS_DIR.glob("*.joblib")]


def cross_validate_model(model_name: str, cv: int = 5) -> dict:
    X_train, _, y_train, _ = get_splits()
    pipeline = build_pipeline(model_name)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # n_jobs=1 on the outer loop avoids nested-parallelism issues on Windows;
    # estimators with n_jobs=-1 still parallelise within each fold.
    results = sklearn_cv(
        pipeline, X_train, y_train,
        cv=skf,
        scoring="accuracy",
        n_jobs=1,
        return_estimator=False,
    )

    scores = results["test_score"]
    fit_times = results["fit_time"]

    return {
        "model_name": model_name,
        "cv_folds": cv,
        "cv_scores": [round(float(s), 4) for s in scores],
        "mean_accuracy": round(float(scores.mean()), 4),
        "std_accuracy": round(float(scores.std()), 4),
        "min_accuracy": round(float(scores.min()), 4),
        "max_accuracy": round(float(scores.max()), 4),
        "mean_fit_time_seconds": round(float(fit_times.mean()), 2),
    }
