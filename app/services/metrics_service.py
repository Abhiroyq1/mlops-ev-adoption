import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.services.preprocessing import get_splits
from app.config import MODELS_DIR


def compute_metrics(model_name: str) -> dict:
    artifact_path = MODELS_DIR / f"{model_name}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No trained artifact for '{model_name}'. Train first.")

    pipeline = joblib.load(artifact_path)
    _, X_test, _, y_test = get_splits()

    y_pred = pipeline.predict(X_test)
    classes = list(pipeline.classes_)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": classes,
    }
