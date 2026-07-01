from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from app.services.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# SVC with probability=True uses Platt scaling (internal 5-fold CV), so expect
# longer training time on large datasets (~40K rows). cache_size=500 helps.
SUPPORTED_MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, C=1.0, random_state=42, cache_size=500),
    "voting_soft": VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ],
        voting="soft",
    ),
}


def build_pipeline(model_name: str = "random_forest") -> Pipeline:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {list(SUPPORTED_MODELS)}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    # clone() gives each pipeline call a fresh unfitted estimator, preventing
    # shared-instance state bugs when the same model is trained multiple times.
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clone(SUPPORTED_MODELS[model_name])),
    ])
