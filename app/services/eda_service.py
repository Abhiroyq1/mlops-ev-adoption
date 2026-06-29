import pandas as pd
from app.services.data_service import load_data


def get_eda() -> dict:
    df = load_data()
    numeric_df = df.select_dtypes(include="number")
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    describe = numeric_df.describe().round(4).to_dict()
    correlation = numeric_df.corr().round(4).to_dict()
    categorical_counts = {
        col: df[col].value_counts().to_dict() for col in categorical_cols
    }

    return {
        "describe": describe,
        "correlation": correlation,
        "categorical_counts": categorical_counts,
    }
