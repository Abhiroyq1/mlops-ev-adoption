import pandas as pd
from functools import lru_cache
from app.config import DATA_PATH, settings


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_data_summary() -> dict:
    df = load_data()
    target = settings.target_column
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df[target].value_counts().to_dict(),
    }
