import pandas as pd
from sklearn.model_selection import train_test_split
from app.services.data_service import load_data
from app.config import settings

NUMERIC_FEATURES = [
    "age", "annual_income", "daily_commute_km", "weekly_travel_distance_km",
    "vehicle_age_years", "fuel_expense_per_month", "charging_station_accessibility",
    "nearest_charging_station_km", "electricity_cost_per_kwh",
    "environmental_awareness_score", "government_incentive_awareness",
    "technology_affinity_score", "range_anxiety_score", "battery_replacement_concern",
    "ev_knowledge_score", "monthly_energy_consumption_kwh", "monthly_charging_cost",
]

CATEGORICAL_FEATURES = [
    "education_level", "city_type", "current_vehicle_type",
    "home_charging_available", "previous_ev_experience",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[settings.target_column]
    return train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )


def get_preprocessing_info() -> dict:
    X_train, X_test, _, _ = get_splits()
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
