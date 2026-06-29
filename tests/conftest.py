import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

# Mirror the feature definitions so fixtures don't import from the service layer
_NUMERIC_FEATURES = [
    "age", "annual_income", "daily_commute_km", "weekly_travel_distance_km",
    "vehicle_age_years", "fuel_expense_per_month", "charging_station_accessibility",
    "nearest_charging_station_km", "electricity_cost_per_kwh",
    "environmental_awareness_score", "government_incentive_awareness",
    "technology_affinity_score", "range_anxiety_score", "battery_replacement_concern",
    "ev_knowledge_score", "monthly_energy_consumption_kwh", "monthly_charging_cost",
]
_CATEGORICAL_FEATURES = [
    "education_level", "city_type", "current_vehicle_type",
    "home_charging_available", "previous_ev_experience",
]
_TARGET_COL = "ev_adoption_likelihood"


@pytest.fixture(scope="session")
def sample_df():
    """150-row synthetic DataFrame — 50 rows per target class for stratified splits."""
    np.random.seed(42)
    n = 150
    data = {col: np.random.uniform(1, 100, n).tolist() for col in _NUMERIC_FEATURES}
    data["education_level"] = ["Bachelor", "Master", "High School"] * 50
    data["city_type"] = ["Urban", "Suburban", "Rural"] * 50
    data["current_vehicle_type"] = ["Gasoline", "Hybrid", "Electric"] * 50
    data["home_charging_available"] = ["Yes", "No"] * 75
    data["previous_ev_experience"] = ["Yes", "No"] * 75
    data[_TARGET_COL] = ["High", "Medium", "Low"] * 50
    return pd.DataFrame(data)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_features():
    """A single valid prediction input covering all 22 feature columns."""
    return {
        "age": 35, "annual_income": 75000, "daily_commute_km": 25,
        "weekly_travel_distance_km": 180, "vehicle_age_years": 5,
        "fuel_expense_per_month": 200, "charging_station_accessibility": 7,
        "nearest_charging_station_km": 3.5, "electricity_cost_per_kwh": 0.12,
        "environmental_awareness_score": 8, "government_incentive_awareness": 6,
        "technology_affinity_score": 9, "range_anxiety_score": 3,
        "battery_replacement_concern": 4, "ev_knowledge_score": 7,
        "monthly_energy_consumption_kwh": 350, "monthly_charging_cost": 45,
        "education_level": "Bachelor", "city_type": "Urban",
        "current_vehicle_type": "Gasoline", "home_charging_available": "Yes",
        "previous_ev_experience": "No",
    }
