from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    data_path: str = "data/global_ev_adoption_behavior_2026.csv"
    models_dir: str = "models"
    target_column: str = "ev_adoption_likelihood"
    test_size: float = 0.2
    random_state: int = 42

    model_config = {"env_file": ".env"}


settings = Settings()
DATA_PATH = Path(settings.data_path)
MODELS_DIR = Path(settings.models_dir)
MODELS_DIR.mkdir(exist_ok=True)
