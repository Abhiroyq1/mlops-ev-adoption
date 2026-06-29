from pathlib import Path
from app.config import settings, DATA_PATH, MODELS_DIR


def test_default_target_column():
    assert settings.target_column == "ev_adoption_likelihood"


def test_default_test_size():
    assert settings.test_size == 0.2


def test_default_random_state():
    assert settings.random_state == 42


def test_data_path_is_path_object():
    assert isinstance(DATA_PATH, Path)


def test_models_dir_is_path_object():
    assert isinstance(MODELS_DIR, Path)


def test_models_dir_exists():
    # MODELS_DIR is created at import time via mkdir(exist_ok=True)
    assert MODELS_DIR.exists()


def test_test_size_is_fraction():
    assert 0 < settings.test_size < 1


def test_settings_data_path_string():
    assert isinstance(settings.data_path, str)
    assert settings.data_path.endswith(".csv")
