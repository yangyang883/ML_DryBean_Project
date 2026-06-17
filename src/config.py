from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
VAL_PATH = DATA_DIR / "val.csv"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2
NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20]

MODEL_NAMES = ["logistic", "knn", "svm", "random_forest", "xgboost"]


def ensure_dirs() -> None:
    """Create output directories used by the project."""
    for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
