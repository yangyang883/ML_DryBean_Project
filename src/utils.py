import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def friendly_file_check(path: Path, purpose: str = "file") -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {purpose}: {path}. Please put the required file in this location."
        )


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    friendly_file_check(path, "JSON result file")
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    friendly_file_check(path, "joblib artifact")
    return joblib.load(path)


def read_csv(path: Path) -> pd.DataFrame:
    friendly_file_check(path, "CSV data file")
    return pd.read_csv(path)

