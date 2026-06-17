from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR
from src.preprocessing import LABEL_ENCODER_PATH, transform_features_for_prediction


VALID_LABELS = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}


def predict_once(model_name: str, df: pd.DataFrame) -> list[str]:
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    X_upload, _ = transform_features_for_prediction(df)
    return label_encoder.inverse_transform(model.predict(X_upload)).tolist()


def main() -> None:
    sample_path = PROJECT_ROOT / "data" / "sample_upload_predict.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample upload file: {sample_path}")

    df = pd.read_csv(sample_path)
    model_name = "logistic"
    first = predict_once(model_name, df)
    second = predict_once(model_name, df)

    assert first == second, "同一 CSV、同一模型重复预测结果不一致。"
    assert len(first) == len(df), "预测结果行数与上传样本数不一致。"
    assert set(first).issubset(VALID_LABELS), "预测类别不属于 7 个标准类别。"

    print("prediction consistency test passed")
    print(f"model={model_name}, rows={len(df)}, labels={sorted(set(first))}")


if __name__ == "__main__":
    main()
