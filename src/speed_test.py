from __future__ import annotations

import time

import pandas as pd

from .config import MODEL_DIR, MODEL_NAMES, RESULTS_DIR, ensure_dirs
from .data_loader import load_train_test
from .preprocessing import fit_transform_train, transform_dataset
from .utils import load_joblib
from .visualization import plot_speed


def run_speed_test(repeats: int = 20) -> pd.DataFrame:
    ensure_dirs()
    train_df, test_df = load_train_test(include_val_in_train=True)
    fit_transform_train(train_df)
    X_test, _, _, _ = transform_dataset(test_df)

    rows = []
    for model_name in MODEL_NAMES:
        model_path = MODEL_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            print(f"[SKIP] {model_name} model not found. Train it first.")
            continue
        model = load_joblib(model_path)
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict(X_test)
        total = time.perf_counter() - start
        rows.append(
            {
                "model": model_name,
                "repeats": repeats,
                "test_rows": X_test.shape[0],
                "avg_predict_time_ms": total / repeats * 1000,
                "avg_per_sample_us": total / repeats / X_test.shape[0] * 1_000_000,
            }
        )

    speed_df = pd.DataFrame(rows)
    speed_df.to_csv(RESULTS_DIR / "speed_summary.csv", index=False)
    plot_speed(speed_df)
    return speed_df
