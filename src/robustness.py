from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .config import MODEL_NAMES, NOISE_LEVELS, RANDOM_STATE, RESULTS_DIR, ensure_dirs
from .data_loader import load_train_test
from .preprocessing import fit_transform_train, transform_dataset
from .train import build_model
from .visualization import plot_robustness, plot_robustness_drop_heatmap


def add_gaussian_noise(X: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    return X + rng.normal(loc=0.0, scale=level, size=X.shape)


def add_mask_noise(X: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    X_noisy = X.copy()
    mask = rng.random(X.shape) < level
    X_noisy[mask] = 0.0
    return X_noisy


def robustness_note(accuracy_drop: float, noise_level: float) -> str:
    if accuracy_drop < 0:
        return "低强度噪声可能起到数据增强作用，模型泛化能力略有提升。"
    if accuracy_drop < 0.005:
        return "准确率下降很小，说明模型对该噪声条件较稳定。"
    if noise_level >= 0.10 and accuracy_drop >= 0.02:
        return "强噪声下准确率下降较明显，说明模型对该类扰动较敏感。"
    return "噪声带来一定精度下降，但整体仍保持可接受表现。"


def run_robustness() -> pd.DataFrame:
    ensure_dirs()
    train_df, test_df = load_train_test(include_val_in_train=True)
    X_train, y_train, _, _ = fit_transform_train(train_df)
    X_test, y_test, _, _ = transform_dataset(test_df)

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for model_name in MODEL_NAMES:
        model = build_model(model_name)
        if model_name == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)
        clean_acc = accuracy_score(y_test, model.predict(X_test))
        rows.append(
            {
                "model": model_name,
                "noise_type": "clean",
                "noise_level": 0.0,
                "clean_accuracy": clean_acc,
                "noisy_accuracy": clean_acc,
                "accuracy_drop": 0.0,
                "robustness_note": "干净训练集基准结果。",
                "protocol": "retrain_on_noisy_training_data",
            }
        )

        for level in NOISE_LEVELS:
            for noise_type, noise_func in [
                ("gaussian", add_gaussian_noise),
                ("mask", add_mask_noise),
            ]:
                X_train_noisy = noise_func(X_train, level, rng)
                noisy_model = build_model(model_name)
                if model_name == "xgboost":
                    noisy_model.fit(X_train_noisy, y_train, eval_set=[(X_test, y_test)], verbose=False)
                else:
                    noisy_model.fit(X_train_noisy, y_train)
                acc = accuracy_score(y_test, noisy_model.predict(X_test))
                drop = clean_acc - acc
                rows.append(
                    {
                        "model": model_name,
                        "noise_type": noise_type,
                        "noise_level": level,
                        "clean_accuracy": clean_acc,
                        "noisy_accuracy": acc,
                        "accuracy_drop": drop,
                        "robustness_note": robustness_note(drop, level),
                        "protocol": "retrain_on_noisy_training_data",
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "robustness_summary.csv", index=False)
    pivot = summary[summary["noise_type"] != "clean"].copy()
    pivot["noise_condition"] = pivot["noise_type"] + "_" + pivot["noise_level"].map(lambda value: f"{value:.2f}")
    pivot_df = pivot.pivot_table(
        index="model",
        columns="noise_condition",
        values="accuracy_drop",
        aggfunc="mean",
    ).reset_index()
    pivot_df.columns.name = None
    pivot_df.to_csv(RESULTS_DIR / "robustness_pivot.csv", index=False)
    plot_robustness(summary)
    plot_robustness_drop_heatmap(pivot_df)
    return summary
