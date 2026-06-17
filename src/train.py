from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings

from .config import FIGURES_DIR, MODEL_DIR, MODEL_NAMES, RANDOM_STATE, RESULTS_DIR, ensure_dirs
from .data_loader import load_train_test
from .evaluate import evaluate_model
from .preprocessing import fit_transform_train, transform_dataset
from .utils import save_joblib
from .visualization import plot_loss_comparison


def build_model(model_name: str):
    xgb_model = None
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier

            xgb_model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Please run: pip install xgboost"
            ) from exc

    models = {
        "logistic": LogisticRegression(
            max_iter=800,
            solver="lbfgs",
            n_jobs=None,
            random_state=RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "svm": SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "xgboost": xgb_model,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(models)}.")
    return models[model_name]


def train_model(model_name: str) -> dict:
    ensure_dirs()
    train_df, test_df = load_train_test(include_val_in_train=True)
    X_train, y_train, _, label_encoder = fit_transform_train(train_df)
    X_test, y_test, _, _ = transform_dataset(test_df)

    model = build_model(model_name)
    start = time.perf_counter()
    if model_name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    save_joblib(model, MODEL_DIR / f"{model_name}.joblib")
    metrics = evaluate_model(model_name)
    metrics["train_time_sec"] = float(train_time)

    metrics_path = RESULTS_DIR / f"metrics_{model_name}.json"
    import json

    existing = json.loads(metrics_path.read_text(encoding="utf-8"))
    existing["train_time_sec"] = float(train_time)
    metrics_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    if model_name == "logistic":
        plot_logistic_loss_curve(X_train, y_train, X_test, y_test, label_encoder.classes_)
    elif model_name == "xgboost":
        plot_xgboost_loss_curve(model)

    print(f"[OK] trained {model_name}: test_acc={metrics['test_accuracy']:.4f}, time={train_time:.2f}s")
    return metrics


def train_all() -> pd.DataFrame:
    rows = [train_model(model_name) for model_name in MODEL_NAMES]
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "model_metrics_summary.csv", index=False)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    plot_loss_comparison()
    from .evaluate import evaluate_all

    return evaluate_all()


def plot_logistic_loss_curve(X_train, y_train, X_test, y_test, classes) -> None:
    """Record epoch loss with SGD logistic regression for a genuine training-process curve."""
    epochs = 80
    train_losses: list[float] = []
    test_losses: list[float] = []
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        alpha=0.0005,
        penalty="l2",
        random_state=RANDOM_STATE,
    )
    class_ids = np.arange(len(classes))
    for epoch in range(1, epochs + 1):
        clf.partial_fit(X_train, y_train, classes=class_ids)
        train_losses.append(log_loss(y_train, clf.predict_proba(X_train), labels=np.arange(len(classes))))
        test_losses.append(log_loss(y_test, clf.predict_proba(X_test), labels=np.arange(len(classes))))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="训练损失")
    plt.plot(range(1, epochs + 1), test_losses, label="测试损失")
    plt.xlabel("训练轮次")
    plt.ylabel("对数损失")
    plt.title("逻辑回归损失曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_curve_logistic.png", dpi=200)
    plt.close()
    pd.DataFrame(
        {"epoch": range(1, epochs + 1), "train_loss": train_losses, "test_loss": test_losses}
    ).to_csv(RESULTS_DIR / "loss_curve_logistic.csv", index=False)


def plot_xgboost_loss_curve(model) -> None:
    """Plot XGBoost train/test multi-class log loss recorded during boosting."""
    results = model.evals_result()
    if not results:
        return
    train_key, test_key = list(results.keys())[:2]
    train_loss = results[train_key]["mlogloss"]
    test_loss = results[test_key]["mlogloss"]
    rounds = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_loss, label="训练多分类损失")
    plt.plot(rounds, test_loss, label="测试多分类损失")
    plt.xlabel("提升轮次")
    plt.ylabel("多分类对数损失")
    plt.title("XGBoost 损失曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_curve_xgboost.png", dpi=200)
    plt.close()
    pd.DataFrame(
        {"round": rounds, "train_loss": train_loss, "test_loss": test_loss}
    ).to_csv(RESULTS_DIR / "loss_curve_xgboost.csv", index=False)
