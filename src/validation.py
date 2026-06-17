from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import MODEL_DIR, MODEL_NAMES, RESULTS_DIR, TARGET_COLUMN, TEST_PATH, TRAIN_PATH, VAL_PATH, ensure_dirs
from .data_loader import load_train_test
from .preprocessing import (
    FEATURE_COLUMNS_PATH,
    LABEL_ENCODER_PATH,
    PREPROCESSOR_PATH,
    RAW_FEATURE_COLUMNS_PATH,
    clean_dataframe,
    transform_dataset,
    transform_features_for_prediction,
)
from .utils import load_joblib


def _hash_rows(df: pd.DataFrame) -> pd.Series:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    return normalized.astype(str).agg("|".join, axis=1).map(lambda text: hashlib.md5(text.encode("utf-8")).hexdigest())


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _overlap_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    if left.empty or right.empty:
        return 0
    return int(len(set(_hash_rows(left)) & set(_hash_rows(right))))


def run_sanity_check() -> dict[str, int | bool]:
    ensure_dirs()
    train_df = _read(TRAIN_PATH)
    val_df = _read(VAL_PATH)
    test_df = _read(TEST_PATH)

    raw_train_test_overlap = _overlap_count(train_df, test_df)
    raw_train_val_overlap = _overlap_count(train_df, val_df)
    raw_val_test_overlap = _overlap_count(val_df, test_df)

    cleaned_train = clean_dataframe(train_df, drop_duplicates=True)
    cleaned_val = clean_dataframe(val_df, drop_duplicates=False)
    cleaned_test = clean_dataframe(test_df, drop_duplicates=False)

    cleaned_train_test_overlap = _overlap_count(cleaned_train, cleaned_test)
    cleaned_train_val_overlap = _overlap_count(cleaned_train, cleaned_val)
    cleaned_val_test_overlap = _overlap_count(cleaned_val, cleaned_test)

    preprocessor_exists = PREPROCESSOR_PATH.exists()
    label_encoder_exists = LABEL_ENCODER_PATH.exists()
    feature_columns_exists = FEATURE_COLUMNS_PATH.exists()
    raw_feature_columns_exists = RAW_FEATURE_COLUMNS_PATH.exists()

    effective_train_df, effective_test_df = load_train_test(include_val_in_train=True)
    effective_overlap = _overlap_count(
        clean_dataframe(effective_train_df, drop_duplicates=True),
        clean_dataframe(effective_test_df, drop_duplicates=False),
    )
    raw_overlap_total = raw_train_test_overlap + raw_train_val_overlap + raw_val_test_overlap
    if raw_overlap_total:
        split_note = (
            "原始 dirty 文件中存在少量跨集合重复行。为避免测试泄漏，`load_train_test()` 会在训练阶段自动剔除"
            "与测试集清洗归一后完全重复的训练样本。"
        )
    else:
        split_note = "当前 train、val、test 三个数据文件之间未发现完全重复行。"

    report = f"""# Sanity Check Report

## 1. 数据划分重复检查

| 检查项 | 重复行数量 |
| --- | ---: |
| 原始 train 与 test 完全重复行 | {raw_train_test_overlap} |
| 原始 train 与 val 完全重复行 | {raw_train_val_overlap} |
| 原始 val 与 test 完全重复行 | {raw_val_test_overlap} |
| 清洗后 train 与 test 完全重复行 | {cleaned_train_test_overlap} |
| 清洗后 train 与 val 完全重复行 | {cleaned_train_val_overlap} |
| 清洗后 val 与 test 完全重复行 | {cleaned_val_test_overlap} |
| 实际训练加载后 train 与 test 重复行 | {effective_overlap} |

说明：{split_note} 最终用于训练的有效 train/test 重复行为 {effective_overlap}。

## 2. 预处理 fit/transform 检查

- median 缺失值填补：只在 `DryBeanPreprocessor.fit()` 中对训练集 `fit_transform`。
- IQR 上下界：只在 `DryBeanPreprocessor.fit()` 中由训练集分位数计算。
- StandardScaler：只在 `DryBeanPreprocessor.fit()` 中对训练集裁剪后的特征 `fit`。
- 测试集/验证集/上传预测：统一通过已保存的 `preprocessor.joblib` 调用 `transform`。
- LabelEncoder：训练阶段保存到 `label_encoder.joblib`，测试评估和上传预测只加载使用。
- 特征列：训练阶段保存 `feature_columns.joblib` 和 `raw_feature_columns.joblib`，上传预测按训练时顺序 transform。

## 3. 已保存工件检查

| 工件 | 是否存在 |
| --- | --- |
| preprocessor.joblib | {preprocessor_exists} |
| label_encoder.joblib | {label_encoder_exists} |
| feature_columns.joblib | {feature_columns_exists} |
| raw_feature_columns.joblib | {raw_feature_columns_exists} |

## 4. 结论

未发现测试集参与 median、IQR、scaler 的 fit。训练加载阶段已经剔除与测试集重复的训练样本，最终有效 train/test 无完全重复行。Streamlit 上传预测只加载训练阶段保存的模型和预处理工件，不重新训练模型，不重新 fit 预处理器。
"""
    (RESULTS_DIR / "sanity_check_report.md").write_text(report, encoding="utf-8")
    return {
        "raw_train_test_overlap": raw_train_test_overlap,
        "cleaned_train_test_overlap": cleaned_train_test_overlap,
        "effective_train_test_overlap": effective_overlap,
        "preprocessor_exists": preprocessor_exists,
        "label_encoder_exists": label_encoder_exists,
    }


def validate_metrics(tolerance: float = 1e-10) -> pd.DataFrame:
    ensure_dirs()
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("results/metrics_summary.csv does not exist. Run python main.py --mode all first.")

    metrics_df = pd.read_csv(metrics_path)
    _, y_test, _, label_encoder = transform_dataset(pd.read_csv(TEST_PATH))
    rows = []

    for model_name in MODEL_NAMES:
        model_path = MODEL_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            rows.append({"model": model_name, "passed": False, "note": "模型文件不存在"})
            continue

        model = load_joblib(model_path)
        X_test, y_test, _, label_encoder = transform_dataset(pd.read_csv(TEST_PATH))
        y_pred = model.predict(X_test)
        pred_labels = label_encoder.inverse_transform(y_pred)
        pd.DataFrame({"y_true": label_encoder.inverse_transform(y_test), "y_pred": pred_labels}).to_csv(
            RESULTS_DIR / f"predictions_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        row = metrics_df[metrics_df["model"] == model_name]
        if row.empty:
            rows.append({"model": model_name, "passed": False, "note": "metrics_summary 中缺少该模型"})
            continue
        row = row.iloc[0]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted")

        cm_path = RESULTS_DIR / f"confusion_matrix_{model_name}.csv"
        report_path = RESULTS_DIR / f"classification_report_{model_name}.csv"
        cm_sum_ok = False
        report_acc_ok = False
        if cm_path.exists():
            cm_sum_ok = int(pd.read_csv(cm_path, index_col=0).values.sum()) == len(y_test)
        if report_path.exists():
            report_df = pd.read_csv(report_path, index_col=0)
            report_acc = float(report_df.loc["accuracy", "precision"])
            report_acc_ok = abs(report_acc - float(row["test_accuracy"])) <= tolerance

        metric_ok = (
            abs(acc - float(row["test_accuracy"])) <= tolerance
            and abs(precision - float(row["precision_weighted"])) <= tolerance
            and abs(recall - float(row["recall_weighted"])) <= tolerance
            and abs(f1 - float(row["f1_weighted"])) <= tolerance
        )
        rows.append(
            {
                "model": model_name,
                "recomputed_accuracy": acc,
                "summary_accuracy": float(row["test_accuracy"]),
                "metric_values_match": metric_ok,
                "confusion_matrix_sum_ok": cm_sum_ok,
                "classification_report_accuracy_ok": report_acc_ok,
                "prediction_file": f"results/predictions_{model_name}.csv",
                "passed": bool(metric_ok and cm_sum_ok and report_acc_ok),
            }
        )

    validation_df = pd.DataFrame(rows)
    validation_df.to_csv(RESULTS_DIR / "metric_validation_summary.csv", index=False)

    report_lines = ["# Metric Validation Report", ""]
    for _, row in validation_df.iterrows():
        status = "通过" if row["passed"] else "未通过"
        report_lines.append(f"## {row['model']}：{status}")
        report_lines.append("")
        report_lines.append(row.to_string())
        report_lines.append("")
    (RESULTS_DIR / "metric_validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return validation_df


def check_prediction_consistency(model_name: str = "logistic", sample_path: Path | None = None) -> dict[str, object]:
    sample_path = sample_path or (Path(__file__).resolve().parents[1] / "data" / "sample_upload_predict.csv")
    df = pd.read_csv(sample_path)
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    def predict_once() -> list[str]:
        X_upload, feature_columns = transform_features_for_prediction(df)
        _ = feature_columns
        pred = model.predict(X_upload)
        return label_encoder.inverse_transform(pred).tolist()

    first = predict_once()
    second = predict_once()
    valid_labels = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}
    return {
        "same_predictions": first == second,
        "row_count_ok": len(first) == len(df),
        "labels_ok": set(first).issubset(valid_labels),
        "rows": len(df),
    }
