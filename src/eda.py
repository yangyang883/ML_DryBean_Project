from __future__ import annotations

import pandas as pd

from .config import RESULTS_DIR, TARGET_COLUMN, TRAIN_PATH, TEST_PATH, VAL_PATH, ensure_dirs
from .data_loader import load_all_available
from .preprocessing import clean_dataframe
from .visualization import plot_class_distribution, plot_correlation_heatmap


def table_text(obj) -> str:
    """Render a compact plain-text table without optional tabulate dependency."""
    if isinstance(obj, pd.Series):
        return obj.to_frame("count").to_string()
    return obj.to_string(index=False)


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.drop(columns=[TARGET_COLUMN], errors="ignore").select_dtypes("number")
    rows = []
    for column in numeric_df.columns:
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((numeric_df[column] < lower) | (numeric_df[column] > upper)).sum())
        rows.append({"feature": column, "outlier_count": count, "lower_bound": lower, "upper_bound": upper})
    return pd.DataFrame(rows)


def run_eda() -> dict:
    ensure_dirs()
    raw_df = load_all_available()
    df = clean_dataframe(raw_df, drop_duplicates=False)
    plot_class_distribution(df)
    plot_correlation_heatmap(df)

    feature_count = df.drop(columns=[TARGET_COLUMN]).shape[1]
    train_rows = pd.read_csv(TRAIN_PATH).shape[0] if TRAIN_PATH.exists() else 0
    test_rows = pd.read_csv(TEST_PATH).shape[0] if TEST_PATH.exists() else 0
    val_rows = pd.read_csv(VAL_PATH).shape[0] if VAL_PATH.exists() else 0
    class_count = df[TARGET_COLUMN].nunique()
    raw_class_count = raw_df[TARGET_COLUMN].astype(str).str.strip().nunique()
    label_pollution_count = int((raw_df[TARGET_COLUMN].astype(str).str.strip() != df[TARGET_COLUMN]).sum())
    missing = raw_df.isna().sum().sort_values(ascending=False)
    duplicates = int(raw_df.duplicated().sum())
    class_distribution = df[TARGET_COLUMN].value_counts()
    outlier_df = detect_outliers_iqr(df)

    missing.to_csv(RESULTS_DIR / "missing_values.csv", header=["missing_count"])
    class_distribution.to_csv(RESULTS_DIR / "class_distribution.csv", header=["count"])
    outlier_df.to_csv(RESULTS_DIR / "outlier_summary.csv", index=False)

    report = f"""# Dry Bean Dataset Experiment Report

## 1. Dataset Overview

- Samples: {len(df)}
- Features: {feature_count}
- Raw label variants before cleaning: {raw_class_count}
- Cleaned classes: {class_count}
- Target column: `{TARGET_COLUMN}`
- Train/Test/Validation split: train={train_rows}, test={test_rows}, validation={val_rows}

Dry Bean Dataset is a public multi-class classification dataset originally built from images of dry bean grains. The goal is to classify bean variety from morphology features rather than raw images. The input features describe area, perimeter, major/minor axis length, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, and shape factors.

The dataset is used to classify dry bean varieties from 16 morphology features extracted from bean images. The provided files are intentionally dirty: they contain missing numeric values, duplicated rows, label format pollution, and numeric outliers.

## 1.1 Data Pollution Observed

- Missing values: mainly in `Perimeter` and `Solidity`.
- Duplicated rows: {duplicates}.
- Label noise: {label_pollution_count} rows have labels needing normalization, such as lowercase labels, extra spaces, `0/O` substitutions, and `3/E` substitutions.
- Outliers: several morphology features contain IQR outliers, which are kept because they may correspond to real large/small bean shapes.
- Class imbalance: BOMBAY has far fewer samples than DERMASON and SIRA, so macro/weighted metrics are both worth checking.
- Feature scale differences: area-like features are much larger than compactness/shape-factor features, so standardization is necessary for Logistic Regression, KNN, and SVM.
- Strong correlation: area, perimeter, convex area, and equivalent diameter are highly related morphology measurements, which may introduce redundant information but is useful for tree and margin-based models.

## 2. Class Distribution

```text
{table_text(class_distribution)}
```

## 3. Missing Values

```text
{table_text(missing[missing > 0]) if (missing > 0).any() else "No missing values were found."}
```

## 4. Duplicate Values

- Duplicate rows: {duplicates}

## 5. Outlier Analysis

Outliers were detected with the IQR rule for each numeric feature. Full details are saved to `results/outlier_summary.csv`.

```text
{table_text(outlier_df.sort_values("outlier_count", ascending=False).head(10))}
```

### 5.1 异常值处理说明

IQR 检测出的极端值不一定全部是错误数据。干豆不同类别之间形态差异较大，例如 BOMBAY、BARBUNYA、CALI 等类别在面积、周长和轴长上本来就可能有明显差别，所以直接删除大量 IQR 异常样本并不合适。本项目采用基于训练集上下界的裁剪（winsorization / clipping），而不是简单删除所有异常样本。这样既能减少极端值对模型的影响，也能尽量保留样本数量。

为了避免数据泄漏，median、IQR 上下界和 StandardScaler 都只在训练集上计算，验证集、测试集和上传预测数据只使用训练阶段保存下来的参数进行 transform。

## 6. Data Cleaning and Feature Engineering

- Duplicated rows are removed from the training data.
- Dirty labels are normalized into the seven canonical classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.
- Numeric missing values are imputed with training-set medians.
- IQR clipping bounds are fitted only on the training set and reused for test/uploaded data to avoid data leakage.
- Additional morphology ratio features are created, such as Area/Perimeter, MajorAxisLength/MinorAxisLength, ConvexArea/Area, and EquivDiameter/Perimeter. These ratios describe bean compactness and shape proportions.
- Labels are encoded by `LabelEncoder`.
- Features are standardized by `StandardScaler`.
- The fitted preprocessor, feature list, and label encoder are saved under `models/`.

## 7. Experiments

Run `python main.py --mode all` to train Logistic Regression, KNN, SVM, Random Forest, and XGBoost; evaluate accuracy, precision, recall, F1-score, confusion matrix, training time, inference speed, overfitting gap, loss curves, and robustness.
"""
    (RESULTS_DIR / "experiment_report.md").write_text(report, encoding="utf-8")
    print("[OK] EDA finished. Report saved to results/experiment_report.md")
    return {
        "samples": len(df),
        "features": feature_count,
        "classes": class_count,
        "duplicates": duplicates,
    }
