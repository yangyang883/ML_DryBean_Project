from __future__ import annotations

import time

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import MODEL_DIR, MODEL_NAMES, RESULTS_DIR, ensure_dirs
from .data_loader import load_train_test
from .preprocessing import fit_transform_train, transform_dataset
from .utils import load_joblib, save_json, load_json
from .visualization import plot_confusion_matrix, plot_metric_bar, plot_loss_comparison


def analyze_overfit(train_acc: float, test_acc: float) -> tuple[float, str, str]:
    gap = train_acc - test_acc
    if gap >= 0.05:
        return gap, "高", "训练集准确率明显高于测试集，存在一定过拟合。"
    if gap >= 0.01:
        return gap, "中", "训练集和测试集存在一定差距，有轻度过拟合迹象。"
    if train_acc < 0.85 and test_acc < 0.85:
        return gap, "可能欠拟合", "训练集和测试集精度都偏低，模型可能欠拟合。"
    return gap, "低", "训练集和测试集表现接近，模型泛化能力较好。"


MODEL_EXPLANATIONS = {
    "logistic": {
        "strength": "推理速度快，泛化稳定，可解释性强，适合作为基线模型。",
        "weakness": "表达能力有限，对复杂非线性分类边界的拟合能力不如核方法和集成模型。",
    },
    "knn": {
        "strength": "实现简单，属于非参数模型，对局部样本分布有较直接的刻画能力。",
        "weakness": "推理速度较慢，对距离度量和特征标准化比较敏感。",
    },
    "svm": {
        "strength": "测试准确率较高，RBF 核适合处理非线性分类边界。",
        "weakness": "推理速度相对较慢，对参数和数据规模比较敏感。",
    },
    "random_forest": {
        "strength": "集成多棵决策树，抗噪能力较好，对特征尺度不敏感。",
        "weakness": "训练集准确率过高时可能存在过拟合，模型解释性弱于线性模型。",
    },
    "xgboost": {
        "strength": "提升树模型能够逐步修正前序模型错误，通常适合结构化表格数据。",
        "weakness": "参数较多，训练过程更复杂，若控制不好可能出现一定过拟合。",
    },
}


def final_comment(model_name: str, test_acc: float, inference_ms: float, overfit_level: str) -> str:
    if overfit_level == "高":
        return "该模型训练集准确率明显高于测试集，存在一定过拟合，需要结合泛化表现谨慎选择。"
    if test_acc >= 0.93 and inference_ms > 100:
        return "该模型测试准确率较高，但推理速度相对较慢，适合更重视精度的场景。"
    if test_acc >= 0.93:
        return "该模型测试准确率较高，泛化表现较好，是本任务中的有竞争力模型。"
    if inference_ms < 5:
        return "该模型推理速度很快，训练集和测试集表现较接近，适合作为稳定基线。"
    return "该模型整体表现较稳定，可作为多模型对比中的参考方案。"


def evaluate_model(model_name: str, repeat_predict: int = 5) -> dict:
    """Evaluate one saved model and write metrics, report and confusion matrix."""
    ensure_dirs()
    model_path = MODEL_DIR / f"{model_name}.joblib"
    model = load_joblib(model_path)

    train_df, test_df = load_train_test(include_val_in_train=True)
    X_train, y_train, _, label_encoder = fit_transform_train(train_df)
    X_test, y_test, _, label_encoder = transform_dataset(test_df)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    start = time.perf_counter()
    for _ in range(repeat_predict):
        model.predict(X_test)
    avg_predict_time_ms = (time.perf_counter() - start) / repeat_predict * 1000

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    weighted_precision = precision_score(y_test, test_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_test, test_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, test_pred, average="weighted")
    gap, overfit_level, overfit_comment = analyze_overfit(train_acc, test_acc)

    report_dict = classification_report(
        y_test,
        test_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        test_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, test_pred)

    pd.DataFrame(report_dict).transpose().to_csv(RESULTS_DIR / f"classification_report_{model_name}.csv")
    pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
        RESULTS_DIR / f"confusion_matrix_{model_name}.csv"
    )
    plot_confusion_matrix(cm, label_encoder.classes_, model_name)

    previous_metrics = {}
    metrics_path = RESULTS_DIR / f"metrics_{model_name}.json"
    if metrics_path.exists():
        previous_metrics = load_json(metrics_path)

    metrics = {
        "model": model_name,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "f1_weighted": float(weighted_f1),
        "avg_predict_time_ms": float(avg_predict_time_ms),
        "inference_time_ms": float(avg_predict_time_ms),
        "overfit_gap": float(gap),
        "overfit_level": overfit_level,
        "overfit_comment": overfit_comment,
        "loss_curve": "available" if model_name in {"logistic", "xgboost"} else "N/A",
        "model_strength": MODEL_EXPLANATIONS[model_name]["strength"],
        "model_weakness": MODEL_EXPLANATIONS[model_name]["weakness"],
        "final_comment": final_comment(model_name, test_acc, avg_predict_time_ms, overfit_level),
        "classification_report": report_text,
    }
    if "train_time_sec" in previous_metrics:
        metrics["train_time_sec"] = previous_metrics["train_time_sec"]
        metrics["train_time_seconds"] = previous_metrics["train_time_sec"]
    save_json(metrics, metrics_path)
    return metrics


def evaluate_all() -> pd.DataFrame:
    rows = []
    for model_name in MODEL_NAMES:
        if (MODEL_DIR / f"{model_name}.joblib").exists():
            rows.append(evaluate_model(model_name))
    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df.to_csv(RESULTS_DIR / "model_metrics_summary.csv", index=False)
        metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
        plot_metric_bar(metrics_df, "test_accuracy", "accuracy_comparison.png", "各模型测试集准确率对比")
        plot_metric_bar(metrics_df, "f1_weighted", "f1_comparison.png", "各模型加权 F1 对比")
        plot_loss_comparison()
        write_final_experiment_report(metrics_df)
    return metrics_df


def _df_text(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def write_final_experiment_report(metrics_df: pd.DataFrame) -> None:
    """Append final model, robustness, speed, and course-summary analysis to the report."""
    report_path = RESULTS_DIR / "experiment_report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Dry Bean Dataset Experiment Report\n"
    base = existing.split("## 8. Final Model Analysis")[0].rstrip()

    best = metrics_df.sort_values("test_accuracy", ascending=False).iloc[0]
    model_table = metrics_df[
        [
            "model",
            "train_accuracy",
            "test_accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "train_time_seconds",
            "inference_time_ms",
            "overfit_gap",
            "overfit_level",
            "model_strength",
            "model_weakness",
            "final_comment",
            "loss_curve",
        ]
    ].copy()

    overfit_lines = []
    for _, row in metrics_df.iterrows():
        overfit_lines.append(f"- {row['model']}: gap={row['overfit_gap']:.4f}. {row['overfit_comment']}")

    speed_path = RESULTS_DIR / "speed_summary.csv"
    robustness_path = RESULTS_DIR / "robustness_summary.csv"
    speed_text = "Speed test has not been run yet."
    if speed_path.exists():
        speed_df = pd.read_csv(speed_path)
        speed_text = _df_text(speed_df.head(10))

    robustness_text = "鲁棒性实验尚未运行。"
    robustness_analysis = "鲁棒性实验尚未运行。"
    if robustness_path.exists():
        robust_df = pd.read_csv(robustness_path)
        robustness_text = _df_text(robust_df.head(15))
        noisy = robust_df[robust_df["noise_type"] != "clean"].copy()
        model_drop = noisy.groupby("model")["accuracy_drop"].mean().sort_values()
        best_robust = model_drop.index[0]
        sensitive = model_drop.index[-1]
        robustness_analysis = (
            f"从平均准确率下降看，{best_robust} 的下降最小，说明整体鲁棒性最好；"
            f"{sensitive} 的平均下降最大，对训练数据噪声更敏感。低强度噪声下个别模型准确率略有上升，"
            "这通常可以理解为噪声起到轻微数据增强作用，使模型泛化能力略有提升；"
            "当噪声强度增大时，特征信息被扰动或遮挡更多，模型准确率整体更容易下降。"
        )

    confusion_analysis = build_confusion_analysis(metrics_df)

    appendix = f"""

## 8. Final Model Analysis

The best model on the clean test set is **{best['model']}**, with test accuracy **{best['test_accuracy']:.4f}** and weighted F1 **{best['f1_weighted']:.4f}**.

```text
{_df_text(model_table)}
```

### 8.1 Overfitting Analysis

{chr(10).join(overfit_lines)}

### 8.2 Loss Curve Analysis

Logistic Regression and XGBoost provide training-process loss curves. KNN and Random Forest are not gradient-epoch training models in this implementation, so their loss curves are recorded as N/A instead of being artificially generated.

Generated figures:

- `results/figures/loss_curve_logistic.png`
- `results/figures/loss_curve_xgboost.png`
- `results/figures/loss_comparison.png`

### 8.3 Inference Speed Analysis

```text
{speed_text}
```

### 8.4 Robustness Analysis

Robustness follows the assignment requirement: Gaussian Noise and Mask Noise are added to the training data, each noisy model is retrained, and accuracy drop is measured on the clean test set.

```text
{robustness_text}
```

{robustness_analysis}

### 8.5 Confusion Matrix and Classification Report Analysis

{confusion_analysis}

## 9. Engineering and Reproducibility

All experiment stages are called through `main.py`, and the Streamlit UI is only used for result display and CSV prediction. The preprocessing artifacts are fitted only on the training set and reused for validation, test, robustness evaluation, speed testing, and uploaded CSV prediction to avoid data leakage.

## 10. Course Summary

通过这门课和这个项目，我把课堂里学到的分类算法真正串成了一个完整流程。以前更关注模型怎么写、准确率是多少，现在更能理解数据清洗、特征工程、训练/测试隔离、过拟合分析、鲁棒性和工程复现同样重要。这个项目也让我练习了把实验结果整理成报告和网页展示的过程。

我觉得课程整体比较实用，能帮助我建立机器学习项目的基本思路。建议后续课程可以多加入一些 dirty data 和部署展示的案例，因为真实项目里数据往往并不干净，只会训练模型还不够，还要能解释结果、复现实验并展示给别人看。
"""
    report_path.write_text(base + appendix, encoding="utf-8")


def build_confusion_analysis(metrics_df: pd.DataFrame) -> str:
    best_model = metrics_df.sort_values("test_accuracy", ascending=False).iloc[0]["model"]
    report_path = RESULTS_DIR / f"classification_report_{best_model}.csv"
    cm_path = RESULTS_DIR / f"confusion_matrix_{best_model}.csv"
    if not report_path.exists() or not cm_path.exists():
        return "分类报告或混淆矩阵尚未生成。"

    report = pd.read_csv(report_path, index_col=0)
    class_report = report.loc[[idx for idx in report.index if idx not in {"accuracy", "macro avg", "weighted avg"}]]
    best_classes = class_report["f1-score"].sort_values(ascending=False).head(2).index.tolist()
    weak_classes = class_report["f1-score"].sort_values(ascending=True).head(2).index.tolist()

    cm = pd.read_csv(cm_path, index_col=0)
    confusions = []
    for true_class in cm.index:
        row = cm.loc[true_class].copy()
        row[true_class] = 0
        pred_class = row.idxmax()
        count = int(row.max())
        if count > 0:
            confusions.append((true_class, pred_class, count))
    confusions = sorted(confusions, key=lambda item: item[2], reverse=True)[:3]
    confusion_text = "；".join([f"{a} 容易被分到 {b}（{c} 个样本）" for a, b, c in confusions])

    return (
        f"以测试准确率最高的 {best_model} 为例，F1 表现较好的类别包括 {', '.join(best_classes)}，"
        f"相对更容易出错的类别包括 {', '.join(weak_classes)}。混淆矩阵中较明显的混淆关系为：{confusion_text}。"
        "这些混淆是合理的，因为 SIRA、DERMASON、CALI、BARBUNYA 等类别在部分面积、周长、轴长和紧致度特征上比较接近，"
        "模型容易把形态学特征相似的豆类分到相邻类别。"
    )
