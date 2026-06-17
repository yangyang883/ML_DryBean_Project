from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from .config import FIGURES_DIR, TARGET_COLUMN, ensure_dirs


def set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/NotoSansSC-VF.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        try:
            font_manager.fontManager.addfont(font_path)
        except FileNotFoundError:
            continue
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


MODEL_LABELS = {
    "logistic": "逻辑回归",
    "knn": "K近邻",
    "svm": "支持向量机",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
}


def chinese_font_properties():
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/NotoSansSC-VF.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        try:
            return font_manager.FontProperties(fname=font_path)
        except FileNotFoundError:
            continue
    return None


def _translated_models(df: pd.DataFrame) -> pd.DataFrame:
    translated = df.copy()
    if "model" in translated.columns:
        translated["model"] = translated["model"].map(lambda x: MODEL_LABELS.get(str(x), x))
    return translated


def plot_class_distribution(df: pd.DataFrame) -> None:
    ensure_dirs()
    set_style()
    plt.figure(figsize=(10, 5))
    order = df[TARGET_COLUMN].value_counts().index
    ax = sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        order=order,
        palette="viridis",
        legend=False,
    )
    ax.set_title("干豆类别分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数量")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    ensure_dirs()
    set_style()
    numeric_df = df.drop(columns=[TARGET_COLUMN], errors="ignore").select_dtypes("number")
    plt.figure(figsize=(13, 10))
    ax = sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, square=False)
    ax.set_title("特征相关性热力图")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name: str) -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    font_prop = chinese_font_properties()
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)} 混淆矩阵", fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=200)
    plt.close(fig)


def plot_metric_bar(metrics_df: pd.DataFrame, metric: str, output_name: str, title: str) -> None:
    ensure_dirs()
    if metrics_df.empty or metric not in metrics_df.columns:
        return
    set_style()
    plt.figure(figsize=(9, 5))
    plot_df = _translated_models(metrics_df)
    ax = sns.barplot(data=plot_df, x="model", y=metric, hue="model", palette="mako", legend=False)
    ax.set_title(title)
    ax.set_xlabel("模型")
    ax.set_ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=200)
    plt.close()


def plot_robustness(summary_df: pd.DataFrame) -> None:
    ensure_dirs()
    if summary_df.empty:
        return
    set_style()
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=summary_df,
        x="noise_level",
        y="noisy_accuracy",
        hue="model",
        style="noise_type",
        markers=True,
        dashes=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    translated_labels = [
        MODEL_LABELS.get(label, {"gaussian": "高斯噪声", "mask": "特征遮挡噪声", "clean": "干净数据"}.get(label, label))
        for label in labels
    ]
    ax.legend(handles, translated_labels, title="模型 / 噪声类型")
    ax.set_title("训练数据加噪后的鲁棒性对比")
    ax.set_xlabel("噪声强度")
    ax.set_ylabel("干净测试集准确率")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robustness_comparison.png", dpi=200)
    plt.close()


def plot_robustness_drop_heatmap(pivot_df: pd.DataFrame) -> None:
    """Plot accuracy-drop heatmap for paper screenshots."""
    ensure_dirs()
    if pivot_df.empty:
        return
    set_style()
    heatmap_df = pivot_df.set_index("model")
    heatmap_df.index = [MODEL_LABELS.get(str(index), str(index)) for index in heatmap_df.index]
    plt.figure(figsize=(13, 6))
    ax = sns.heatmap(heatmap_df, annot=True, fmt=".4f", cmap="Reds", linewidths=0.5)
    ax.set_title("不同噪声条件下模型精度下降对比")
    ax.set_xlabel("噪声类型与强度")
    ax.set_ylabel("模型")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robustness_drop_heatmap.png", dpi=200)
    plt.close()


def plot_speed(speed_df: pd.DataFrame) -> None:
    plot_df = speed_df.copy()
    if "avg_predict_time_ms" not in plot_df.columns and "inference_time_ms" in plot_df.columns:
        plot_df["avg_predict_time_ms"] = plot_df["inference_time_ms"]
    plot_metric_bar(
        plot_df,
        "avg_predict_time_ms",
        "speed_comparison.png",
        "完整测试集平均推理时间对比",
    )


def plot_loss_comparison() -> None:
    """Create one paper-friendly figure comparing available train/test loss curves."""
    ensure_dirs()
    loss_files = {
        "logistic": FIGURES_DIR.parent / "loss_curve_logistic.csv",
        "xgboost": FIGURES_DIR.parent / "loss_curve_xgboost.csv",
    }
    set_style()
    plt.figure(figsize=(10, 6))
    plotted = False
    for model_name, path in loss_files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        x_col = "epoch" if "epoch" in df.columns else "round"
        model_label = MODEL_LABELS.get(model_name, model_name)
        plt.plot(df[x_col], df["train_loss"], label=f"{model_label} 训练损失")
        plt.plot(df[x_col], df["test_loss"], linestyle="--", label=f"{model_label} 测试损失")
        plotted = True
    if plotted:
        plt.xlabel("训练轮次")
        plt.ylabel("对数损失")
        plt.title("损失曲线对比")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "loss_comparison.png", dpi=200)
    plt.close()
