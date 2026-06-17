from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

MODEL_LABELS = {
    "logistic": "逻辑回归",
    "knn": "K近邻",
    "svm": "支持向量机",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
}

NOISE_LABELS = {
    "clean": "干净数据",
    "gaussian": "高斯噪声",
    "mask": "掩码噪声",
}

CLASS_ORDER = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
PALETTE = ["#184E77", "#1E6091", "#1A759F", "#168AAD", "#34A0A4", "#52B69A", "#76C893"]
ACCENT = "#FFB703"
TEXT = "#102A43"
GRID = "#D9E2EC"


def setup_style() -> None:
    chosen_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    sns.set_theme(style="whitegrid", rc={"grid.color": GRID, "grid.linewidth": 0.8})
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": chosen_fonts,
            "axes.unicode_minus": False,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#BCCCDC",
            "axes.labelcolor": TEXT,
            "xtick.color": "#334E68",
            "ytick.color": "#334E68",
            "text.color": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#D9E2EC",
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )


def finish(fig, path: Path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def add_bar_labels(ax, orient: str = "v", fmt: str = "{:.4f}", offset: float = 0.002) -> None:
    for patch in ax.patches:
        if orient == "v":
            value = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            y = value + offset
            ax.text(x, y, fmt.format(value), ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            value = patch.get_width()
            x = value + offset
            y = patch.get_y() + patch.get_height() / 2
            ax.text(x, y, fmt.format(value), ha="left", va="center", fontsize=9, fontweight="bold")


def plot_class_distribution() -> None:
    df = pd.concat([pd.read_csv(DATA / name) for name in ["train.csv", "val.csv", "test.csv"]], ignore_index=True)
    counts = df["Class"].astype(str).str.strip().str.upper().value_counts().reindex(CLASS_ORDER).dropna().astype(int)
    percent = counts / counts.sum() * 100
    plot_df = pd.DataFrame({"类别": counts.index, "样本数": counts.values, "占比": percent.values})

    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = sns.color_palette(PALETTE, len(plot_df))
    bars = ax.bar(plot_df["类别"], plot_df["样本数"], color=colors, edgecolor="#0B2545", linewidth=0.8)
    for bar, pct in zip(bars, plot_df["占比"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + plot_df["样本数"].max() * 0.015,
            f"{int(bar.get_height())}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title("Dry Bean 七类别样本分布", pad=14)
    ax.set_xlabel("干豆类别")
    ax.set_ylabel("样本数量")
    ax.set_ylim(0, plot_df["样本数"].max() * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    finish(fig, FIGURES / "class_distribution.png")


def plot_correlation_heatmap() -> None:
    df = pd.concat([pd.read_csv(DATA / name) for name in ["train.csv", "val.csv", "test.csv"]], ignore_index=True)
    numeric = df.drop(columns=["Class"], errors="ignore").select_dtypes("number")
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = LinearSegmentedColormap.from_list("drybean_corr", ["#1D4E89", "#F8FAFC", "#B42318"])
    fig, ax = plt.subplots(figsize=(12.8, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="#E6EEF6",
        cbar_kws={"shrink": 0.72, "label": "Pearson 相关系数"},
        ax=ax,
    )
    ax.set_title("干豆形态学特征相关性热力图", pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    finish(fig, FIGURES / "feature_correlation_heatmap.png")


def plot_metric_bars(metric: str, output: str, title: str, ylabel: str) -> None:
    df = pd.read_csv(RESULTS / "metrics_summary.csv")
    df = df.sort_values(metric, ascending=False).copy()
    df["模型"] = df["model"].map(MODEL_LABELS)
    colors = [ACCENT if i == 0 else PALETTE[min(i, len(PALETTE) - 1)] for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(data=df, x="模型", y=metric, palette=colors, ax=ax, edgecolor="#0B2545", linewidth=0.8)
    add_bar_labels(ax, "v", "{:.4f}", 0.0015)
    ax.set_title(title, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ymin = max(0, df[metric].min() - 0.015)
    ymax = min(1.01, df[metric].max() + 0.02)
    ax.set_ylim(ymin, ymax)
    ax.axhline(df[metric].mean(), color="#D62828", linestyle="--", linewidth=1.2, label=f"平均值 {df[metric].mean():.4f}")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    finish(fig, FIGURES / output)


def plot_speed() -> None:
    df = pd.read_csv(RESULTS / "speed_summary.csv").sort_values("avg_predict_time_ms", ascending=True)
    df["模型"] = df["model"].map(MODEL_LABELS)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [ACCENT if i == 0 else PALETTE[min(i, len(PALETTE) - 1)] for i in range(len(df))]
    sns.barplot(data=df, y="模型", x="avg_predict_time_ms", palette=colors, ax=ax, edgecolor="#0B2545", linewidth=0.8)
    add_bar_labels(ax, "h", "{:.2f} ms", max(df["avg_predict_time_ms"].max() * 0.01, 0.3))
    ax.set_title("完整测试集平均推理时间对比", pad=14)
    ax.set_xlabel("平均推理时间（毫秒，越低越好）")
    ax.set_ylabel("")
    ax.set_xscale("symlog", linthresh=1)
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax)
    finish(fig, FIGURES / "speed_comparison.png")


def plot_loss_curve(csv_name: str, output: str, title: str, xlabel: str) -> None:
    path = RESULTS / csv_name
    if not path.exists():
        return
    df = pd.read_csv(path)
    x = df["epoch"] if "epoch" in df.columns else df["round"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(x, df["train_loss"], color="#1D4E89", linewidth=2.6, label="训练损失")
    ax.plot(x, df["test_loss"], color="#D62828", linewidth=2.6, linestyle="--", label="测试损失")
    ax.fill_between(x, df["train_loss"], df["test_loss"], color="#F77F00", alpha=0.10, label="泛化差距")
    ax.scatter([x.iloc[-1]], [df["test_loss"].iloc[-1]], color="#D62828", s=55, zorder=3)
    ax.set_title(title, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("对数损失")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine(ax=ax)
    finish(fig, FIGURES / output)


def plot_loss_comparison() -> None:
    curves = [
        ("逻辑回归", RESULTS / "loss_curve_logistic.csv", "#1D4E89"),
        ("XGBoost", RESULTS / "loss_curve_xgboost.csv", "#2A9D8F"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 6))
    plotted = False
    for label, path, color in curves:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        x = df["epoch"] if "epoch" in df.columns else df["round"]
        ax.plot(x, df["train_loss"], color=color, linewidth=2.4, label=f"{label} 训练损失")
        ax.plot(x, df["test_loss"], color=color, linewidth=2.4, linestyle="--", alpha=0.78, label=f"{label} 测试损失")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("训练型模型 Loss 曲线综合对比", pad=14)
    ax.set_xlabel("训练轮次 / 提升轮次")
    ax.set_ylabel("对数损失")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine(ax=ax)
    finish(fig, FIGURES / "loss_comparison.png")


def plot_robustness() -> None:
    df = pd.read_csv(RESULTS / "robustness_summary.csv")
    df = df[df["noise_type"] != "clean"].copy()
    df["模型"] = df["model"].map(MODEL_LABELS)
    df["噪声类型"] = df["noise_type"].map(NOISE_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, noise in zip(axes, ["高斯噪声", "掩码噪声"]):
        sub = df[df["噪声类型"] == noise]
        sns.lineplot(
            data=sub,
            x="noise_level",
            y="noisy_accuracy",
            hue="模型",
            marker="o",
            linewidth=2.4,
            markersize=7,
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title(noise)
        ax.set_xlabel("噪声强度")
        ax.set_ylabel("干净测试集准确率" if ax is axes[0] else "")
        ax.set_ylim(max(0.80, sub["noisy_accuracy"].min() - 0.02), min(1.0, sub["noisy_accuracy"].max() + 0.02))
        ax.grid(True, linestyle="--", alpha=0.5)
        sns.despine(ax=ax)
        if ax is axes[1]:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("训练数据加噪后的模型鲁棒性对比", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    finish(fig, FIGURES / "robustness_comparison.png")


def plot_robustness_heatmap() -> None:
    df = pd.read_csv(RESULTS / "robustness_pivot.csv")
    df = df.set_index("model")
    df.index = df.index.map(MODEL_LABELS)
    df.columns = [
        col.replace("gaussian", "高斯").replace("mask", "掩码").replace("_", "\n")
        for col in df.columns
    ]
    max_abs = max(abs(df.min().min()), abs(df.max().max()), 0.001)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    sns.heatmap(
        df,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.8,
        linecolor="#FFFFFF",
        cbar_kws={"label": "准确率下降（负值表示略有提升）"},
        ax=ax,
    )
    ax.set_title("不同噪声条件下模型准确率下降热力图", pad=14)
    ax.set_xlabel("噪声类型与强度")
    ax.set_ylabel("模型")
    finish(fig, FIGURES / "robustness_drop_heatmap.png")


def plot_confusion_matrices() -> None:
    for path in sorted(RESULTS.glob("confusion_matrix_*.csv")):
        model = path.stem.replace("confusion_matrix_", "")
        cm = pd.read_csv(path, index_col=0)
        cm = cm.reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
        norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
        annot = cm.astype(int).astype(str) + "\n" + (norm * 100).round(1).astype(str) + "%"
        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        sns.heatmap(
            norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=max(1.0, norm.max().max()),
            linewidths=0.6,
            linecolor="#E6EEF6",
            cbar_kws={"label": "按真实类别归一化比例"},
            ax=ax,
        )
        ax.set_title(f"{MODEL_LABELS.get(model, model)} 混淆矩阵（数量 / 行占比）", pad=14)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        finish(fig, FIGURES / f"confusion_matrix_{model}.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_class_distribution()
    plot_correlation_heatmap()
    plot_metric_bars("test_accuracy", "accuracy_comparison.png", "各模型测试集准确率对比", "测试集准确率")
    plot_metric_bars("f1_weighted", "f1_comparison.png", "各模型加权 F1-score 对比", "加权 F1-score")
    plot_speed()
    plot_loss_curve("loss_curve_logistic.csv", "loss_curve_logistic.png", "逻辑回归训练过程 Loss 曲线", "训练轮次")
    plot_loss_curve("loss_curve_xgboost.csv", "loss_curve_xgboost.png", "XGBoost 多分类 Logloss 曲线", "提升轮次")
    plot_loss_comparison()
    plot_robustness()
    plot_robustness_heatmap()
    plot_confusion_matrices()
    print("[OK] Enhanced figures regenerated in results/figures")


if __name__ == "__main__":
    main()
