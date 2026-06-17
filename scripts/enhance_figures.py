from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_LABELS = {
    "logistic": "逻辑回归",
    "knn": "K近邻",
    "svm": "支持向量机",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
}

NOISE_LABELS = {
    "gaussian": "高斯噪声",
    "mask": "掩码噪声",
}

CLASS_ORDER = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
INFO_COLORS = ["#1488CC", "#28B5B5", "#9ACD32", "#F5A623", "#EF3E4A", "#6BCB77", "#4D96FF"]
TEXT = "#2C3E50"
MUTED = "#8A8F98"
LIGHT_GRID = "#EAECEF"


def setup_style() -> None:
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D3D8",
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "savefig.dpi": 340,
        }
    )


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def adjust_color(color: str, factor: float) -> tuple[float, float, float]:
    rgb = np.array(hex_to_rgb(color))
    if factor >= 1:
        return tuple(1 - (1 - rgb) / factor)
    return tuple(rgb * factor)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.5, 0.955, title, ha="center", va="top", fontsize=28, fontweight="bold", color="#666666")
    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha="center", va="top", fontsize=11, color=MUTED)


def draw_3d_column(
    ax: plt.Axes,
    x: float,
    bottom: float,
    top: float,
    color: str,
    width: float = 0.55,
    dx: float = 0.12,
    dy: float = 0.35,
    zorder: int = 3,
) -> None:
    height = top - bottom
    if height < 0:
        return
    shadow = Ellipse((x + dx * 0.9, bottom - dy * 0.20), width * 1.25, dy * 0.65, color="black", alpha=0.12, zorder=1)
    ax.add_patch(shadow)
    front = Rectangle((x - width / 2, bottom), width, height, facecolor=color, edgecolor="none", zorder=zorder)
    side = Polygon(
        [
            (x + width / 2, bottom),
            (x + width / 2 + dx, bottom + dy),
            (x + width / 2 + dx, top + dy),
            (x + width / 2, top),
        ],
        closed=True,
        facecolor=adjust_color(color, 0.62),
        edgecolor="none",
        zorder=zorder - 0.1,
    )
    top_face = Polygon(
        [
            (x - width / 2, top),
            (x + width / 2, top),
            (x + width / 2 + dx, top + dy),
            (x - width / 2 + dx, top + dy),
        ],
        closed=True,
        facecolor=adjust_color(color, 1.55),
        edgecolor="none",
        zorder=zorder + 0.1,
    )
    ax.add_patch(front)
    ax.add_patch(side)
    ax.add_patch(top_face)


def add_circle_label(ax: plt.Axes, x: float, y: float, text: str, color: str, radius: float = 0.32) -> None:
    circle = Circle((x, y), radius=radius, facecolor=color, edgecolor="white", linewidth=2.2, zorder=8)
    ax.add_patch(circle)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=9)


def clean_infographic_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B0B6BF")
    ax.tick_params(axis="both", length=0)
    ax.grid(axis=grid_axis, color=LIGHT_GRID, linewidth=1.0)
    if grid_axis == "y":
        ax.grid(axis="x", visible=False)
    else:
        ax.grid(axis="y", visible=False)


def plot_class_distribution() -> None:
    df = pd.concat([pd.read_csv(DATA_DIR / name) for name in ["train.csv", "val.csv", "test.csv"]], ignore_index=True)
    counts = df["Class"].astype(str).str.strip().str.upper().value_counts().reindex(CLASS_ORDER).dropna().astype(int)
    percents = counts / counts.sum() * 100
    x = np.arange(len(counts))

    fig = plt.figure(figsize=(14, 8))
    add_title(fig, "干豆类别分布", "七类干豆样本规模与占比，立体信息图风格展示")
    ax = fig.add_axes([0.08, 0.18, 0.62, 0.62])
    ax.set_ylim(0, counts.max() * 1.30)
    ax.set_xlim(-0.7, len(counts) - 0.25)
    for i, (label, value) in enumerate(counts.items()):
        color = INFO_COLORS[i % len(INFO_COLORS)]
        draw_3d_column(ax, i, 0, value, color, dy=counts.max() * 0.025)
        ax.text(i + 0.04, value + counts.max() * 0.075, f"{value}\n{percents[label]:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("样本数量", fontsize=11)
    clean_infographic_axes(ax)

    pie_ax = fig.add_axes([0.74, 0.24, 0.22, 0.42])
    pie_ax.pie(
        counts.values,
        labels=counts.index,
        colors=INFO_COLORS[: len(counts)],
        startangle=92,
        autopct="%1.1f%%",
        pctdistance=0.76,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 8.5, "color": TEXT},
    )
    pie_ax.text(0, 0, f"总样本数\n{counts.sum()}", ha="center", va="center", fontsize=16, fontweight="bold", color="#666666")
    pie_ax.set_title("类别占比", fontsize=16, fontweight="bold", color="#666666", pad=8)
    save(fig, FIGURES_DIR / "class_distribution.png")


def plot_metric_3d(metric: str, output: str, title: str, subtitle: str) -> None:
    df = pd.read_csv(RESULTS_DIR / "metrics_summary.csv").sort_values(metric, ascending=False)
    values = df[metric].to_numpy() * 100
    labels = df["model"].map(MODEL_LABELS).to_list()
    base = max(0, values.min() - 2.0)
    x = np.arange(len(values))
    fig = plt.figure(figsize=(13.5, 7.2))
    add_title(fig, title, subtitle)
    ax = fig.add_axes([0.08, 0.18, 0.86, 0.62])
    ax.set_ylim(base, values.max() + 3.3)
    ax.set_xlim(-0.65, len(values) - 0.2)
    for i, value in enumerate(values):
        color = "#EF3E4A" if i == 0 else INFO_COLORS[(i + 1) % len(INFO_COLORS)]
        draw_3d_column(ax, i, base, value, color, dy=(values.max() - base) * 0.045)
        add_circle_label(ax, i + 0.06, value + (values.max() - base) * 0.18, f"{value:.2f}%", color, radius=0.25)
    mean_value = values.mean()
    ax.axhline(mean_value, color="#C7CBD1", linewidth=2, zorder=0)
    ax.text(len(values) - 0.25, mean_value + 0.12, f"平均 {mean_value:.2f}%", fontsize=10, color=MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("百分比（%）", fontsize=11)
    clean_infographic_axes(ax)
    save(fig, FIGURES_DIR / output)


def plot_speed_performance() -> None:
    df = pd.read_csv(RESULTS_DIR / "speed_summary.csv").sort_values("avg_predict_time_ms", ascending=True)
    labels = df["model"].map(MODEL_LABELS).to_list()
    times = df["avg_predict_time_ms"].to_numpy()
    log_height = np.log10(times + 1)
    # Use a compressed visual index so very fast linear models do not make
    # every other model look almost flat. Real inference time is still shown.
    score = 55 + 45 * (log_height - log_height.min()) / (log_height.max() - log_height.min())
    x = np.arange(len(df))

    fig = plt.figure(figsize=(13.5, 7.2))
    add_title(fig, "模型推理时间对比", "柱高为压缩后的耗时指数，标签显示真实平均推理时间")
    ax = fig.add_axes([0.08, 0.18, 0.86, 0.62])
    ax.set_ylim(48, 112)
    ax.set_xlim(-0.65, len(df) - 0.2)
    for i, (s, t) in enumerate(zip(score, times)):
        color = "#1488CC" if i == 0 else INFO_COLORS[(i + 2) % len(INFO_COLORS)]
        draw_3d_column(ax, i, 0, s, color, dy=3.0)
        ax.text(
            i + 0.06,
            s + 2.8,
            f"{t:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.28,rounding_size=0.8", "facecolor": color, "edgecolor": "white", "linewidth": 1.6},
            zorder=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("压缩耗时指数", fontsize=11)
    clean_infographic_axes(ax)
    save(fig, FIGURES_DIR / "speed_comparison.png")


def plot_correlation_heatmap() -> None:
    df = pd.concat([pd.read_csv(DATA_DIR / name) for name in ["train.csv", "val.csv", "test.csv"]], ignore_index=True)
    numeric = df.drop(columns=["Class"], errors="ignore").select_dtypes("number")
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = LinearSegmentedColormap.from_list("infographic_corr", ["#1488CC", "#FFFFFF", "#EF3E4A"])
    fig = plt.figure(figsize=(13, 10))
    add_title(fig, "特征相关性热力图", "干豆形态学特征 Pearson 相关性，颜色越深表示相关越强")
    ax = fig.add_axes([0.10, 0.08, 0.76, 0.76])
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        linecolor="#F0F2F5",
        cbar_kws={"shrink": 0.68, "label": "Pearson 相关系数"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    save(fig, FIGURES_DIR / "feature_correlation_heatmap.png")


def plot_loss_curve(csv_name: str, output: str, title: str, xlabel: str) -> None:
    path = RESULTS_DIR / csv_name
    if not path.exists():
        return
    df = pd.read_csv(path)
    x = df["epoch"] if "epoch" in df.columns else df["round"]
    fig = plt.figure(figsize=(12, 6.8))
    add_title(fig, title, "圆点强调关键轮次，阴影区域表示训练与测试损失差距")
    ax = fig.add_axes([0.09, 0.17, 0.84, 0.62])
    ax.plot(x, df["train_loss"], color="#1488CC", linewidth=3.0, label="训练损失")
    ax.plot(x, df["test_loss"], color="#EF3E4A", linewidth=3.0, label="测试损失")
    ax.fill_between(x, df["train_loss"], df["test_loss"], color="#F5A623", alpha=0.13)
    for idx in np.linspace(0, len(x) - 1, 5, dtype=int):
        ax.scatter(x.iloc[idx], df["test_loss"].iloc[idx], s=120, color="#EF3E4A", edgecolor="white", linewidth=2, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("对数损失")
    ax.legend(loc="upper right")
    clean_infographic_axes(ax)
    save(fig, FIGURES_DIR / output)


def plot_loss_comparison() -> None:
    curves = [
        ("逻辑回归", RESULTS_DIR / "loss_curve_logistic.csv", "#1488CC"),
        ("XGBoost", RESULTS_DIR / "loss_curve_xgboost.csv", "#28B5B5"),
    ]
    fig = plt.figure(figsize=(12, 6.8))
    add_title(fig, "损失曲线综合对比", "训练型模型损失曲线综合对比")
    ax = fig.add_axes([0.09, 0.17, 0.84, 0.62])
    plotted = False
    for label, path, color in curves:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        x = df["epoch"] if "epoch" in df.columns else df["round"]
        ax.plot(x, df["train_loss"], color=color, linewidth=3.0, label=f"{label} 训练")
        ax.plot(x, df["test_loss"], color=color, linewidth=2.6, linestyle="--", alpha=0.75, label=f"{label} 测试")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("训练轮次 / 提升轮次")
    ax.set_ylabel("对数损失")
    ax.legend(ncol=2, loc="upper right")
    clean_infographic_axes(ax)
    save(fig, FIGURES_DIR / "loss_comparison.png")


def plot_robustness() -> None:
    df = pd.read_csv(RESULTS_DIR / "robustness_summary.csv")
    df = df[df["noise_type"] != "clean"].copy()
    df["模型"] = df["model"].map(MODEL_LABELS)
    df["噪声类型"] = df["noise_type"].map(NOISE_LABELS)
    fig = plt.figure(figsize=(14, 7.4))
    add_title(fig, "模型鲁棒性对比", "训练集加入不同噪声，测试集保持干净")
    axes = [fig.add_axes([0.07, 0.18, 0.40, 0.58]), fig.add_axes([0.55, 0.18, 0.40, 0.58])]
    for ax, noise in zip(axes, ["高斯噪声", "掩码噪声"]):
        sub = df[df["噪声类型"] == noise]
        for i, (model, g) in enumerate(sub.groupby("模型")):
            color = INFO_COLORS[i % len(INFO_COLORS)]
            ax.plot(g["noise_level"], g["noisy_accuracy"] * 100, color=color, linewidth=2.8, marker="o", markersize=8, label=model)
        ax.set_title(noise, fontsize=17, fontweight="bold", color="#666666")
        ax.set_xlabel("噪声强度")
        ax.set_ylabel("测试准确率（%）" if ax is axes[0] else "")
        ax.set_ylim(df["noisy_accuracy"].min() * 100 - 1.0, df["noisy_accuracy"].max() * 100 + 1.5)
        clean_infographic_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    save(fig, FIGURES_DIR / "robustness_comparison.png")


def plot_robustness_heatmap() -> None:
    df = pd.read_csv(RESULTS_DIR / "robustness_pivot.csv").set_index("model")
    df.index = df.index.map(MODEL_LABELS)
    df.columns = [c.replace("gaussian", "高斯").replace("mask", "掩码").replace("_", "\n") for c in df.columns]
    max_abs = max(abs(df.min().min()), abs(df.max().max()), 0.001)
    fig = plt.figure(figsize=(12.8, 6.8))
    add_title(fig, "准确率下降热力图", "负值表示低强度噪声带来轻微提升")
    ax = fig.add_axes([0.10, 0.17, 0.76, 0.62])
    sns.heatmap(
        df,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "准确率下降"},
        ax=ax,
    )
    ax.set_xlabel("噪声类型与强度")
    ax.set_ylabel("模型")
    save(fig, FIGURES_DIR / "robustness_drop_heatmap.png")


def plot_confusion_matrices() -> None:
    for path in sorted(RESULTS_DIR.glob("confusion_matrix_*.csv")):
        model = path.stem.replace("confusion_matrix_", "")
        cm = pd.read_csv(path, index_col=0).reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
        norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
        annot = cm.astype(int).astype(str) + "\n" + (norm * 100).round(1).astype(str) + "%"
        fig = plt.figure(figsize=(10, 8.5))
        add_title(fig, f"{MODEL_LABELS.get(model, model)} 混淆矩阵", "单元格显示：数量 / 真实类别行占比")
        ax = fig.add_axes([0.12, 0.12, 0.72, 0.70])
        sns.heatmap(
            norm,
            annot=annot,
            fmt="",
            cmap=LinearSegmentedColormap.from_list("cm", ["#F7FBFF", "#1488CC", "#0B3D91"]),
            vmin=0,
            vmax=1,
            linewidths=0.8,
            linecolor="#E8EEF3",
            cbar_kws={"label": "按真实类别归一化比例"},
            ax=ax,
        )
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        save(fig, FIGURES_DIR / f"confusion_matrix_{model}.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_class_distribution()
    plot_correlation_heatmap()
    plot_metric_3d("test_accuracy", "accuracy_comparison.png", "模型测试准确率对比", "红色立体柱表示当前最高模型")
    plot_metric_3d("f1_weighted", "f1_comparison.png", "模型加权 F1 对比", "加权 F1-score 体现类别不平衡下的综合表现")
    plot_speed_performance()
    plot_loss_curve("loss_curve_logistic.csv", "loss_curve_logistic.png", "逻辑回归损失曲线", "训练轮次")
    plot_loss_curve("loss_curve_xgboost.csv", "loss_curve_xgboost.png", "XGBoost 多分类损失曲线", "提升轮次")
    plot_loss_comparison()
    plot_robustness()
    plot_robustness_heatmap()
    plot_confusion_matrices()
    print("[OK] Infographic-style figures regenerated in results/figures")


if __name__ == "__main__":
    main()
