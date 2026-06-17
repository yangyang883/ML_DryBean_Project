from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import FIGURES_DIR, MODEL_DIR, MODEL_NAMES, RESULTS_DIR, TARGET_COLUMN
from src.preprocessing import normalize_label, transform_features_for_prediction


MODEL_LABELS = {
    "logistic": "逻辑回归",
    "knn": "K近邻",
    "svm": "支持向量机",
    "random_forest": "随机森林",
    "xgboost": "XGBoost梯度提升树",
}

COLUMN_LABELS = {
    "model": "模型",
    "train_accuracy": "训练集准确率",
    "test_accuracy": "测试集准确率",
    "precision_weighted": "加权精确率",
    "recall_weighted": "加权召回率",
    "weighted_f1": "加权F1",
    "f1_weighted": "加权F1",
    "train_time_sec": "训练时间(秒)",
    "train_time_seconds": "训练时间(秒)",
    "avg_predict_time_ms": "平均推理时间(毫秒)",
    "inference_time_ms": "推理时间(毫秒)",
    "overfit_gap": "过拟合差值",
    "overfit_level": "过拟合等级",
    "overfit_comment": "过拟合说明",
    "loss_curve": "损失曲线",
    "model_strength": "模型优点",
    "model_weakness": "模型缺点",
    "final_comment": "最终评价",
    "noise_type": "噪声类型",
    "noise_level": "噪声强度",
    "clean_accuracy": "干净训练准确率",
    "noisy_train_accuracy": "加噪训练后准确率",
    "noisy_accuracy": "加噪训练后准确率",
    "accuracy_drop": "准确率下降",
    "robustness_note": "鲁棒性说明",
    "protocol": "实验流程",
    "repeats": "重复次数",
    "test_rows": "测试样本数",
    "avg_per_sample_us": "单样本平均耗时(微秒)",
    "missing_count": "缺失值数量",
    "Unnamed: 0": "类别/统计项",
    "precision": "精确率",
    "recall": "召回率",
    "f1-score": "F1分数",
    "support": "样本数",
    "feature": "特征",
    "outlier_count": "异常值数量",
    "lower_bound": "下界",
    "upper_bound": "上界",
}


st.set_page_config(
    page_title="干豆数据集多分类系统",
    page_icon="豆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def translate_model_name(value: object) -> object:
    return MODEL_LABELS.get(str(value), value)


def make_unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_columns: list[str] = []
    for column in columns:
        count = counts.get(column, 0)
        unique_columns.append(column if count == 0 else f"{column}_{count + 1}")
        counts[column] = count + 1
    return unique_columns


def translate_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    translated = df.copy()
    translated = translated.drop(
        columns=["weighted_f1", "avg_predict_time_ms", "classification_report"],
        errors="ignore",
    )
    if "model" in translated.columns:
        translated["model"] = translated["model"].map(translate_model_name)
    if "loss_curve" in translated.columns:
        translated["loss_curve"] = translated["loss_curve"].replace({"available": "已生成", "N/A": "不适用"})
    if "overfit_level" in translated.columns:
        translated["overfit_level"] = translated["overfit_level"].replace(
            {"low": "低", "medium": "中", "high": "高", "underfit": "可能欠拟合"}
        )
    if "noise_type" in translated.columns:
        translated["noise_type"] = translated["noise_type"].replace(
            {"clean": "干净数据", "gaussian": "高斯噪声", "mask": "特征遮挡噪声"}
        )
    for column in translated.columns:
        if column.startswith("Unnamed"):
            translated[column] = translated[column].replace(
                {
                    "accuracy": "准确率",
                    "macro avg": "宏平均",
                    "weighted avg": "加权平均",
                }
            )
    translated = translated.rename(columns=COLUMN_LABELS)
    translated.columns = make_unique_columns([str(column) for column in translated.columns])
    return translated


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.info(f"暂未生成：{path.name}。请先运行对应命令。")


def metric_card(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def run_prediction(input_df: pd.DataFrame, model_name: str) -> None:
    st.write("数据预览")
    st.dataframe(input_df.head(), use_container_width=True)

    model_path = MODEL_DIR / f"{model_name}.joblib"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    if not model_path.exists() or not encoder_path.exists():
        st.error("模型或标签编码器不存在，请先运行训练命令。")
        return

    try:
        X_upload, _ = transform_features_for_prediction(input_df)
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        pred = model.predict(X_upload)
        pred_label = label_encoder.inverse_transform(pred)
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_upload)
            confidence = proba.max(axis=1)

        result_df = input_df.copy()
        result_df.insert(0, "预测类别", pred_label)
        if confidence is not None:
            result_df.insert(1, "预测置信度", [f"{value:.4f}" for value in confidence])

        st.success("预测完成")
        st.markdown(f"**当前使用模型：** {translate_model_name(model_name)}")
        st.markdown(f"**上传样本数量：** {len(result_df)}")
        st.write("各类别预测数量统计")
        count_df = result_df["预测类别"].value_counts().rename_axis("预测类别").reset_index(name="数量")
        st.dataframe(count_df, use_container_width=True)

        st.write("预测结果")
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "下载预测结果 CSV",
            result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="drybean_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"预测失败：{exc}")


st.markdown(
    """
    <style>
    .main { background: #f7faf9; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6efec;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(33, 64, 55, 0.05);
    }
    h1, h2, h3 { color: #163b2f; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("干豆数据集多分类机器学习系统")
st.caption("数据分析 · 数据清洗 · 多算法实验 · 鲁棒性分析 · 推理速度 · 上传预测")

with st.sidebar:
    st.header("导航")
    page = st.radio(
        "选择页面",
        ["项目概览", "数据分析", "模型结果", "鲁棒性与速度", "上传预测"],
    )
    st.divider()
    st.write("常用命令")
    st.code("python main.py --mode all\nstreamlit run app/streamlit_app.py", language="bash")

if page == "项目概览":
    st.subheader("数据集介绍")
    st.write(
        "干豆数据集基于干豆图像提取 16 个形态学特征，用于识别 BARBUNYA、BOMBAY、CALI、"
        "DERMASON、HOROZ、SEKER、SIRA 七种豆类。当前输入数据是脏数据版本，包含缺失值、"
        "重复值、标签污染和异常值，因此需要先做数据清洗与特征工程。"
    )

    train_path = PROJECT_ROOT / "data" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "test.csv"
    train_df = load_csv_if_exists(train_path)
    test_df = load_csv_if_exists(test_path)
    clean_classes = train_df[TARGET_COLUMN].map(normalize_label).nunique() if TARGET_COLUMN in train_df else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("训练样本", str(len(train_df)) if not train_df.empty else "未找到")
    with col2:
        metric_card("测试样本", str(len(test_df)) if not test_df.empty else "未找到")
    with col3:
        metric_card("原始特征数", str(train_df.drop(columns=[TARGET_COLUMN], errors="ignore").shape[1]) if not train_df.empty else "未找到")
    with col4:
        metric_card("清洗后类别数", str(clean_classes) if clean_classes else "未找到")

    st.subheader("数据处理流程")
    st.markdown(
        """
        1. 读取训练集、验证集和测试集，检查标签列和特征列。
        2. 分析缺失值、重复值、异常值、类别不平衡和标签污染。
        3. 删除训练集重复样本。
        4. 将小写、空格、`0/O`、`3/E` 等脏标签统一为 7 个标准类别。
        5. 使用训练集的中位数填补数值缺失值。
        6. 使用训练集拟合 IQR 异常值裁剪上下界，测试集只复用该上下界。
        说明：异常值处理采用训练集 IQR 上下界裁剪，而不是简单删除样本，以减少极端值影响并保留数据量。
        7. 增加面积/周长、长轴/短轴等形态学比例特征。
        8. 使用标签编码和标准化，并保存预处理器，避免数据泄漏。
        """
    )

    st.subheader("算法实现")
    st.write(
        "- 逻辑回归：线性可解释基线模型。\n"
        "- K近邻：基于样本距离的非参数分类方法。\n"
        "- 支持向量机：使用 RBF 核，适合非线性分类边界。\n"
        "- 随机森林：课堂外扩展算法，集成多棵决策树，提升稳定性。\n"
        "- XGBoost梯度提升树：课堂外扩展算法，逐步修正前序模型错误。"
    )

elif page == "数据分析":
    st.subheader("探索性数据分析")
    st.markdown(
        """
        本页展示对脏数据的观察结果：类别分布、缺失值、异常值和特征相关性。
        完整文字报告已自动保存到 `results/experiment_report.md`。
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image(FIGURES_DIR / "class_distribution.png", "清洗后类别分布")
    with col2:
        show_image(FIGURES_DIR / "feature_correlation_heatmap.png", "特征相关性热力图")

    missing_df = load_csv_if_exists(RESULTS_DIR / "missing_values.csv")
    outlier_df = load_csv_if_exists(RESULTS_DIR / "outlier_summary.csv")
    if not missing_df.empty:
        st.write("缺失值统计")
        st.dataframe(translate_table(missing_df), use_container_width=True)
    if not outlier_df.empty:
        st.write("异常值 IQR 统计")
        st.dataframe(translate_table(outlier_df), use_container_width=True)

elif page == "模型结果":
    st.subheader("模型实验结果")
    metrics_df = load_csv_if_exists(RESULTS_DIR / "metrics_summary.csv")
    if metrics_df.empty:
        metrics_df = load_csv_if_exists(RESULTS_DIR / "model_metrics_summary.csv")
    if metrics_df.empty:
        st.warning("尚未生成模型结果，请运行 `python main.py --mode all`。")
    else:
        st.dataframe(translate_table(metrics_df), use_container_width=True)
        best = metrics_df.sort_values("test_accuracy", ascending=False).iloc[0]
        st.success(
            f"当前测试集准确率最高模型：{translate_model_name(best['model'])}，准确率={best['test_accuracy']:.4f}"
        )
        fastest = metrics_df.sort_values("inference_time_ms", ascending=True).iloc[0]
        lowest_gap = metrics_df.assign(abs_gap=metrics_df["overfit_gap"].abs()).sort_values("abs_gap").iloc[0]
        recommended = metrics_df.assign(score=metrics_df["test_accuracy"] - 0.2 * metrics_df["overfit_gap"].clip(lower=0)).sort_values("score", ascending=False).iloc[0]
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("准确率最高", translate_model_name(best["model"]), f"{best['test_accuracy']:.4f}")
        col_b.metric("推理最快", translate_model_name(fastest["model"]), f"{fastest['inference_time_ms']:.2f} ms")
        col_c.metric("过拟合差值最小", translate_model_name(lowest_gap["model"]), f"{lowest_gap['overfit_gap']:.4f}")
        col_d.metric("综合推荐模型", translate_model_name(recommended["model"]), f"{recommended['test_accuracy']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        show_image(FIGURES_DIR / "accuracy_comparison.png", "测试集准确率对比")
    with col2:
        show_image(FIGURES_DIR / "f1_comparison.png", "加权 F1 对比")

    st.subheader("损失曲线")
    st.info("K近邻、随机森林等模型不属于典型梯度迭代训练模型，因此不绘制传统意义上的损失曲线；本项目主要展示逻辑回归和 XGBoost 的训练/测试损失变化。")
    show_image(FIGURES_DIR / "loss_comparison.png", "损失曲线总对比")
    col1, col2 = st.columns(2)
    with col1:
        show_image(FIGURES_DIR / "loss_curve_logistic.png", "逻辑回归损失曲线")
    with col2:
        show_image(FIGURES_DIR / "loss_curve_xgboost.png", "XGBoost损失曲线")

    st.subheader("分类报告与混淆矩阵")
    tabs = st.tabs([MODEL_LABELS.get(name, name) for name in MODEL_NAMES])
    for tab, model_name in zip(tabs, MODEL_NAMES):
        with tab:
            show_image(FIGURES_DIR / f"confusion_matrix_{model_name}.png", f"{MODEL_LABELS.get(model_name, model_name)} 混淆矩阵")
            report_df = load_csv_if_exists(RESULTS_DIR / f"classification_report_{model_name}.csv")
            if not report_df.empty:
                st.dataframe(translate_table(report_df), use_container_width=True)

elif page == "鲁棒性与速度":
    st.subheader("鲁棒性分析")
    st.write("鲁棒性实验：对训练集加入高斯噪声或特征遮挡噪声，重新训练模型，再在干净测试集上评估准确率下降。")
    robustness_df = load_csv_if_exists(RESULTS_DIR / "robustness_summary.csv")
    if not robustness_df.empty:
        st.dataframe(translate_table(robustness_df), use_container_width=True)
    show_image(FIGURES_DIR / "robustness_comparison.png", "训练数据加噪后的鲁棒性对比")
    show_image(FIGURES_DIR / "robustness_drop_heatmap.png", "不同噪声条件下模型精度下降热力图")

    st.subheader("推理速度分析")
    speed_df = load_csv_if_exists(RESULTS_DIR / "speed_summary.csv")
    if not speed_df.empty:
        st.dataframe(translate_table(speed_df), use_container_width=True)
    show_image(FIGURES_DIR / "speed_comparison.png", "平均推理速度对比")

elif page == "上传预测":
    st.subheader("上传 CSV 进行预测")
    st.info("如果上传组件出现 403，请重启页面或点击下方按钮直接加载内置测试样本。")
    st.caption("预测阶段只加载训练阶段保存的模型、预处理器、标签编码器和特征列，不会在网页中重新训练模型。")
    label_to_model = {MODEL_LABELS.get(name, name): name for name in MODEL_NAMES}
    selected_label = st.selectbox("选择模型", list(label_to_model.keys()))
    model_name = label_to_model[selected_label]

    sample_path = PROJECT_ROOT / "data" / "sample_upload_predict.csv"
    if st.button("加载内置干豆测试样本"):
        if sample_path.exists():
            run_prediction(pd.read_csv(sample_path), model_name)
        else:
            st.error("未找到内置测试样本，请先生成 data/sample_upload_predict.csv。")

    uploaded_file = st.file_uploader("上传包含干豆特征列的 CSV 文件", type=["csv"])
    if uploaded_file is not None:
        run_prediction(pd.read_csv(uploaded_file), model_name)
