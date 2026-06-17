# 工程系统与实验展示章节模板

## 1. 项目简介

本项目基于 Dry Bean Dataset 构建多分类机器学习系统。数据集由干豆图像提取出的 16 个形态学特征组成，目标是识别 BARBUNYA、BOMBAY、CALI、DERMASON、HOROZ、SEKER、SIRA 七种豆类。输入数据为 dirty 版本，包含缺失值、重复样本、异常值和标签污染，因此项目重点包含数据质量分析、数据清洗、特征工程、多算法建模和系统化展示。

建议截图：

- Streamlit 项目概览页
- 项目文件夹结构

## 2. 数据分析

项目首先对输入数据进行探索性分析，统计样本数、特征数、原始标签变体数和清洗后类别数。分析发现数据中存在以下污染：

- `Perimeter` 和 `Solidity` 等字段存在缺失值。
- 数据集中存在重复行。
- 标签存在大小写不一致、前后空格、`0/O` 和 `3/E` 字符替换等污染。
- 多个形态学特征存在 IQR 异常值，这些异常值可能对应真实豆类形态差异，因此不直接删除。

建议截图：

- `results/figures/class_distribution.png`
- `results/figures/feature_correlation_heatmap.png`
- Streamlit EDA 页面

## 3. 数据清洗与特征工程

本项目的数据处理流程如下：

1. 删除训练集重复样本。
2. 将 dirty 标签统一清洗为 7 个标准类别。
3. 将所有特征列转换为数值类型。
4. 使用训练集 median 对数值缺失值进行填补。
5. 使用 `LabelEncoder` 将类别标签编码为整数。
6. 使用 `StandardScaler` 对特征进行标准化。
7. 保存预处理器、标签编码器和特征列，保证测试集和上传预测使用同一套处理流程。

对应代码模块：

- `src/data_loader.py`
- `src/preprocessing.py`
- `src/eda.py`

建议截图：

- Streamlit 数据处理流程说明
- `results/experiment_report.md` 中的数据污染分析

## 4. 多算法实验分析

本项目实现了 5 种多分类算法：

- Logistic Regression
- KNN
- SVM
- Random Forest
- XGBoost

其中 Random Forest 和 XGBoost 属于课堂外扩展算法。每个模型统计训练集准确率、测试集准确率、precision、recall、F1-score、混淆矩阵、训练时间、推理速度和过拟合差异。

建议截图：

- Streamlit 模型结果表格
- `results/figures/accuracy_comparison.png`
- `results/figures/f1_comparison.png`
- 各模型混淆矩阵

## 5. Loss 曲线、速度、鲁棒性与过拟合分析

训练型算法保存 loss 曲线：

- Logistic Regression：`results/figures/loss_curve_logistic.png`
- XGBoost：`results/figures/loss_curve_xgboost.png`

推理速度分析对每个模型重复执行预测并取平均时间，结果保存在：

- `results/speed_summary.csv`
- `results/figures/speed_comparison.png`

鲁棒性分析对训练集加入 Gaussian Noise 和 Mask Noise，噪声强度为 0.01、0.05、0.10、0.20。每次加入噪声后重新训练模型，并在干净测试集上评估准确率下降情况。

输出文件：

- `results/robustness_summary.csv`
- `results/figures/robustness_comparison.png`

过拟合分析通过训练集准确率与测试集准确率差值判断，差值越大说明过拟合风险越高。

建议截图：

- loss 曲线
- 推理速度对比图
- 鲁棒性对比图
- Streamlit 鲁棒性与速度页面

## 6. 系统展示与统一调用

项目使用模块化工程结构，包含数据加载、预处理、训练、评估、鲁棒性、速度测试和可视化模块。所有实验通过统一命令行入口 `main.py` 调用。

示例命令：

```bash
python main.py --mode all
python main.py --mode train --model xgboost
python main.py --mode robustness
python main.py --mode speed
streamlit run app/streamlit_app.py
```

GitHub 仓库链接：

```text
待上传后填写
```

网页展示链接：

```text
待部署或课堂演示时填写
```

建议截图：

- GitHub 仓库首页
- README 页面
- Streamlit 展示页面首页
- 上传 CSV 预测页面

## 7. 课程总结

通过本课程和本项目，我学习了机器学习项目从数据到系统的完整流程，包括数据探索、数据清洗、特征工程、分类模型训练、模型评估、过拟合分析、鲁棒性分析和可视化展示。项目让我认识到，真实机器学习任务不仅需要追求模型精度，还需要关注数据质量、工程结构、实验可复现性和结果展示。

课程建议方面，希望后续可以增加更多真实 dirty 数据案例和模型部署实践，例如 Streamlit、Flask 或 FastAPI 展示，让学生更好地理解机器学习模型如何从实验走向应用。
