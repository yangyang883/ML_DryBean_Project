# Sanity Check Report

## 1. 数据划分重复检查

| 检查项 | 重复行数量 |
| --- | ---: |
| 原始 train 与 test 完全重复行 | 0 |
| 原始 train 与 val 完全重复行 | 0 |
| 原始 val 与 test 完全重复行 | 0 |
| 清洗后 train 与 test 完全重复行 | 0 |
| 清洗后 train 与 val 完全重复行 | 0 |
| 清洗后 val 与 test 完全重复行 | 0 |
| 实际训练加载后 train 与 test 重复行 | 0 |

说明：当前 train、val、test 三个数据文件之间未发现完全重复行。 最终用于训练的有效 train/test 重复行为 0。

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
| preprocessor.joblib | True |
| label_encoder.joblib | True |
| feature_columns.joblib | True |
| raw_feature_columns.joblib | True |

## 4. 结论

未发现测试集参与 median、IQR、scaler 的 fit。训练加载阶段已经剔除与测试集重复的训练样本，最终有效 train/test 无完全重复行。Streamlit 上传预测只加载训练阶段保存的模型和预处理工件，不重新训练模型，不重新 fit 预处理器。
