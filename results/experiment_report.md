# Dry Bean Dataset Experiment Report

## 1. Dataset Overview

- Samples: 13578
- Features: 16
- Raw label variants before cleaning: 18
- Cleaned classes: 7
- Target column: `Class`
- Train/Test/Validation split: train=9509, test=2737, validation=1332

Dry Bean Dataset is a public multi-class classification dataset originally built from images of dry bean grains. The goal is to classify bean variety from morphology features rather than raw images. The input features describe area, perimeter, major/minor axis length, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, and shape factors.

The dataset is used to classify dry bean varieties from 16 morphology features extracted from bean images. The provided files are intentionally dirty: they contain missing numeric values, duplicated rows, label format pollution, and numeric outliers.

## 1.1 Data Pollution Observed

- Missing values: mainly in `Perimeter` and `Solidity`.
- Duplicated rows: 21.
- Label noise: 451 rows have labels needing normalization, such as lowercase labels, extra spaces, `0/O` substitutions, and `3/E` substitutions.
- Outliers: several morphology features contain IQR outliers, which are kept because they may correspond to real large/small bean shapes.
- Class imbalance: BOMBAY has far fewer samples than DERMASON and SIRA, so macro/weighted metrics are both worth checking.
- Feature scale differences: area-like features are much larger than compactness/shape-factor features, so standardization is necessary for Logistic Regression, KNN, and SVM.
- Strong correlation: area, perimeter, convex area, and equivalent diameter are highly related morphology measurements, which may introduce redundant information but is useful for tree and margin-based models.

## 2. Class Distribution

```text
          count
Class          
DERMASON   3546
SIRA       2636
SEKER      2027
HOROZ      1895
CALI       1630
BARBUNYA   1322
BOMBAY      522
```

## 3. Missing Values

```text
           count
Perimeter    676
Solidity     388
```

## 4. Duplicate Values

- Duplicate rows: 21

## 5. Outlier Analysis

Outliers were detected with the IQR rule for each numeric feature. Full details are saved to `results/outlier_summary.csv`.

```text
        feature  outlier_count  lower_bound   upper_bound
   Eccentricity            843     0.574133      0.951605
   ShapeFactor4            763     0.987441      1.004155
       Solidity            734     0.979145      0.996546
           Area            578 -1269.500000  98776.500000
MinorAxisLength            567   113.910260    279.101147
     ConvexArea            550 -1750.750000 100745.250000
   ShapeFactor1            533     0.003837      0.009332
  EquivDiameter            526   118.282239    376.184186
   AspectRation            481     1.021321      2.115922
      Perimeter            478   292.147375   1388.698375
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

## 8. Final Model Analysis

The best model on the clean test set is **svm**, with test accuracy **0.9342** and weighted F1 **0.9343**.

```text
        model  train_accuracy  test_accuracy  precision_weighted  recall_weighted  f1_weighted  train_time_seconds  inference_time_ms  overfit_gap overfit_level                 model_strength                    model_weakness                          final_comment loss_curve
     logistic        0.925950       0.926562            0.927276         0.926562     0.926810            0.434792            0.71678    -0.000612             低     推理速度快，泛化稳定，可解释性强，适合作为基线模型。 表达能力有限，对复杂非线性分类边界的拟合能力不如核方法和集成模型。       该模型推理速度很快，训练集和测试集表现较接近，适合作为稳定基线。  available
          knn        1.000000       0.920351            0.921286         0.920351     0.920670            0.002638          100.00612     0.079649             高 实现简单，属于非参数模型，对局部样本分布有较直接的刻画能力。           推理速度较慢，对距离度量和特征标准化比较敏感。 该模型训练集准确率明显高于测试集，存在一定过拟合，需要结合泛化表现谨慎选择。        N/A
          svm        0.939077       0.934235            0.934457         0.934235     0.934306            4.552700          739.94124     0.004843             低      测试准确率较高，RBF 核适合处理非线性分类边界。            推理速度相对较慢，对参数和数据规模比较敏感。       该模型测试准确率较高，但推理速度相对较慢，适合更重视精度的场景。        N/A
random_forest        1.000000       0.918524            0.918829         0.918524     0.918629           19.786068          140.17530     0.081476             高       集成多棵决策树，抗噪能力较好，对特征尺度不敏感。     训练集准确率过高时可能存在过拟合，模型解释性弱于线性模型。 该模型训练集准确率明显高于测试集，存在一定过拟合，需要结合泛化表现谨慎选择。        N/A
      xgboost        0.984561       0.925466            0.925830         0.925466     0.925614            9.857015          135.58652     0.059096             高 提升树模型能够逐步修正前序模型错误，通常适合结构化表格数据。      参数较多，训练过程更复杂，若控制不好可能出现一定过拟合。 该模型训练集准确率明显高于测试集，存在一定过拟合，需要结合泛化表现谨慎选择。  available
```

### 8.1 Overfitting Analysis

- logistic: gap=-0.0006. 训练集和测试集表现接近，模型泛化能力较好。
- knn: gap=0.0796. 训练集准确率明显高于测试集，存在一定过拟合。
- svm: gap=0.0048. 训练集和测试集表现接近，模型泛化能力较好。
- random_forest: gap=0.0815. 训练集准确率明显高于测试集，存在一定过拟合。
- xgboost: gap=0.0591. 训练集准确率明显高于测试集，存在一定过拟合。

### 8.2 Loss Curve Analysis

Logistic Regression and XGBoost provide training-process loss curves. KNN and Random Forest are not gradient-epoch training models in this implementation, so their loss curves are recorded as N/A instead of being artificially generated.

Generated figures:

- `results/figures/loss_curve_logistic.png`
- `results/figures/loss_curve_xgboost.png`
- `results/figures/loss_comparison.png`

### 8.3 Inference Speed Analysis

```text
        model  repeats  test_rows  avg_predict_time_ms  avg_per_sample_us
     logistic       20       2737             1.066370           0.389613
          knn       20       2737           100.284350          36.640245
          svm       20       2737           682.912485         249.511321
random_forest       20       2737           153.096805          55.935990
      xgboost       20       2737           145.859880          53.291882
```

### 8.4 Robustness Analysis

Robustness follows the assignment requirement: Gaussian Noise and Mask Noise are added to the training data, each noisy model is retrained, and accuracy drop is measured on the clean test set.

```text
   model noise_type  noise_level  clean_accuracy  noisy_accuracy  accuracy_drop             robustness_note                       protocol
logistic      clean         0.00        0.926562        0.926562       0.000000                  干净训练集基准结果。 retrain_on_noisy_training_data
logistic   gaussian         0.01        0.926562        0.925100       0.001461      准确率下降很小，说明模型对该噪声条件较稳定。 retrain_on_noisy_training_data
logistic       mask         0.01        0.926562        0.924004       0.002558      准确率下降很小，说明模型对该噪声条件较稳定。 retrain_on_noisy_training_data
logistic   gaussian         0.05        0.926562        0.925466       0.001096      准确率下降很小，说明模型对该噪声条件较稳定。 retrain_on_noisy_training_data
logistic       mask         0.05        0.926562        0.919985       0.006577     噪声带来一定精度下降，但整体仍保持可接受表现。 retrain_on_noisy_training_data
logistic   gaussian         0.10        0.926562        0.925100       0.001461      准确率下降很小，说明模型对该噪声条件较稳定。 retrain_on_noisy_training_data
logistic       mask         0.10        0.926562        0.918889       0.007673     噪声带来一定精度下降，但整体仍保持可接受表现。 retrain_on_noisy_training_data
logistic   gaussian         0.20        0.926562        0.923274       0.003288      准确率下降很小，说明模型对该噪声条件较稳定。 retrain_on_noisy_training_data
logistic       mask         0.20        0.926562        0.905371       0.021191  强噪声下准确率下降较明显，说明模型对该类扰动较敏感。 retrain_on_noisy_training_data
     knn      clean         0.00        0.920351        0.920351       0.000000                  干净训练集基准结果。 retrain_on_noisy_training_data
     knn   gaussian         0.01        0.920351        0.920716      -0.000365 低强度噪声可能起到数据增强作用，模型泛化能力略有提升。 retrain_on_noisy_training_data
     knn       mask         0.01        0.920351        0.924370      -0.004019 低强度噪声可能起到数据增强作用，模型泛化能力略有提升。 retrain_on_noisy_training_data
     knn   gaussian         0.05        0.920351        0.921812      -0.001461 低强度噪声可能起到数据增强作用，模型泛化能力略有提升。 retrain_on_noisy_training_data
     knn       mask         0.05        0.920351        0.919620       0.000731      准确率下降很小，说明模型对该噪声条件较稳定。 retrain_on_noisy_training_data
     knn   gaussian         0.10        0.920351        0.923274      -0.002923 低强度噪声可能起到数据增强作用，模型泛化能力略有提升。 retrain_on_noisy_training_data
```

从平均准确率下降看，random_forest 的下降最小，说明整体鲁棒性最好；logistic 的平均下降最大，对训练数据噪声更敏感。低强度噪声下个别模型准确率略有上升，这通常可以理解为噪声起到轻微数据增强作用，使模型泛化能力略有提升；当噪声强度增大时，特征信息被扰动或遮挡更多，模型准确率整体更容易下降。

### 8.5 Confusion Matrix and Classification Report Analysis

以测试准确率最高的 svm 为例，F1 表现较好的类别包括 BOMBAY, HOROZ，相对更容易出错的类别包括 SIRA, DERMASON。混淆矩阵中较明显的混淆关系为：SIRA 容易被分到 DERMASON（46 个样本）；DERMASON 容易被分到 SIRA（37 个样本）；BARBUNYA 容易被分到 CALI（14 个样本）。这些混淆是合理的，因为 SIRA、DERMASON、CALI、BARBUNYA 等类别在部分面积、周长、轴长和紧致度特征上比较接近，模型容易把形态学特征相似的豆类分到相邻类别。

## 9. Engineering and Reproducibility

All experiment stages are called through `main.py`, and the Streamlit UI is only used for result display and CSV prediction. The preprocessing artifacts are fitted only on the training set and reused for validation, test, robustness evaluation, speed testing, and uploaded CSV prediction to avoid data leakage.

## 10. Course Summary

通过这门课和这个项目，我把课堂里学到的分类算法真正串成了一个完整流程。以前更关注模型怎么写、准确率是多少，现在更能理解数据清洗、特征工程、训练/测试隔离、过拟合分析、鲁棒性和工程复现同样重要。这个项目也让我练习了把实验结果整理成报告和网页展示的过程。

我觉得课程整体比较实用，能帮助我建立机器学习项目的基本思路。建议后续课程可以多加入一些 dirty data 和部署展示的案例，因为真实项目里数据往往并不干净，只会训练模型还不够，还要能解释结果、复现实验并展示给别人看。
