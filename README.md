# 使用SageMaker+XGBoost，将时间序列转换为监督学习，完成预测性维护的实践

**关键字**：SageMaker；XGBoost；Python；滑窗；滑动窗口方法；时间序列预测转化为监督学习问题；将多元时间序列数据转换为监督学习问题；如何用Python将时间序列问题转化为有监督学习问题；时间序列预测的机器学习模型；

[TOC]

## 一、前言

​	《预测性维护》是传统制造业常见AI场景。过去多年，制造业一直在努力提高运营效率，并避免由于组件故障而导致停机。通常使用的方法是：

1. 通常采用的方法是使用“物理传感器（标签）”做数据连接，存储和大屏上进行了大量重复投资，以监视设备状况并获得实时警报。
2. 主要的数据分析方法是单变量阈值和基于物理的建模方法，尽管这些方法在检测特定故障类型和操作条件方面很有效，但它们通常会错过通过推导每台设备的多元关系而检测到的重要信息。
3. 借助机器学习，可以提供从设备的历史数据中学习的数据驱动模型。主要挑战在于，ML的项目投资和工程师培训，实施这样的机器学习解决方案既耗时又昂贵。



**AWS Sagemaker提供了一个简单有效的解决方案，就是使用Sagemaker+XGboost完成检测到异常的设备行为，实现《预测性维护》的场景需求。**

1. 使用了**“滑窗”**方法进行数据集的重构，并配合XGBoost算法，**将多元时间序列数据集转换为监督学习问题（复杂问题转换为简单问题）；**
2. 使用Sagemaker Studio各项功能（自动机器学习Autopilot、自动化的调参 Hyperparameter tuning jobs、多模型终端节点multi-model endpoints等）**加速XGBoost超参数优化的速度，有效提高模型准确度**；
3. 使用Sagemaker Studio **完成数据预处理与特征工程：**1）探索相关性；2）缩小特征值范围；3）将海量数据分为几批进行预处理，以避免服务器内存溢出；4）数据清理，滑动窗口清除无效数据；5）过滤数据，解决正负样本不平衡的问题；
4. 使用Sagemaker+XGboost训练了6个预测模型，分别覆盖提前5、10、20、30、40、50分钟，演示实验结果。

```mermaid
stateDiagram-v2
    [*] --> ETL数据标注
    ETL数据标注 --> 数据处理特征工程      
    数据处理特征工程 --> 算法选择构建模型
    算法选择构建模型 --> 模型训练
    模型训练 --> 评估Evaluation
    评估Evaluation -->超参数优化(反复调参试错)
    超参数优化(反复调参试错) --> 模型训练
    评估Evaluation --> 部署型线上推理
    部署型线上推理 --> 持续监控数据收集    
    持续监控数据收集 --> ETL数据标注
    持续监控数据收集 --> [*]
```

<img src="https://raw.githubusercontent.com/liangyimingcom/storage/master/uPic/image-20210327221754657.png" alt="image-20210327221754657" style="zoom: 40%;" />



## 二、需求分析与预测结果





## 三、数据处理与特征工程

ETL数据标注
数据处理
特征工程



## 四、SageMaker+XGBoost 训练与朝参数调优

算法选择构建模型
模型训练
超参数优化(反复调参试错)
评估Evaluation



## 五、模型部署与使用

部署模型线上推理
持续监控数据收集

## 六、结论



## 七、引用

引用reference：

1. 机器学习中梯度提升算法的简要介绍 https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
2. 时间序列预测转化为监督学习问题 https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
3. 如何用Python将时间序列问题转化为有监督学习问题 https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
4. How To Backtest Machine Learning Models for Time Series Forecasting如何回测时间序列预测的机器学习模型 https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
5. How to Use XGBoost for Time Series Forecasting https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

