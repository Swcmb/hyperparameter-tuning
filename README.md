# Hyperparameter Tuning for Bio-Association Prediction

一个用于生物关联预测任务的深度学习超参数优化系统，支持LDA（lncRNA-Disease Association）、MDA（miRNA-Disease Association）、LMI（lncRNA-miRNA Interaction）三种任务的自动超参数调优。

## 项目概述

本项目包含两个核心系统：

1. **AutoDL 贝叶斯超参数优化系统** - 基于高斯过程的智能超参数调优
2. **深度学习训练系统** - 支持图神经网络、MoCo对比学习的训练框架

## 核心特性

### AutoDL优化系统
- **智能优化算法**: 基于高斯过程的贝叶斯优化
- **多目标支持**: 同时优化AUROC、AUPRC、F1等多个指标
- **状态管理**: 支持中断续传和检查点保存
- **采集函数**: 支持EI、PI、UCB三种采集函数
- **可视化分析**: 自动生成详细的分析报告和图表

### 训练系统
- **图神经网络**: 支持GAT、Transformer等多种图神经网络结构
- **对比学习**: 完整支持MoCo（Momentum Contrast）和BYOL
- **注意力机制**: 支持自注意力、协作注意力、混合注意力
- **数据增强**: 支持多种数据增强策略
- **对抗训练**: 支持PGD对抗训练
- **5折交叉验证**: 标准的交叉验证评估流程

## 快速开始

### 环境要求

```bash
Python 3.7+
PyTorch 1.10+
PyTorch Geometric
scikit-learn
numpy
matplotlib
seaborn
```

### 基础训练

```bash
# 使用默认参数进行5折交叉验证训练
python main.py

# 指定任务类型
python main.py --task_type LDA
```

### AutoDL超参数优化

```bash
# 运行LDA任务的超参数优化（30次迭代）
python autodl.py --task_type LDA --max_iterations 30

# 使用配置文件
python autodl.py --config config.json

# 多目标优化
python autodl.py --task_type LDA --objectives AUROC AUPRC F1 --objective_weights 0.5 0.3 0.2
```

## 项目结构

```
hyperparameter-tuning/
├── autodl.py                 # AutoDL优化系统主入口
├── autodl_core.py            # AutoDL核心数据结构
├── bayesian_optimizer.py     # 贝叶斯优化器
├── gaussian_process.py       # 高斯过程模型
├── acquisition_function.py   # 采集函数(EI/PI/UCB)
├── task_evaluator.py         # 任务评估器
├── state_manager.py          # 状态管理器
├── result_analyzer.py        # 结果分析器
├── report_generator.py       # 报告生成器
├── parameter_manager.py      # 参数管理器
├── parameter_validator.py    # 参数验证器
│
├── main.py                   # 训练系统主入口
├── train.py                  # 训练流程
├── layer.py                  # 神经网络层定义
├── instantiation.py          # 模型实例化
├── data_preprocess.py        # 数据预处理
├── utils.py                  # 工具函数
├── parms_setting.py          # 参数设置
├── visualization.py          # 可视化工具
│
├── dataset1/                 # 数据集1
│   ├── LDA.edgelist
│   ├── MDA.edgelist
│   └── LMI.edgelist
├── dataset2/                 # 数据集2
│
├── logs/                     # 日志文件
├── results/                  # 结果输出
├── checkpoints/              # 检查点
│
├── docs/                     # 文档
├── examples/                 # 示例代码
└── tests/                    # 测试文件
```

## 支持的任务类型

| 任务类型 | 说明 | 输入文件 |
|---------|------|----------|
| **LDA** | lncRNA-Disease Association预测 | LDA.edgelist |
| **MDA** | miRNA-Disease Association预测 | MDA.edgelist |
| **LMI** | lncRNA-miRNA Interaction预测 | LMI.edgelist |

## 主要参数说明

### 训练参数
- `--task_type`: 任务类型 (LDA/MDA/LMI)
- `--epochs`: 训练轮数 (默认: 50)
- `--lr`: 学习率
- `--batch`: 批大小
- `--hidden1`, `--hidden2`: 隐藏层维度
- `--gat_heads`, `gt_heads`: 注意力头数

### MoCo参数
- `--moco_momentum`: 动量系数 (0.9-0.9999)
- `--moco_t`: 基础温度 (0.01-1.0)
- `--moco_tau1`, `--moco_tau2`: DoubleTau温度参数
- `--moco_K`: 队列大小 (1024/2048/4096/8192)
- `--moco_type`: MoCo类型 (basic/double_tau)

### 损失权重
- `--alpha`: 监督损失权重
- `--beta`: 对比学习损失权重
- `--gamma`: 节点对抗损失权重

### 数据增强
- `--augment`: 增强方式
- `--augment_mode`: 增强模式 (static/online)
- `--noise_std`: 噪声标准差
- `--mask_rate`: 掩码率

### 对抗训练
- `--adv_mode`: 对抗模式 (none/mgraph)
- `--adv_eps`: 扰动幅度
- `--adv_steps`: PGD步数
- `--adv_alpha`: PGD步长

## 输出结果

### 训练输出
```
OUTPUT/result/
├── metrics/
│   ├── train_epoch_metrics_fold_1_*.csv
│   ├── y_true_pred_fold_1_*.csv
│   ├── threshold_scan_fold_1_*.txt
│   ├── temperature_fold_1_*.json
│   └── adv_config_fold_1_*.json
├── figure/
│   ├── loss_breakdown_fold_1.png
│   ├── epoch_curves_fold_1.png
│   ├── roc_fold_1.png
│   ├── pr_fold_1.png
│   └── confusion_matrix_sum.png
└── result_summary_*.txt
```

### AutoDL输出
```
results/
├── optimization_report_*.json
├── optimization_report_*.html
└── charts_*/
    ├── convergence.png
    ├── parameter_dist.png
    ├── parameter_corr.png
    └── pareto_front.png
```

## 配置文件示例

### AutoDL配置文件 (config.json)
```json
{
  "task_type": "LDA",
  "max_iterations": 50,
  "acquisition_function": "EI",
  "objectives": ["AUROC", "AUPRC", "F1"],
  "objective_weights": {
    "AUROC": 0.5,
    "AUPRC": 0.3,
    "F1": 0.2
  },
  "cv_folds": 5,
  "data_path": "dataset1",
  "output_dir": "results",
  "generate_report": true,
  "generate_charts": true
}
```

## 文档

详细文档请查看 `docs/` 目录：

- `docs/05-项目文档/AutoDL使用说明.md` - AutoDL系统使用指南
- `docs/05-项目文档/API文档.md` - 完整API参考
- `docs/01-核心组件说明/` - 各组件详细说明
- `docs/03-MoCo参数相关/` - MoCo参数相关文档

## 示例代码

查看 `examples/` 目录获取更多示例：

- `examples/core_examples/quick_start_example.py` - 快速开始
- `examples/application_scenarios/moco_parameter_example.py` - MoCo参数示例
- `examples/application_scenarios/multi_objective_example.py` - 多目标优化示例
- `examples/analysis_tools/visualizer_example.py` - 可视化示例

## 常见问题

### Q: 如何设置GPU？
A: 通过环境变量 `CUDA_VISIBLE_DEVICES` 指定GPU，例如：
```bash
export CUDA_VISIBLE_DEVICES=0
```

### Q: 如何恢复中断的优化？
A: 使用 `--resume` 参数：
```bash
python autodl.py --resume
```

### Q: 如何调整MoCo参数？
A: MoCo参数需要通过配置文件设置，参见上面的配置文件示例。

### Q: 训练时出现CUDA OOM错误？
A: 减小批大小 `--batch` 或隐藏层维度 `--hidden1`, `--hidden2`

## 引用

如果本项目对您的研究有帮助，请引用相关论文。

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提交Issue。