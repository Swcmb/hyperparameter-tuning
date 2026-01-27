# AutoDL贝叶斯超参数优化系统

## 概述

AutoDL是一个基于贝叶斯优化的超参数调优系统，专为LDA/MDA/LMI三种深度学习任务设计。系统使用高斯过程作为代理模型，支持多种采集函数，并提供完整的优化流程管理、状态保存恢复、结果分析和可视化功能。

## 主要特性

- 🎯 **智能优化**: 基于贝叶斯优化的高效参数搜索
- 🔄 **多目标支持**: 同时优化多个性能指标（AUROC、AUPRC、F1等）
- 💾 **状态管理**: 自动保存和恢复优化状态，支持中断续传
- 📊 **结果分析**: 自动生成详细的分析报告和可视化图表
- ⚙️ **灵活配置**: 支持命令行参数和配置文件两种配置方式
- 🔍 **实时监控**: 提供详细的日志记录和进度监控

## 快速开始

### 1. 基本使用

```bash
# 最简单的使用方式
python autodl.py --task_type LDA --max_iterations 30

# 使用配置文件
python autodl.py --config config_example.json
```

### 2. 运行快速示例

```bash
# 运行交互式快速开始指南
python quick_start_example.py
```

### 3. 多目标优化

```bash
# 同时优化AUROC、AUPRC和F1分数
python autodl.py \
  --objectives AUROC AUPRC F1 \
  --objective_weights 0.5 0.3 0.2 \
  --max_iterations 50
```

## 详细使用说明

### 命令行参数

#### 基本参数
- `--config`: 配置文件路径（JSON格式）
- `--task_type`: 任务类型（LDA/MDA/LMI）
- `--max_iterations`: 最大迭代次数
- `--max_time_hours`: 最大运行时间（小时）
- `--random_seed`: 随机种子

#### 优化参数
- `--acquisition_function`: 采集函数类型（EI/PI/UCB）
- `--acquisition_params`: 采集函数参数（JSON格式）
- `--objectives`: 目标函数列表（多目标优化）
- `--objective_weights`: 目标函数权重

#### 数据参数
- `--data_path`: 数据路径
- `--cv_folds`: 交叉验证折数

#### 状态管理
- `--checkpoint_dir`: 检查点目录
- `--save_frequency`: 保存频率（每N次迭代）
- `--resume`: 恢复之前的优化
- `--checkpoint_name`: 要恢复的检查点名称

#### 输出参数
- `--output_dir`: 输出目录
- `--log_dir`: 日志目录
- `--no_report`: 不生成报告
- `--no_html`: 不生成HTML报告
- `--no_charts`: 不生成图表

### 配置文件格式

创建JSON配置文件来管理复杂的配置：

```json
{
  "task_type": "LDA",
  "max_iterations": 50,
  "max_time_hours": 24,
  "random_seed": 42,
  
  "acquisition_function": "EI",
  "acquisition_params": {
    "xi": 0.01
  },
  
  "objectives": ["AUROC", "AUPRC", "F1"],
  "objective_weights": {
    "AUROC": 0.5,
    "AUPRC": 0.3,
    "F1": 0.2
  },
  
  "cv_folds": 5,
  "checkpoint_dir": "checkpoints",
  "save_frequency": 1,
  "output_dir": "results",
  "generate_report": true
}
```

## 使用示例

### 示例1：基本单目标优化

```bash
python autodl.py \
  --task_type LDA \
  --max_iterations 30 \
  --acquisition_function EI \
  --output_dir results_lda
```

### 示例2：多目标优化

```bash
python autodl.py \
  --task_type MDA \
  --objectives AUROC AUPRC F1 \
  --objective_weights 0.4 0.4 0.2 \
  --max_iterations 50 \
  --max_time_hours 12
```

### 示例3：使用UCB采集函数

```bash
python autodl.py \
  --task_type LMI \
  --acquisition_function UCB \
  --acquisition_params '{"beta": 2.0}' \
  --max_iterations 40
```

### 示例4：恢复中断的优化

```bash
# 首先运行优化
python autodl.py --task_type LDA --max_iterations 100

# 如果中断，可以恢复
python autodl.py --resume --checkpoint_name latest
```

### 示例5：长时间运行

```bash
python autodl.py \
  --config long_run_config.json \
  --max_iterations 200 \
  --max_time_hours 48 \
  --save_frequency 5
```

## 输出文件说明

### 日志文件
- `logs/autodl_YYYYMMDD_HHMMSS.log`: 详细的运行日志

### 检查点文件
- `checkpoints/iteration_N.json`: 每次迭代的状态保存
- `checkpoints/latest.json`: 最新状态

### 结果文件
- `results/optimization_report_YYYYMMDD_HHMMSS.json`: JSON格式的详细报告
- `results/optimization_report_YYYYMMDD_HHMMSS.html`: HTML格式的可视化报告
- `results/charts_YYYYMMDD_HHMMSS/`: 包含各种可视化图表的目录
  - `convergence.png`: 收敛曲线
  - `parameter_dist.png`: 参数分布图
  - `parameter_corr.png`: 参数相关性热力图
  - `pareto_front.png`: 帕累托前沿图（多目标优化）

## 高级功能

### 多目标优化

系统支持同时优化多个目标函数：

```bash
python autodl.py \
  --objectives AUROC AUPRC F1 Precision Recall \
  --objective_weights 0.3 0.3 0.2 0.1 0.1
```

多目标优化会：
- 维护帕累托前沿
- 生成帕累托最优解集合
- 提供目标权重的加权求和
- 生成多目标可视化图表

### 采集函数选择

支持三种采集函数：

1. **Expected Improvement (EI)**: 平衡探索和利用
2. **Probability of Improvement (PI)**: 保守的改进策略
3. **Upper Confidence Bound (UCB)**: 乐观的探索策略

```bash
# EI采集函数
python autodl.py --acquisition_function EI --acquisition_params '{"xi": 0.01}'

# UCB采集函数
python autodl.py --acquisition_function UCB --acquisition_params '{"beta": 2.0}'

# PI采集函数
python autodl.py --acquisition_function PI --acquisition_params '{"xi": 0.01}'
```

### 状态管理

系统自动保存优化状态，支持：
- 自动检查点保存
- 中断后恢复
- 状态验证和修复
- 历史状态查看

```bash
# 设置保存频率
python autodl.py --save_frequency 5  # 每5次迭代保存一次

# 从特定检查点恢复
python autodl.py --resume --checkpoint_name iteration_20
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少`cv_folds`数量
   - 降低`max_iterations`
   - 使用更小的批大小

2. **优化收敛慢**
   - 尝试不同的采集函数
   - 调整采集函数参数
   - 增加初始随机采样数量

3. **参数约束违反**
   - 检查参数空间定义
   - 使用参数修复功能
   - 查看详细的验证错误信息

4. **状态恢复失败**
   - 检查检查点文件完整性
   - 使用`--checkpoint_name`指定特定检查点
   - 清理损坏的检查点文件

### 调试模式

启用详细日志记录：

```bash
# 设置环境变量启用调试模式
export AUTODL_DEBUG=1
python autodl.py --task_type LDA --max_iterations 10
```

## 性能优化建议

1. **参数空间设计**
   - 合理设置参数范围
   - 使用对数尺度处理大范围参数
   - 添加合适的参数约束

2. **采集函数调优**
   - EI适合大多数情况
   - UCB适合需要更多探索的场景
   - PI适合保守的优化策略

3. **计算资源管理**
   - 合理设置交叉验证折数
   - 使用检查点避免重复计算
   - 监控内存和CPU使用情况

## 扩展开发

系统采用模块化设计，支持扩展：

- 添加新的采集函数
- 支持新的任务类型
- 自定义参数空间
- 扩展结果分析功能

详细的开发文档请参考各个模块的README文件。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。