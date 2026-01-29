# 贝叶斯超参数优化系统 - 核心数据结构

本模块实现了贝叶斯超参数优化系统的核心数据结构和配置管理功能。

## 核心组件

### 1. 数据类 (autodl_core.py)

#### ParameterConfig
- 定义单个参数的类型、范围和约束
- 支持连续型、离散型、分类型参数
- 提供参数验证和随机采样功能

#### OptimizationResult
- 记录单次参数评估的完整信息
- 包含参数组合、性能指标、评估时间等
- 支持结果比较和序列化

#### OptimizationHistory
- 管理整个优化过程的历史记录
- 提供收敛分析和参数历史查询
- 自动维护最佳结果

#### ParameterSpace
- 管理所有可优化参数的定义
- 提供参数验证和随机采样
- 支持不同类型参数的统一管理

### 2. 参数验证器 (parameter_validator.py)

#### ParameterValidator
- 智能的参数约束检查
- 自动参数修复建议
- 详细的约束违反报告

#### ConfigurationConverter
- 将优化参数转换为实验配置
- 支持不同任务类型的配置
- 确保类型正确性和完整性

## 主要特性

### 🔧 灵活的参数定义
```python
space = ParameterSpace()
space.add_continuous_parameter('lr', 1e-5, 1e-2, log_scale=True)
space.add_discrete_parameter('batch_size', [16, 32, 64])
space.add_categorical_parameter('optimizer', ['adam', 'sgd'])
```

### ✅ 智能约束检查
- 隐藏层维度递减约束
- 注意力头数整除约束
- 学习率和权重衰减合理性约束
- 损失权重正值约束
- 批大小和MoCo队列比例约束

### 🔄 自动参数修复
```python
validator = ParameterValidator(space)
fixed_params = validator.suggest_parameter_fix(broken_params)
```

### 📊 完整的历史管理
```python
history = OptimizationHistory()
history.add_result(result)
convergence = history.get_convergence_curve()
best_params = history.get_best_parameters()
```

### 💾 数据持久化
```python
# 序列化
data = history.to_dict()

# 反序列化
restored_history = OptimizationHistory.from_dict(data)
```

## 使用示例

### 基本使用流程

```python
from autodl_core import create_default_parameter_space, OptimizationHistory
from parameter_validator import ParameterValidator, ConfigurationConverter

# 1. 创建参数空间
space = create_default_parameter_space("LDA")

# 2. 创建验证器
validator = ParameterValidator(space)

# 3. 采样和验证参数
params = space.sample_random_parameters(seed=42)
is_valid, errors = validator.validate_parameters(params)

# 4. 转换为实验配置
converter = ConfigurationConverter("LDA")
exp_config = converter.convert_to_experiment_config(params)

# 5. 记录优化历史
history = OptimizationHistory()
result = OptimizationResult(
    parameters=params,
    objective_value=0.85,
    metrics={'AUROC': 0.85, 'AUPRC': 0.78},
    iteration=1,
    timestamp=datetime.now(),
    evaluation_time=120.0
)
history.add_result(result)
```

## 测试

运行测试套件：
```bash
python test_core_structures.py  # 完整测试
python test_parameter_fix.py    # 参数修复测试
python example_usage.py         # 使用示例
```

## 支持的参数类型

### 连续型参数
- 学习率 (lr): 1e-5 到 1e-2，对数尺度
- 模型维度 (dimensions, hidden1, hidden2, decoder1)
- 正则化参数 (dropout, weight_decay)
- 损失权重 (alpha, beta, gamma)

### 离散型参数
- 注意力头数 (gat_heads, gt_heads, fusion_heads): [2, 4, 8, 16]
- 批大小 (batch): [16, 25, 32, 64]
- MoCo队列大小 (moco_K): [1024, 2048, 4096, 8192]

### 分类型参数
- 融合策略 (fusion_strategy): ['self_attention', 'co_attention', 'hybrid', 'transformer_multihead']
- 特征类型 (feature_type): ['normal', 'uniform', 'one_hot']
- MoCo类型 (moco_type): ['basic', 'double_tau']

## 约束规则

1. **隐藏层递减**: dimensions ≥ hidden1 ≥ hidden2
2. **解码器约束**: decoder1 ≥ hidden2
3. **注意力头数**: 隐藏维度必须能被头数整除
4. **学习率合理性**: weight_decay ≤ lr × 10
5. **损失权重**: 至少一个权重 > 0
6. **批队列比例**: moco_K ≥ batch × 4

## 文件结构

```
hyperparameter-tuning/
├── autodl_core.py              # 核心数据结构
├── parameter_validator.py      # 参数验证和转换
├── test_core_structures.py     # 完整测试套件
├── test_parameter_fix.py       # 参数修复测试
├── example_usage.py            # 使用示例
└── README_core.md              # 本文档
```

这些核心数据结构为后续的贝叶斯优化器、高斯过程模型、采集函数等组件提供了坚实的基础。