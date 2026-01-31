# 参数空间管理器 (ParameterSpace)

## 概述

参数空间管理器是贝叶斯超参数优化系统的核心组件，负责管理所有可优化参数的定义、验证和采样。

## 主要功能

### 1. 参数类型支持
- **连续型参数**: 支持数值范围和对数尺度
- **离散型参数**: 支持预定义的数值列表
- **分类型参数**: 支持字符串类别选择

### 2. 参数验证
- 基本范围检查
- 复杂约束验证（隐藏层递减、注意力头数整除等）
- 详细错误报告

### 3. 参数采样
- 随机采样（支持种子设定）
- 对数尺度采样
- 约束满足采样

### 4. 参数修复
- 自动修复超出范围的参数
- 智能约束修复
- 最小化修改原则

### 5. 序列化支持
- JSON格式保存/加载
- 完整状态恢复

## 使用示例

```python
from autodl_core import create_default_parameter_space

# 创建参数空间
space = create_default_parameter_space("LDA")

# 随机采样
params = space.sample_random_parameters(seed=42)

# 验证参数
is_valid, errors = space.validate_parameters_detailed(params)

# 修复参数
if not is_valid:
    fixed_params = space.suggest_parameter_fix(params)
```

## 支持的参数

### 连续型参数 (10个)
- `dimensions`: 主维度 (128-512)
- `hidden1`: 第一隐藏层 (64-256)
- `hidden2`: 第二隐藏层 (32-128)
- `decoder1`: 解码器维度 (256-1024)
- `lr`: 学习率 (1e-5 to 1e-2, 对数尺度)
- `dropout`: Dropout率 (0.0-0.5)
- `weight_decay`: 权重衰减 (1e-6 to 1e-2, 对数尺度)
- `alpha`, `beta`, `gamma`: 损失权重 (0.1-2.0)

### 离散型参数 (5个)
- `gat_heads`: GAT注意力头数 [2, 4, 8, 16]
- `gt_heads`: GT注意力头数 [2, 4, 8, 16]
- `fusion_heads`: 融合注意力头数 [2, 4, 8, 16]
- `batch`: 批大小 [16, 25, 32, 64]
- `moco_K`: MoCo队列大小 [1024, 2048, 4096, 8192]

### 分类型参数 (3个)
- `fusion_strategy`: 融合策略 ['self_attention', 'co_attention', 'hybrid', 'transformer_multihead']
- `feature_type`: 特征类型 ['normal', 'uniform', 'one_hot']
- `moco_type`: MoCo类型 ['basic', 'double_tau']

## 约束规则

1. **隐藏层递减**: dimensions ≥ hidden1 ≥ hidden2
2. **解码器约束**: decoder1 ≥ hidden2
3. **注意力头数**: 隐藏维度必须能被头数整除
4. **学习率约束**: weight_decay ≤ lr × 10
5. **损失权重**: 至少一个权重 > 0
6. **MoCo约束**: moco_K ≥ batch × 4

## 测试

运行测试以验证功能：

```bash
python test_parameter_space.py
python parameter_space_example.py
```

## 文件结构

- `autodl_core.py`: 核心实现
- `test_parameter_space.py`: 全面测试
- `parameter_space_example.py`: 使用示例
- `parameter_space_config.json`: 序列化配置示例