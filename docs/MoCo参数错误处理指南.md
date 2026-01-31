# MoCo参数错误处理指南

本文档提供MoCo（Momentum Contrast）参数相关错误的诊断和解决方案。

## 目录

1. [参数约束错误](#参数约束错误)
2. [参数解析错误](#参数解析错误)
3. [配置转换错误](#配置转换错误)
4. [优化过程错误](#优化过程错误)
5. [性能相关问题](#性能相关问题)
6. [兼容性问题](#兼容性问题)

## 参数约束错误

### 错误1: MoCo温度约束违反

**错误信息:**
```
MoCoConstraintViolationError: DoubleTau MoCo温度约束违反: tau2 (0.15) < tau1 (0.25)
```

**原因:** DoubleTau模式下，负样本温度系数(tau2)必须大于等于正样本温度系数(tau1)。

**解决方案:**
1. **手动调整参数:**
   ```python
   # 错误的参数设置
   params = {
       'moco_tau1': 0.25,
       'moco_tau2': 0.15  # 错误: tau2 < tau1
   }
   
   # 正确的参数设置
   params = {
       'moco_tau1': 0.15,
       'moco_tau2': 0.25  # 正确: tau2 >= tau1
   }
   ```

2. **使用自动修复:**
   ```python
   from autodl_core import create_default_parameter_space
   
   parameter_space = create_default_parameter_space()
   fixed_params = parameter_space.suggest_parameter_fix(invalid_params)
   ```

3. **配置文件设置:**
   ```json
   {
     "moco_tau1": 0.15,
     "moco_tau2": 0.25,
     "constraint_check": true
   }
   ```

### 错误2: MoCo动量系数超出范围

**错误信息:**
```
MoCoConstraintViolationError: MoCo动量系数超出范围: 0.85 不在 [0.9, 0.9999] 内
```

**原因:** MoCo动量系数必须在0.9到0.9999之间，以确保特征表示的稳定性。

**解决方案:**
1. **调整动量系数:**
   ```python
   # 推荐的动量系数范围
   recommended_momentum = {
       'small_dataset': 0.995,
       'medium_dataset': 0.999,
       'large_dataset': 0.9999
   }
   ```

2. **命令行参数:**
   ```bash
   python autodl.py --moco_momentum 0.999
   ```

### 错误3: 温度参数为负值或零

**错误信息:**
```
MoCoConstraintViolationError: MoCo温度参数必须为正值: moco_t = -0.1
```

**原因:** 所有MoCo温度参数(moco_t, moco_tau1, moco_tau2)必须为正值。

**解决方案:**
1. **检查参数设置:**
   ```python
   # 确保所有温度参数为正值
   temp_params = {
       'moco_t': 0.2,      # > 0
       'moco_tau1': 0.15,  # > 0
       'moco_tau2': 0.25   # > 0
   }
   ```

## 参数解析错误

### 错误4: 参数类型转换失败

**错误信息:**
```
ParameterParsingError: 无法将 'invalid_value' 转换为布尔类型 (enable_view_0)
```

**原因:** enable_view_0参数只接受'true'或'false'字符串值。

**解决方案:**
1. **正确的参数值:**
   ```python
   # 正确的布尔参数设置
   params = {
       'enable_view_0': 'true'   # 或 'false'
   }
   ```

2. **命令行使用:**
   ```bash
   python autodl.py --enable_view_0 true
   ```

### 错误5: 未知MoCo参数

**错误信息:**
```
ParameterError: 未知的MoCo参数: moco_invalid_param
```

**原因:** 使用了不存在的MoCo参数名称。

**解决方案:**
1. **检查支持的参数列表:**
   ```python
   supported_moco_params = [
       'moco_momentum',  # 动量系数
       'moco_t',         # 基础温度
       'moco_tau1',      # 正样本温度
       'moco_tau2',      # 负样本温度
       'moco_K',         # 队列大小
       'moco_type',      # MoCo类型
       'enable_view_0'   # 视图控制
   ]
   ```

## 配置转换错误

### 错误6: 配置映射失败

**错误信息:**
```
ConfigurationError: MoCo参数映射失败: enable_view_0 无法转换为布尔值
```

**原因:** 配置转换过程中，字符串到布尔值的转换失败。

**解决方案:**
1. **检查配置转换器:**
   ```python
   # 正确的转换逻辑
   if 'enable_view_0' in parameters:
       config['enable_view_0'] = str(parameters['enable_view_0']).lower() == 'true'
   ```

2. **使用标准化的配置格式:**
   ```json
   {
     "enable_view_0": "true",
     "moco_type": "double_tau"
   }
   ```

## 优化过程错误

### 错误7: 特征维度不匹配

**错误信息:**
```
OptimizationError: 特征维度不匹配: 期望23，实际14
```

**原因:** 新增MoCo参数后，特征向量维度发生变化，但模型期望的维度未更新。

**解决方案:**
1. **重新初始化优化器:**
   ```python
   optimizer = create_bayesian_optimizer(
       task_type="LDA",
       acquisition_function_type="EI",
       n_initial_points=5,
       random_state=42
   )
   optimizer._initialize_optimization()  # 重新初始化
   ```

2. **清除旧的检查点:**
   ```bash
   rm -rf checkpoints/*  # 删除旧的检查点文件
   ```

### 错误8: 采集函数优化失败

**错误信息:**
```
AcquisitionOptimizationError: 采集函数优化失败，所有重启都失败了
```

**原因:** MoCo参数的约束条件导致采集函数优化困难。

**解决方案:**
1. **调整采集函数参数:**
   ```python
   acquisition_params = {
       'beta': 1.0,  # 降低探索程度
       'xi': 0.01    # 调整改进阈值
   }
   ```

2. **使用回退策略:**
   ```python
   # 系统会自动使用随机采样作为回退
   # 可以通过日志查看回退使用情况
   ```

## 性能相关问题

### 问题1: 优化收敛缓慢

**症状:** 优化过程中性能改进很小或停滞。

**可能原因:**
- 温度参数设置过高
- 动量系数不合适
- 队列大小与数据集不匹配

**解决方案:**
1. **调整温度参数:**
   ```python
   # 降低温度参数以增加学习难度
   params = {
       'moco_t': 0.1,        # 从0.2降低到0.1
       'moco_tau1': 0.1,     # 从0.15降低到0.1
       'moco_tau2': 0.2      # 从0.25降低到0.2
   }
   ```

2. **优化动量设置:**
   ```python
   # 根据数据集大小调整动量
   momentum_by_dataset_size = {
       'small': 0.995,
       'medium': 0.999,
       'large': 0.9999
   }
   ```

### 问题2: 内存使用过高

**症状:** 训练过程中出现内存不足错误。

**解决方案:**
1. **减小队列大小:**
   ```python
   params = {
       'moco_K': 2048  # 从4096或8192减小到2048
   }
   ```

2. **禁用某些视图:**
   ```python
   params = {
       'enable_view_0': 'false'  # 禁用第0视图
   }
   ```

## 兼容性问题

### 问题3: 向后兼容性错误

**错误信息:**
```
CompatibilityError: 历史数据缺少新MoCo参数: moco_tau1, moco_tau2
```

**原因:** 加载旧版本的优化历史时，缺少新增的MoCo参数。

**解决方案:**
1. **自动填充默认值:**
   ```python
   # 系统会自动为缺失的参数添加默认值
   default_values = {
       'moco_tau1': 0.2,
       'moco_tau2': 0.3,
       'enable_view_0': 'true'
   }
   ```

2. **手动更新历史数据:**
   ```python
   from autodl_core import OptimizationHistory
   
   # 加载并更新历史数据
   history = OptimizationHistory.from_dict(old_data)
   # 系统会自动处理缺失参数
   ```

## 调试技巧

### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 参数验证检查

```python
from parameter_validator import ParameterValidator
from autodl_core import create_default_parameter_space

parameter_space = create_default_parameter_space()
validator = ParameterValidator(parameter_space)

# 详细验证
is_valid, errors = validator.validate_parameters(params, strict=True)
if not is_valid:
    print(f"参数验证失败: {errors}")
```

### 3. 约束函数测试

```python
# 测试特定约束
tau_constraint = validator.constraint_functions['moco_tau_ordering']
momentum_constraint = validator.constraint_functions['moco_momentum_range']
temp_constraint = validator.constraint_functions['moco_temperature_positive']

print(f"Tau约束: {tau_constraint(params)}")
print(f"动量约束: {momentum_constraint(params)}")
print(f"温度约束: {temp_constraint(params)}")
```

### 4. 参数修复测试

```python
# 测试参数修复功能
fixed_params = parameter_space.suggest_parameter_fix(invalid_params)
print(f"修复前: {invalid_params}")
print(f"修复后: {fixed_params}")
```

## 常见问题FAQ

**Q: 为什么tau2必须大于等于tau1？**
A: 在DoubleTau MoCo中，tau1用于正样本，tau2用于负样本。较大的tau2使负样本学习更容易，有助于模型稳定性。

**Q: 动量系数应该设置多大？**
A: 建议范围0.995-0.9999。大数据集使用更高的动量(0.9999)，小数据集使用较低的动量(0.995)。

**Q: 什么时候使用DoubleTau模式？**
A: 当基础MoCo模式性能不佳时，可以尝试DoubleTau模式进行精细调优。

**Q: enable_view_0参数的作用是什么？**
A: 控制是否启用MoCo的第0视图。启用可以增加数据多样性，但会增加计算开销。

**Q: 如何选择合适的队列大小？**
A: 一般建议4096或8192。大数据集使用更大的队列，内存受限时使用较小的队列。

## 联系支持

如果遇到本文档未涵盖的问题，请：

1. 检查日志文件中的详细错误信息
2. 验证参数配置是否符合约束条件
3. 尝试使用默认配置进行测试
4. 查看示例代码和配置文件

更多技术支持，请参考项目文档或提交问题报告。