# Hyperparameter Tuning 测试套件

本目录包含 AutoDL 贝叶斯优化系统的完整测试套件，测试文件按功能模块分类组织。

## 目录结构

```
tests/
├── 01_core_components/          # 核心组件测试
├── 02_bayesian_optimization/    # 贝叶斯优化相关测试
├── 03_gaussian_process/         # 高斯过程相关测试
├── 04_integration/              # 集成和端到端测试
├── 05_multi_objective/          # 多目标优化测试
├── 06_parameter_fixing/         # 参数修复和约束测试
├── 07_logging_output/           # 日志和输出测试
├── 08_performance/              # 性能测试
├── 09_report_analysis/          # 报告和分析相关测试
├── 10_state_management/         # 状态管理测试
├── 11_training_output/          # 真实训练输出测试
├── checkpoints/                 # 检查点文件存储
├── test_benchmark_results/      # 基准测试结果
├── test_checkpoints/            # 测试检查点
├── test_logs/                   # 测试日志
├── test_optimization_results/   # 优化结果
└── test_visualizer_output/      # 可视化输出
```

## 测试分类说明

### 01_core_components - 核心组件测试

测试系统的核心数据结构和基础功能。

- **test_core_structures.py**: 测试核心数据结构和配置管理
  - ParameterConfig 参数配置类
  - OptimizationResult 优化结果类
  - OptimizationHistory 优化历史类
  - ParameterSpace 参数空间类
  - ParameterValidator 参数验证器
  - ConfigurationConverter 配置转换器

- **test_parameter_space.py**: 参数空间管理器全面测试
  - 参数空间创建和管理
  - 参数验证和约束检查
  - 随机采样
  - 参数修复功能
  - 序列化和反序列化

- **test_parameter_manager_basic.py**: 参数管理器基础测试

### 02_bayesian_optimization - 贝叶斯优化相关测试

测试贝叶斯优化器的核心功能和MoCo参数集成。

- **test_bayesian_optimizer_moco.py**: 贝叶斯优化器MoCo参数集成测试
  - 参数特征编码
  - 参数建议功能
  - MoCo参数约束验证

- **test_bayesian_optimizer_constraints.py**: 贝叶斯优化器约束测试

- **test_acquisition_function.py**: 采集函数测试
  - Expected Improvement (EI)
  - Probability of Improvement (PI)
  - Upper Confidence Bound (UCB)
  - Entropy Search (ES)
  - 采集函数优化器

- **test_acquisition_integration.py**: 采集函数集成测试

- **test_acquisition_properties.py**: 采集函数属性测试

### 03_gaussian_process - 高斯过程相关测试

测试高斯过程模型与系统的集成。

- **test_gaussian_process_integration.py**: 高斯过程模型集成测试
  - 参数空间到高斯过程输入的转换
  - 高斯过程与优化结果的集成
  - 采集函数与参数空间的集成
  - 优化历史的集成
  - 模型持久化

### 04_integration - 集成和端到端测试

测试完整系统的集成和工作流程。

- **test_integration_end_to_end.py**: 端到端集成测试
  - 单目标优化完整流程
  - 多目标优化完整流程
  - 参数空间约束处理
  - 错误处理和恢复机制
  - 组件集成测试

- **test_moco_integration_complete.py**: MoCo集成完整测试

### 05_multi_objective - 多目标优化测试

测试多目标优化功能。

- **test_multi_objective_moco.py**: 多目标优化MoCo参数支持测试
  - 多目标优化器MoCo参数处理
  - 多目标优化过程中的MoCo参数处理
  - MoCo参数对目标函数的影响
  - 帕累托前沿中的MoCo参数多样性

- **test_multi_objective_properties.py**: 多目标优化属性测试

- **test_multi_objective_properties_mock.py**: 多目标优化属性模拟测试

### 06_parameter_fixing - 参数修复和约束测试

测试参数修复和约束处理功能。

- **test_parameter_fix.py**: 参数修复测试

- **test_autodl_fix.py**: AutoDL修复测试

### 07_logging_output - 日志和输出测试

测试日志记录和输出功能。

- **test_enhanced_logging.py**: 增强日志功能测试

- **test_enhanced_task_evaluator.py**: 增强任务评估器测试

- **test_detailed_output.py**: 详细输出测试

### 08_performance - 性能测试

测试系统性能和优化。

- **test_performance_basic.py**: 基础性能测试
  - 基础日志性能
  - 标签处理性能
  - 格式化性能
  - 性能优化分析

### 09_report_analysis - 报告和分析相关测试

测试报告生成和结果分析功能。

- **test_report_generator.py**: 报告生成器单元测试
  - 基本报告生成功能
  - JSON报告生成
  - HTML报告生成
  - 自定义配置
  - 错误处理
  - 报告数据访问

- **test_report_generator_properties.py**: 报告生成器属性测试

- **test_result_analyzer.py**: 结果分析器测试
  - 统计摘要
  - 参数敏感性分析
  - 收敛性分析
  - 参数重要性排序
  - 最佳参数分析
  - 参数相关性分析
  - 报告生成和保存

### 10_state_management - 状态管理测试

测试状态保存、恢复和检查点管理功能。

- **test_state_manager.py**: 状态管理器测试
  - 状态保存和加载
  - 检查点验证
  - 按频率创建检查点
  - 列出和获取检查点
  - 清理损坏的检查点
  - 检查点信息获取
  - 检查点导出和导入
  - 最大检查点数量限制
  - 高斯过程序列化
  - 错误处理
  - 优化工作流程模拟

### 11_training_output - 真实训练输出测试

测试真实训练输出的处理。

- **test_real_training_output.py**: 真实训练输出测试

## 运行测试

### 运行所有测试

```bash
# 使用 pytest 运行所有测试
pytest

# 运行特定分类的测试
pytest 01_core_components/
pytest 02_bayesian_optimization/
pytest 04_integration/

# 运行特定测试文件
pytest 01_core_components/test_core_structures.py

# 详细输出
pytest -v

# 显示打印输出
pytest -s
```

### 运行单个测试文件

```bash
# 直接运行 Python 测试文件
python 01_core_components/test_core_structures.py
python 02_bayesian_optimization/test_acquisition_function.py
```

## 测试依赖

测试套件依赖以下主要模块：

- `autodl_core`: 核心数据结构和配置
- `bayesian_optimizer`: 贝叶斯优化器
- `acquisition_function`: 采集函数
- `gaussian_process`: 高斯过程模型
- `task_evaluator`: 任务评估器
- `state_manager`: 状态管理器
- `result_analyzer`: 结果分析器
- `visualizer`: 可视化器
- `report_generator`: 报告生成器

## 测试覆盖率

当前测试覆盖了以下主要功能模块：

- ✅ 核心数据结构 (ParameterConfig, OptimizationResult, OptimizationHistory)
- ✅ 参数空间管理 (ParameterSpace, ParameterValidator)
- ✅ 贝叶斯优化器 (单目标和多目标)
- ✅ 采集函数 (EI, PI, UCB, ES)
- ✅ 高斯过程模型
- ✅ MoCo参数集成
- ✅ 状态管理和检查点
- ✅ 报告生成
- ✅ 结果分析
- ✅ 参数修复和约束

## 测试数据

测试使用模拟数据进行，包括：

- 模拟的参数空间采样
- 模拟的目标函数评估
- 模拟的任务评估器
- 模拟的优化历史

部分集成测试使用真实的AutoDL任务配置，但不执行实际的训练过程。

## 测试输出目录

- `checkpoints/`: 测试过程中生成的检查点文件
- `test_logs/`: 测试日志输出
- `test_benchmark_results/`: 性能基准测试结果
- `test_optimization_results/`: 优化结果输出
- `test_visualizer_output/`: 可视化输出（图表等）

## 注意事项

1. 部分测试需要较长时间运行，特别是集成测试
2. 测试可能会在测试目录下生成临时文件和目录
3. 运行测试前确保已安装所有依赖包
4. 某些测试使用随机种子，结果应该可重现

## 维护指南

### 添加新测试

1. 根据测试功能选择合适的分类文件夹
2. 遵循现有测试文件的命名规范 (`test_<module>.py`)
3. 使用 pytest 或 unittest 框架
4. 添加必要的测试文档和注释
5. 更新本 README 文件

### 测试命名规范

- 测试类使用描述性名称：`Test<ModuleName>`
- 测试方法使用描述性名称：`test_<feature_being_tested>`
- 文件名使用下划线分隔：`test_<module_name>.py`

## 贡献

欢迎提交新的测试用例和改进建议。请确保：

- 新测试通过所有现有测试
- 添加适当的文档说明
- 遵循现有的代码风格

## 许可证

与主项目保持一致。