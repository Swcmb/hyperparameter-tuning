# 贝叶斯超参数优化系统示例代码

本目录包含了贝叶斯超参数优化系统的完整示例代码，涵盖从入门到高级的各种使用场景。

## 目录结构

```
examples/
├── core_examples/           # 核心功能示例
├── advanced_algorithms/     # 高级算法示例
├── application_scenarios/   # 应用场景示例
├── analysis_tools/          # 分析工具示例
├── utility_tools/           # 实用工具示例
├── config_examples/         # 配置文件示例
└── example_reports/         # 示例报告输出
```

## 文件夹说明

### 📁 core_examples/ - 核心功能示例

这些示例展示了系统的基础功能和核心数据结构的使用方法。

- **`example_usage.py`** - 核心数据结构使用示例
  - 演示参数空间的创建和采样
  - 展示参数验证和修复功能
  - 介绍优化结果和历史记录的管理
  - 展示配置转换功能

- **`quick_start_example.py`** - 快速开始示例
  - 提供最简单的使用方法
  - 演示基本的优化流程
  - 包含命令行交互界面
  - 适合新手上手

- **`parameter_space_example.py`** - 参数空间管理器使用示例
  - 详细展示参数空间的创建和使用
  - 演示随机采样和参数验证
  - 展示参数修复功能
  - 包含序列化和反序列化示例

**使用建议**：建议从 `quick_start_example.py` 开始学习，然后逐步查看其他核心示例。

---

### 📁 advanced_algorithms/ - 高级算法示例

这些示例展示了系统底层算法的详细实现和使用方法。

- **`acquisition_function_demo.py`** - 采集函数演示
  - 展示所有支持的采集函数类型（EI、PI、UCB、ES）
  - 演示参数验证和更新功能
  - 展示采集函数优化方法
  - 包含多维优化演示
  - 展示参数敏感性分析

- **`gaussian_process_example.py`** - 高斯过程模型使用示例
  - 演示高斯过程模型的创建和训练
  - 展示一维和参数空间优化
  - 比较不同采集函数的行为
  - 展示模型的持久化和加载

**使用建议**：适合需要深入理解贝叶斯优化算法原理的用户。

---

### 📁 application_scenarios/ - 应用场景示例

这些示例展示了系统在具体应用场景中的使用方法。

- **`moco_parameter_example.py`** - MoCo参数优化使用示例
  - 展示如何优化MoCo对比学习参数
  - 演示MoCo参数约束验证
  - 提供完整的优化工作流
  - 包含参数配置建议

- **`multi_objective_example.py`** - 多目标贝叶斯优化示例
  - 演示多目标优化的使用方法
  - 展示帕累托前沿计算和可视化
  - 测试不同目标权重配置
  - 展示收敛性分析

- **`simple_multi_objective_test.py`** - 简单的多目标优化测试
  - 验证多目标优化功能
  - 测试加权目标函数计算
  - 测试帕累托前沿计算
  - 使用模拟数据，无需实际训练

**使用建议**：根据具体应用场景选择对应的示例进行学习。

---

### 📁 analysis_tools/ - 分析工具示例

这些示例展示了如何使用系统的分析工具来理解优化结果。

- **`report_generator_example.py`** - 报告生成器使用示例
  - 演示基本报告生成功能（JSON、HTML、PDF）
  - 展示综合报告生成
  - 演示自定义分析报告
  - 展示如何访问和使用报告数据

- **`result_analyzer_example.py`** - 结果分析器使用示例
  - 演示基本分析功能（统计摘要、敏感性分析、收敛性分析）
  - 展示高级分析功能（参数相关性、模式识别）
  - 演示报告生成
  - 创建可视化示例

- **`visualizer_example.py`** - 可视化器使用示例
  - 演示基本可视化功能（收敛曲线、参数分布、相关性热力图等）
  - 展示综合报告生成
  - 演示自定义分析可视化
  - 展示从检查点加载可视化器

**使用建议**：完成优化后使用这些工具深入分析结果。

---

### 📁 utility_tools/ - 实用工具示例

这些示例展示了系统提供的实用工具功能。

- **`state_manager_example.py`** - 状态管理器使用示例
  - 演示带检查点的优化过程
  - 展示状态保存和恢复功能
  - 演示检查点管理功能
  - 展示错误恢复功能

- **`task_evaluator_example.py`** - TaskEvaluator使用示例
  - 演示贝叶斯优化过程中的参数评估
  - 展示参数验证功能
  - 演示多任务评估
  - 展示模拟评估模式

**使用建议**：适合需要实现高级功能的开发者。

---

### 📁 config_examples/ - 配置文件示例

包含各种配置文件的示例。

- **`config_example.json`** - 基本配置文件示例
- **`moco_config_example.json`** - MoCo任务配置文件示例
- **`parameter_space_config.json`** - 参数空间配置文件示例
- **`config_comparison.json`** - 配置对比文件
- **`default_config.json`** - 默认配置文件
- **`optimized_config.json`** - 优化后的配置文件
- **`implementation_guide.md`** - 实现指南

---

### 📁 example_reports/ - 示例报告输出

包含示例代码生成的报告文件。

- **`basic/`** - 基本报告示例
  - `optimization_report.html` - HTML格式报告
  - `optimization_report.json` - JSON格式报告

- **`comprehensive/`** - 综合报告示例
  - `comprehensive_optimization_report.html` - 综合HTML报告
  - `comprehensive_optimization_report.json` - 综合JSON报告

- **`custom/`** - 自定义分析报告示例
  - `sensitivity_analysis_report.html` - 敏感性分析HTML报告
  - `sensitivity_analysis_report.json` - 敏感性分析JSON报告

---

## 快速开始

### 1. 运行快速开始示例

```bash
cd core_examples
python quick_start_example.py
```

### 2. 查看核心功能示例

```bash
cd core_examples
python example_usage.py
```

### 3. 尝试高级算法

```bash
cd advanced_algorithms
python gaussian_process_example.py
```

### 4. 应用到具体场景

```bash
cd application_scenarios
python moco_parameter_example.py
```

### 5. 分析优化结果

```bash
cd analysis_tools
python result_analyzer_example.py
```

---

## 示例代码运行要求

- Python 3.7+
- 依赖库：numpy, matplotlib, scipy, scikit-learn
- 可选依赖：plotly（用于交互式可视化）、weasyprint（用于PDF生成）

---

## 学习路径建议

### 初学者
1. `core_examples/quick_start_example.py` - 快速了解系统
2. `core_examples/example_usage.py` - 学习核心功能
3. `core_examples/parameter_space_example.py` - 理解参数管理

### 进阶用户
1. `advanced_algorithms/gaussian_process_example.py` - 理解底层算法
2. `advanced_algorithms/acquisition_function_demo.py` - 学习采集函数
3. `application_scenarios/moco_parameter_example.py` - 实际应用场景

### 高级用户
1. `application_scenarios/multi_objective_example.py` - 多目标优化
2. `analysis_tools/report_generator_example.py` - 报告生成
3. `analysis_tools/visualizer_example.py` - 高级可视化
4. `utility_tools/state_manager_example.py` - 状态管理

---

## 注意事项

1. 部分示例需要较长的运行时间，建议先从简单的示例开始
2. 某些可视化示例需要安装额外的依赖库（plotly、weasyprint）
3. 示例代码中的某些参数配置可能需要根据实际任务进行调整
4. 运行示例前请确保已正确安装所有依赖库

---

## 技术支持

如有问题，请参考主项目的README文件或提交Issue。

---

## 更新日志

- 2026-01-31: 重新组织示例代码结构，按功能分类
- 示例代码持续更新中...