# 可视化器（Visualizer）使用指南

## 概述

可视化器（Visualizer）是贝叶斯超参数优化系统的可视化组件，提供了丰富的图表和分析功能，帮助用户理解优化过程、参数影响和结果分析。

## 主要功能

### 1. 收敛曲线可视化
- 显示优化过程中目标函数值的变化
- 支持平滑曲线和置信区间
- 标记最佳结果点
- 显示优化统计信息

### 2. 参数分布分析
- 展示各参数值与目标函数的关系
- 自动识别连续型和分类型参数
- 支持散点图和箱线图
- 显示参数重要性排序

### 3. 参数相关性热力图
- 分析参数之间的相关性
- 支持多种相关性计算方法
- 自定义颜色映射
- 只显示下三角矩阵避免冗余

### 4. 参数重要性图表
- 基于敏感性分析的参数重要性排序
- 多指标对比（敏感性、相关性、互信息）
- 支持条形图和雷达图
- 可配置显示参数数量

### 5. 性能热力图
- 二维参数空间的性能分布
- 支持连续参数插值和分类参数分组
- 自动选择最重要的参数组合
- 3D表面图可视化

### 6. 帕累托前沿分析
- 多目标优化的帕累托前沿可视化
- 标记帕累托最优解
- 显示前沿统计信息
- 支持自定义目标函数

### 7. 参数演化分析
- 显示参数值随迭代次数的变化
- 支持连续参数和分类参数
- 颜色映射表示参数值
- 可选择特定参数进行分析

### 8. 交互式仪表板
- 基于Plotly的交互式可视化
- 多图表综合展示
- 支持缩放、平移等交互操作
- 导出HTML格式

## 快速开始

```python
from visualizer import Visualizer
from autodl_core import OptimizationHistory

# 假设你已经有了优化历史数据
history = OptimizationHistory()
# ... 添加优化结果 ...

# 创建可视化器
visualizer = Visualizer(history, parameter_space)

# 生成收敛曲线
visualizer.plot_convergence_curve(save_path="convergence.png")

# 生成综合报告
visualizer.generate_comprehensive_report("output_dir")
```

## 详细使用方法

### 创建可视化器

```python
# 方法1：从优化历史创建
visualizer = Visualizer(optimization_history, parameter_space)

# 方法2：从检查点文件创建
visualizer = create_visualizer_from_checkpoint("checkpoint.json")
```

### 单独生成图表

```python
# 收敛曲线（支持平滑和置信区间）
visualizer.plot_convergence_curve(
    save_path="convergence.png",
    show_confidence_interval=True,
    smooth=True,
    window_size=5
)

# 参数分布
visualizer.plot_parameter_distributions(
    save_path="distributions.png",
    max_params=12
)

# 参数重要性
visualizer.plot_parameter_importance(
    save_path="importance.png",
    top_k=15
)
```
### 自定义分析

```python
# 指定特定参数的性能热力图
visualizer.plot_performance_heatmap(
    save_path="custom_heatmap.png",
    param1="lr",
    param2="batch"
)

# 分析特定参数的演化
visualizer.plot_parameter_evolution(
    save_path="evolution.png",
    params=["lr", "dimensions", "hidden1"]
)

# 3D优化景观
visualizer.plot_optimization_landscape_3d(
    save_path="landscape_3d.png",
    param1="lr",
    param2="dimensions"
)
```

### 生成综合报告

```python
# 一键生成所有可视化图表
visualizer.generate_comprehensive_report("output_directory")
```

这将生成以下文件：
- `convergence_curve.png` - 收敛曲线
- `parameter_distributions.png` - 参数分布
- `parameter_correlations.png` - 参数相关性热力图
- `parameter_importance.png` - 参数重要性
- `performance_heatmap.png` - 性能热力图
- `pareto_frontier.png` - 帕累托前沿（如果有多目标数据）
- `parameter_evolution.png` - 参数演化
- `optimization_landscape_3d.png` - 3D优化景观
- `interactive_dashboard.html` - 交互式仪表板

## 配置选项

### 绘图风格设置

可视化器自动设置中文字体和高质量绘图参数：
- 支持中文标签和标题
- 300 DPI高分辨率输出
- 抗锯齿和平滑线条
- 专业的配色方案

### 参数类型识别

可视化器能够自动识别参数类型：
- **连续型参数**：数值范围大，用散点图和趋势线
- **离散型参数**：数值选项有限，用散点图
- **分类型参数**：字符串类型，用箱线图和分组

### 数据处理

- 自动处理缺失值和异常值
- 支持大规模数据的采样和聚合
- 智能选择重要参数进行可视化
- 自动调整图表布局和大小

## 依赖库

必需依赖：
```bash
pip install matplotlib seaborn pandas numpy scipy scikit-learn
```

可选依赖（用于交互式功能）：
```bash
pip install plotly
```

## 示例代码

完整的使用示例请参考 `visualizer_example.py` 文件，包含：
- 基本可视化功能演示
- 综合报告生成
- 自定义分析示例
- 从检查点加载数据

## 注意事项

1. **内存使用**：大规模数据集可能需要较多内存，建议适当采样
2. **文件格式**：默认保存PNG格式，支持SVG等矢量格式
3. **中文字体**：自动配置中文字体，如有问题请检查系统字体
4. **交互式功能**：需要安装plotly库才能使用交互式仪表板

## 故障排除

### 常见问题

1. **中文显示问题**
   ```python
   # 手动设置中文字体
   plt.rcParams['font.sans-serif'] = ['SimHei']
   ```

2. **内存不足**
   ```python
   # 限制显示的参数数量
   visualizer.plot_parameter_distributions(max_params=8)
   ```

3. **plotly导入错误**
   ```bash
   pip install plotly
   ```

### 性能优化

- 对于大数据集，使用采样减少计算量
- 限制显示的参数数量
- 选择合适的图表类型
- 使用批量生成避免重复计算

## 扩展功能

可视化器设计为可扩展的，你可以：
- 添加新的图表类型
- 自定义颜色方案
- 集成其他可视化库
- 添加新的分析指标

## 更新日志

- v1.0: 初始版本，包含基本可视化功能
- 支持收敛曲线、参数分布、重要性分析
- 支持帕累托前沿和3D景观图
- 集成交互式仪表板功能