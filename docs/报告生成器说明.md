# 报告生成器（ReportGenerator）

报告生成器是贝叶斯超参数优化系统的重要组件，用于生成详细的优化报告，包含实验配置、最佳参数、性能指标和统计分析。支持多种输出格式（JSON、HTML、PDF）。

## 功能特性

### 核心功能
- **实验配置报告**: 包含任务类型、采集函数、参数空间配置等
- **优化摘要**: 总评估次数、最佳结果、成功率等关键指标
- **最佳参数分析**: 最佳参数组合、性能指标、参数推荐
- **收敛性分析**: 收敛状态、改进速率、平台期分析
- **参数敏感性分析**: 参数重要性排序、相关性分析
- **可视化图表**: 收敛曲线、参数分布、重要性图表
- **优化建议**: 基于分析结果的智能建议

### 输出格式
- **JSON**: 结构化数据，便于程序处理
- **HTML**: 可视化报告，包含图表和格式化内容
- **PDF**: 打印友好的报告格式（需要weasyprint）

## 安装依赖

### 基础依赖
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### 可选依赖
```bash
# HTML模板支持
pip install jinja2

# PDF生成支持
pip install weasyprint

# 交互式图表支持
pip install plotly
```

## 快速开始

### 基本使用

```python
from report_generator import ReportGenerator, ReportConfig
from autodl_core import OptimizationHistory

# 假设你已经有了优化历史数据
history = OptimizationHistory()  # 加载你的优化历史
parameter_space = create_default_parameter_space()  # 参数空间

# 创建报告生成器
generator = ReportGenerator(history, parameter_space)

# 生成HTML报告
generator.save_html_report("optimization_report.html")

# 生成JSON报告
generator.save_json_report("optimization_report.json")

# 生成PDF报告（需要weasyprint）
generator.save_pdf_report("optimization_report.pdf")
```

### 自定义配置

```python
from report_generator import ReportConfig

# 创建自定义配置
config = ReportConfig(
    title="我的优化报告",
    author="研究团队",
    include_charts=True,
    include_parameter_details=True,
    include_convergence_analysis=True,
    include_sensitivity_analysis=True,
    chart_dpi=300,
    max_parameters_in_charts=15
)

generator = ReportGenerator(history, parameter_space, config=config)
```

### 生成所有格式

```python
# 一次性生成所有格式的报告
generator.generate_all_formats(
    output_dir="reports",
    base_filename="comprehensive_report"
)
```

## 详细功能说明

### 1. 实验配置部分

包含以下信息：
- 任务类型（LDA/MDA/LMI）
- 采集函数类型
- 优化开始和结束时间
- 参数空间摘要
- 参数详细配置（可选）

### 2. 优化摘要

提供优化过程的关键统计信息：
- 总评估次数和成功率
- 最佳目标值和对应迭代
- 相对于基线的改进幅度
- 评估时间统计

### 3. 最佳参数分析

深入分析最佳参数组合：
- 最佳单个结果的详细信息
- 前k个最佳结果的统计分析
- 基于统计的参数推荐
- 参数值分布分析

### 4. 收敛性分析

评估优化过程的收敛状态：
- 收敛检测和收敛迭代
- 改进速率和最终改进幅度
- 平台期长度分析
- 收敛曲线可视化

### 5. 参数敏感性分析

识别重要参数和参数关系：
- 参数重要性排序
- 敏感性得分和统计显著性
- 参数相关性矩阵
- 多种分析方法（相关性、互信息、随机森林）

### 6. 可视化图表

自动生成多种图表：
- 收敛曲线图
- 参数重要性条形图
- 目标函数值分布直方图
- 参数相关性热力图（在完整可视化中）

### 7. 优化建议

基于分析结果提供智能建议：
- 参数调优建议
- 优化策略建议
- 下一步行动建议

## 配置选项

### ReportConfig 参数说明

```python
@dataclass
class ReportConfig:
    title: str = "贝叶斯超参数优化报告"  # 报告标题
    author: str = "AutoDL系统"  # 报告作者
    include_charts: bool = True  # 是否包含图表
    include_parameter_details: bool = True  # 是否包含参数详情
    include_convergence_analysis: bool = True  # 是否包含收敛分析
    include_sensitivity_analysis: bool = True  # 是否包含敏感性分析
    include_best_parameters_analysis: bool = True  # 是否包含最佳参数分析
    chart_dpi: int = 300  # 图表分辨率
    max_parameters_in_charts: int = 15  # 图表中最大参数数量
    language: str = "zh"  # 语言设置
```

## 高级用法

### 从检查点文件生成报告

```python
from report_generator import create_report_generator_from_checkpoint

# 从检查点文件创建报告生成器
generator = create_report_generator_from_checkpoint(
    checkpoint_path="checkpoints/optimization_state.pkl",
    config=ReportConfig(title="从检查点恢复的报告")
)

if generator:
    generator.save_html_report("recovered_report.html")
```

### 访问报告数据

```python
# 获取结构化的报告数据
report_data = generator.generate_report_data()

# 访问特定部分
best_params = report_data['best_parameters']['best_single_result']['parameters']
importance_ranking = report_data['parameter_analysis']['importance_ranking']
convergence_info = report_data['convergence_analysis']

# 自定义处理报告数据
for param_name, score in importance_ranking[:5]:
    print(f"{param_name}: {score:.4f}")
```

### 自定义HTML模板

如果安装了jinja2，可以自定义HTML模板：

```python
# 报告生成器会自动使用jinja2模板
# 模板包含丰富的样式和交互功能
generator.save_html_report("styled_report.html")
```

## 输出示例

### JSON报告结构
```json
{
  "metadata": {
    "title": "贝叶斯超参数优化报告",
    "generation_time": "2024-01-15T10:30:00",
    "system_info": {...}
  },
  "experiment_configuration": {...},
  "optimization_summary": {...},
  "best_parameters": {...},
  "convergence_analysis": {...},
  "parameter_analysis": {...},
  "recommendations": {...}
}
```

### HTML报告特性
- 响应式设计，适配不同屏幕尺寸
- 丰富的CSS样式和布局
- 内嵌的Base64编码图表
- 清晰的章节结构和导航
- 专业的报告外观

### PDF报告特性
- 打印友好的布局
- 高质量的图表渲染
- 完整的内容包含
- 专业的文档格式

## 性能考虑

### 大数据集优化
- 图表生成限制参数数量（max_parameters_in_charts）
- 历史记录摘要只显示最近结果
- 可选择性包含分析部分以减少计算时间

### 内存使用
- 图表数据使用Base64编码缓存
- 报告数据支持延迟计算
- 大型数据集建议分批处理

## 故障排除

### 常见问题

1. **PDF生成失败**
   ```bash
   pip install weasyprint
   # 在某些系统上可能需要额外的系统依赖
   ```

2. **图表不显示**
   - 检查matplotlib是否正确安装
   - 确保使用非交互式后端

3. **中文字体问题**
   - 系统需要安装中文字体
   - matplotlib会自动选择可用字体

4. **内存不足**
   - 减少max_parameters_in_charts
   - 关闭不必要的分析选项
   - 分批处理大型数据集

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 生成报告时会输出详细的调试信息
generator.save_html_report("debug_report.html")
```

## 扩展开发

### 添加自定义分析

```python
class CustomReportGenerator(ReportGenerator):
    def _generate_custom_analysis(self):
        # 添加你的自定义分析逻辑
        return {"custom_metric": "custom_value"}
    
    def generate_report_data(self):
        data = super().generate_report_data()
        data['custom_analysis'] = self._generate_custom_analysis()
        return data
```

### 自定义图表

```python
def add_custom_chart(self):
    # 生成自定义图表
    fig, ax = plt.subplots()
    # ... 绘图代码 ...
    return self._fig_to_base64(fig)
```

## 版本历史

- **v1.0.0**: 初始版本，支持基本报告生成功能
- 支持JSON、HTML、PDF三种格式
- 包含完整的分析功能和可视化

## 许可证

本项目采用MIT许可证。详见LICENSE文件。