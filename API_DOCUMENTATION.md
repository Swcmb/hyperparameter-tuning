# AutoDL贝叶斯优化系统 API文档

## 概述

本文档详细描述了AutoDL贝叶斯优化系统的API接口，包括所有核心组件的类、方法和参数说明。

## 核心组件

### 1. autodl_core.py - 核心数据结构

#### ParameterType (枚举)
参数类型枚举类。

```python
class ParameterType(Enum):
    CONTINUOUS = "continuous"    # 连续型参数
    DISCRETE = "discrete"        # 离散型参数
    CATEGORICAL = "categorical"  # 分类型参数
```

#### ParameterConfig (数据类)
参数配置数据类，定义单个参数的类型、范围和约束条件。

```python
@dataclass
class ParameterConfig:
    name: str                                    # 参数名称
    param_type: ParameterType                   # 参数类型
    bounds: Optional[Tuple[float, float]]       # 连续型参数的范围
    values: Optional[List[Any]]                 # 离散型或分类型参数的可选值
    log_scale: bool = False                     # 是否使用对数尺度
    constraints: Optional[List[str]] = None     # 参数约束条件
```

**主要方法:**
- `validate_value(value: Any) -> bool`: 验证给定值是否符合参数配置
- `sample_random_value(rng: np.random.Generator) -> Any`: 随机采样一个有效的参数值
- `to_dict() -> Dict[str, Any]`: 转换为字典格式
- `from_dict(data: Dict[str, Any]) -> 'ParameterConfig'`: 从字典创建实例

#### OptimizationResult (数据类)
单次优化结果数据类，记录一次参数评估的完整信息。

```python
@dataclass
class OptimizationResult:
    parameters: Dict[str, Any]                  # 参数组合
    objective_value: float                      # 主要优化目标值
    metrics: Dict[str, float]                   # 所有性能指标
    iteration: int                              # 迭代次数
    timestamp: datetime                         # 评估时间戳
    evaluation_time: float                      # 评估耗时（秒）
    fold_results: Optional[Dict[str, List[float]]] = None  # 各折的详细结果
    error_info: Optional[str] = None            # 错误信息
    
    # 多目标优化支持
    objective_values: Optional[Dict[str, float]] = None  # 多个目标函数值
    is_pareto_optimal: bool = False             # 是否为帕累托最优解
    dominance_count: int = 0                    # 被支配的解的数量
    dominated_by: List[int] = field(default_factory=list)  # 支配此解的解的索引
```

**主要方法:**
- `is_better_than(other: 'OptimizationResult', maximize: bool = True) -> bool`: 比较两个结果的优劣
- `dominates(other: 'OptimizationResult', objectives: List[str], maximize: Dict[str, bool]) -> bool`: 检查帕累托支配关系
- `get_objective_vector(objectives: List[str]) -> List[float]`: 获取目标函数向量

#### OptimizationHistory (数据类)
优化历史数据类，记录整个优化过程的历史信息。

```python
@dataclass
class OptimizationHistory:
    results: List[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    parameter_space: Dict[str, Any] = field(default_factory=dict)
    acquisition_function: str = "EI"
    task_type: str = "LDA"
    total_iterations: int = 0
    total_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # 多目标优化支持
    objectives: List[str] = field(default_factory=lambda: ['primary'])
    maximize_objectives: Dict[str, bool] = field(default_factory=lambda: {'primary': True})
    pareto_front: List[OptimizationResult] = field(default_factory=list)
    objective_weights: Optional[Dict[str, float]] = None
```

**主要方法:**
- `add_result(result: OptimizationResult, maximize: bool = True)`: 添加新的优化结果
- `compute_pareto_front() -> List[OptimizationResult]`: 重新计算完整的帕累托前沿
- `get_weighted_objective_value(result: OptimizationResult) -> float`: 计算加权目标函数值
- `set_objectives(objectives: List[str], maximize: Dict[str, bool], weights: Optional[Dict[str, float]] = None)`: 设置多目标优化配置
- `get_convergence_curve() -> List[float]`: 获取收敛曲线
- `get_parameter_history(param_name: str) -> List[Any]`: 获取特定参数的历史值

#### ParameterSpace (类)
参数空间管理器，管理所有可优化参数的定义、验证和采样。

```python
class ParameterSpace:
    def __init__(self):
        self.parameters: Dict[str, ParameterConfig] = {}
        self.constraints: List[str] = []
```

**主要方法:**
- `add_parameter(config: ParameterConfig)`: 添加参数配置
- `add_continuous_parameter(name: str, low: float, high: float, log_scale: bool = False, constraints: Optional[List[str]] = None)`: 添加连续型参数
- `add_discrete_parameter(name: str, values: List[Union[int, float]], constraints: Optional[List[str]] = None)`: 添加离散型参数
- `add_categorical_parameter(name: str, categories: List[str], constraints: Optional[List[str]] = None)`: 添加分类型参数
- `validate_parameters(parameters: Dict[str, Any]) -> bool`: 验证参数组合是否有效
- `sample_random_parameters(seed: Optional[int] = None, max_attempts: int = 100) -> Dict[str, Any]`: 随机采样参数组合
- `suggest_parameter_fix(parameters: Dict[str, Any]) -> Dict[str, Any]`: 建议参数修复

**工厂函数:**
- `create_default_parameter_space(task_type: str = "LDA") -> ParameterSpace`: 创建默认的参数空间配置

### 2. bayesian_optimizer.py - 贝叶斯优化器

#### BayesianOptimizer (类)
主要的贝叶斯优化协调器类。

```python
class BayesianOptimizer:
    def __init__(self, parameter_space: ParameterSpace, task_evaluator, 
                 acquisition_function: AcquisitionFunction, 
                 gaussian_process: GaussianProcess, random_seed: int = 42):
```

**主要方法:**
- `suggest_parameters() -> Dict[str, Any]`: 建议下一个要评估的参数组合
- `update_with_result(result: OptimizationResult)`: 使用评估结果更新模型
- `get_state() -> Dict[str, Any]`: 获取优化器状态
- `load_state(state_data: Dict[str, Any])`: 加载优化器状态
- `get_best_result() -> Optional[OptimizationResult]`: 获取当前最佳结果

#### MultiObjectiveOptimizer (类)
多目标贝叶斯优化器。

```python
class MultiObjectiveOptimizer(BayesianOptimizer):
    def __init__(self, parameter_space: ParameterSpace, task_evaluator,
                 objectives: List[str], maximize_objectives: Dict[str, bool],
                 objective_weights: Optional[Dict[str, float]] = None, ...):
```

**额外方法:**
- `get_pareto_front() -> List[OptimizationResult]`: 获取当前帕累托前沿
- `compute_weighted_objective(result: OptimizationResult) -> float`: 计算加权目标函数值

**工厂函数:**
- `create_bayesian_optimizer(...)`: 创建标准贝叶斯优化器
- `create_multi_objective_optimizer(...)`: 创建多目标优化器

### 3. task_evaluator.py - 任务评估器

#### TaskEvaluator (类)
任务评估器，执行具体的模型训练和评估。

```python
class TaskEvaluator:
    def __init__(self, task_type: str = "LDA", data_path: Optional[str] = None,
                 cv_folds: int = 5, random_seed: int = 42):
```

**主要方法:**
- `evaluate_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]`: 评估参数组合
- `validate_parameters(parameters: Dict[str, Any]) -> Tuple[bool, List[str]]`: 验证参数有效性
- `get_evaluation_metrics() -> List[str]`: 获取支持的评估指标

**工厂函数:**
- `create_task_evaluator(task_type: str = "LDA", ...) -> TaskEvaluator`: 创建任务评估器

### 4. acquisition_function.py - 采集函数

#### AcquisitionFunction (抽象基类)
采集函数基类。

```python
class AcquisitionFunction(ABC):
    @abstractmethod
    def compute(self, X: np.ndarray, gp: GaussianProcess, 
                best_value: float) -> np.ndarray:
```

#### ExpectedImprovement (类)
期望改进采集函数。

```python
class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, xi: float = 0.01):
```

#### ProbabilityOfImprovement (类)
改进概率采集函数。

```python
class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, xi: float = 0.01):
```

#### UpperConfidenceBound (类)
上置信界采集函数。

```python
class UpperConfidenceBound(AcquisitionFunction):
    def __init__(self, beta: float = 2.0):
```

**工厂函数:**
- `create_acquisition_function(name: str, params: Dict[str, Any] = {}) -> AcquisitionFunction`: 创建采集函数

### 5. gaussian_process.py - 高斯过程

#### GaussianProcess (类)
高斯过程模型包装类。

```python
class GaussianProcess:
    def __init__(self, kernel_type: str = "matern52", alpha: float = 1e-6,
                 n_restarts_optimizer: int = 10, random_seed: int = 42):
```

**主要方法:**
- `fit(X: np.ndarray, y: np.ndarray)`: 拟合高斯过程模型
- `predict(X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]`: 预测均值和标准差
- `update(X_new: np.ndarray, y_new: np.ndarray)`: 增量更新模型
- `get_state() -> Dict[str, Any]`: 获取模型状态
- `load_state(state_data: Dict[str, Any])`: 加载模型状态

### 6. state_manager.py - 状态管理器

#### StateManager (类)
优化状态的保存和恢复功能。

```python
class StateManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", 
                 save_frequency: int = 1, max_checkpoints: int = 10):
```

**主要方法:**
- `save_state(state_data: Dict[str, Any], checkpoint_name: str)`: 保存状态
- `load_state(checkpoint_name: str) -> Optional[Dict[str, Any]]`: 加载状态
- `list_checkpoints() -> List[Dict[str, Any]]`: 列出所有检查点
- `cleanup_old_checkpoints(keep_count: int = 5)`: 清理旧检查点
- `validate_checkpoint(checkpoint_path: str) -> bool`: 验证检查点完整性

### 7. result_analyzer.py - 结果分析器

#### ResultAnalyzer (类)
参数敏感性分析和优化历史统计分析。

```python
class ResultAnalyzer:
    def __init__(self, history: OptimizationHistory, 
                 parameter_space: ParameterSpace):
```

**主要方法:**
- `analyze_parameter_sensitivity() -> Dict[str, ParameterSensitivityResult]`: 分析参数敏感性
- `analyze_convergence() -> ConvergenceAnalysisResult`: 分析收敛性
- `generate_statistical_summary() -> StatisticalSummary`: 生成统计摘要
- `get_parameter_importance_ranking() -> List[Tuple[str, float]]`: 获取参数重要性排序

### 8. visualizer.py - 可视化器

#### Visualizer (类)
生成收敛曲线、参数分布和性能热力图。

```python
class Visualizer:
    def __init__(self, history: OptimizationHistory, 
                 parameter_space: ParameterSpace):
```

**主要方法:**
- `plot_convergence_curve(save_path: str)`: 绘制收敛曲线
- `plot_parameter_distribution(save_path: str)`: 绘制参数分布
- `plot_parameter_correlation(save_path: str)`: 绘制参数相关性热力图
- `plot_pareto_front(save_path: str)`: 绘制帕累托前沿（多目标优化）
- `plot_parameter_importance(save_path: str)`: 绘制参数重要性图表

### 9. report_generator.py - 报告生成器

#### ReportGenerator (类)
生成详细的优化报告。

```python
class ReportGenerator:
    def __init__(self, history: OptimizationHistory, 
                 parameter_space: ParameterSpace,
                 result_analyzer: ResultAnalyzer,
                 visualizer: Visualizer, config: ReportConfig):
```

**主要方法:**
- `generate_json_report(output_path: str)`: 生成JSON格式报告
- `generate_html_report(output_path: str)`: 生成HTML格式报告
- `generate_pdf_report(output_path: str)`: 生成PDF格式报告

#### ReportConfig (数据类)
报告配置。

```python
@dataclass
class ReportConfig:
    title: str = "贝叶斯超参数优化报告"
    author: str = "AutoDL系统"
    include_charts: bool = True
    include_parameter_details: bool = True
    include_convergence_analysis: bool = True
    include_sensitivity_analysis: bool = True
```

### 10. autodl.py - 主入口脚本

#### AutoDLOptimizer (类)
贝叶斯超参数优化系统主类。

```python
class AutoDLOptimizer:
    def __init__(self, config: Dict[str, Any]):
```

**主要方法:**
- `initialize_components()`: 初始化所有组件
- `run_optimization()`: 运行完整的贝叶斯优化流程
- `_save_state()`: 保存当前优化状态
- `_generate_final_report()`: 生成最终分析报告

## 使用示例

### 基本使用流程

```python
from autodl_core import create_default_parameter_space
from bayesian_optimizer import create_bayesian_optimizer
from task_evaluator import create_task_evaluator

# 1. 创建参数空间
parameter_space = create_default_parameter_space("LDA")

# 2. 创建任务评估器
task_evaluator = create_task_evaluator(
    task_type="LDA",
    cv_folds=5,
    random_seed=42
)

# 3. 创建贝叶斯优化器
optimizer = create_bayesian_optimizer(
    parameter_space=parameter_space,
    task_evaluator=task_evaluator,
    acquisition_function="EI",
    random_seed=42
)

# 4. 运行优化循环
for iteration in range(50):
    # 获取参数建议
    suggested_params = optimizer.suggest_parameters()
    
    # 评估参数
    evaluation_result = task_evaluator.evaluate_parameters(suggested_params)
    
    # 创建结果对象
    result = OptimizationResult(
        parameters=suggested_params,
        objective_value=evaluation_result['objective_value'],
        metrics=evaluation_result['metrics'],
        iteration=iteration + 1,
        timestamp=datetime.now(),
        evaluation_time=evaluation_result.get('evaluation_time', 0.0)
    )
    
    # 更新优化器
    optimizer.update_with_result(result)
```

### 多目标优化示例

```python
from bayesian_optimizer import create_multi_objective_optimizer

# 创建多目标优化器
objectives = ['AUROC', 'AUPRC', 'F1']
maximize_objectives = {'AUROC': True, 'AUPRC': True, 'F1': True}
objective_weights = {'AUROC': 0.5, 'AUPRC': 0.3, 'F1': 0.2}

optimizer = create_multi_objective_optimizer(
    parameter_space=parameter_space,
    task_evaluator=task_evaluator,
    objectives=objectives,
    maximize_objectives=maximize_objectives,
    objective_weights=objective_weights,
    acquisition_function="EI",
    random_seed=42
)

# 运行优化并获取帕累托前沿
# ... 运行优化循环 ...
pareto_front = optimizer.get_pareto_front()
```

### 状态保存和恢复示例

```python
from state_manager import create_default_state_manager

# 创建状态管理器
state_manager = create_default_state_manager(
    checkpoint_dir="checkpoints",
    save_frequency=5
)

# 保存状态
state_data = {
    'history': history.to_dict(),
    'optimizer_state': optimizer.get_state(),
    'iteration': current_iteration
}
state_manager.save_state(state_data, f"iteration_{current_iteration}")

# 恢复状态
loaded_state = state_manager.load_state("iteration_20")
if loaded_state:
    history = OptimizationHistory.from_dict(loaded_state['history'])
    optimizer.load_state(loaded_state['optimizer_state'])
```

## 错误处理

### 常见异常类型

- `ValueError`: 参数值无效或超出范围
- `RuntimeError`: 优化过程中的运行时错误
- `CheckpointError`: 检查点保存或加载失败
- `ParameterValidationError`: 参数验证失败

### 错误处理最佳实践

```python
try:
    # 参数采样
    params = parameter_space.sample_random_parameters()
    
    # 参数验证
    is_valid, errors = parameter_space.validate_parameters_detailed(params)
    if not is_valid:
        # 尝试修复参数
        params = parameter_space.suggest_parameter_fix(params)
    
    # 参数评估
    result = task_evaluator.evaluate_parameters(params)
    
except ValueError as e:
    print(f"参数错误: {e}")
except RuntimeError as e:
    print(f"评估失败: {e}")
    # 跳过此次评估，继续下一次
except Exception as e:
    print(f"未知错误: {e}")
    # 记录错误并决定是否继续
```

## 配置参数说明

### 优化器配置

```python
optimizer_config = {
    'acquisition_function': 'EI',  # 采集函数类型
    'acquisition_params': {        # 采集函数参数
        'xi': 0.01                 # EI的探索参数
    },
    'gaussian_process_params': {   # 高斯过程参数
        'kernel': 'matern52',      # 核函数类型
        'alpha': 1e-6,             # 噪声水平
        'n_restarts_optimizer': 10 # 优化重启次数
    }
}
```

### 任务评估器配置

```python
evaluator_config = {
    'task_type': 'LDA',           # 任务类型
    'cv_folds': 5,                # 交叉验证折数
    'random_seed': 42,            # 随机种子
    'data_path': '/path/to/data', # 数据路径
    'early_stopping': True,       # 是否启用早停
    'patience': 10                # 早停耐心值
}
```

### 状态管理配置

```python
state_config = {
    'checkpoint_dir': 'checkpoints',  # 检查点目录
    'save_frequency': 1,              # 保存频率
    'max_checkpoints': 10,            # 最大检查点数量
    'compression': True               # 是否压缩检查点
}
```

## 性能优化建议

### 参数空间设计
- 合理设置参数范围，避免过大的搜索空间
- 对于大范围的参数使用对数尺度
- 添加合适的参数约束以减少无效采样

### 采集函数选择
- EI适合大多数情况，平衡探索和利用
- UCB适合需要更多探索的场景
- PI适合保守的优化策略

### 计算资源管理
- 合理设置交叉验证折数
- 使用检查点避免重复计算
- 监控内存和CPU使用情况

### 收敛加速
- 增加初始随机采样数量
- 调整采集函数参数
- 使用更好的初始参数估计