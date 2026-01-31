"""
贝叶斯超参数优化系统的核心数据结构和配置管理

本模块定义了优化过程中使用的核心数据类和参数空间管理功能。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import json
import numpy as np


class ParameterType(Enum):
    """参数类型枚举"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class ParameterConfig:
    """
    参数配置数据类
    
    定义单个参数的类型、范围和约束条件
    """
    name: str
    param_type: ParameterType
    bounds: Optional[Tuple[float, float]] = None  # 连续型参数的范围
    values: Optional[List[Any]] = None  # 离散型或分类型参数的可选值
    log_scale: bool = False  # 是否使用对数尺度（仅适用于连续型参数）
    constraints: Optional[List[str]] = None  # 参数约束条件
    
    def __post_init__(self):
        """初始化后验证参数配置的有效性"""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.bounds is None or len(self.bounds) != 2:
                raise ValueError(f"连续型参数 {self.name} 必须指定有效的bounds (low, high)")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"参数 {self.name} 的下界必须小于上界")
            if self.log_scale and self.bounds[0] <= 0:
                raise ValueError(f"对数尺度参数 {self.name} 的下界必须大于0")
        
        elif self.param_type in [ParameterType.DISCRETE, ParameterType.CATEGORICAL]:
            if self.values is None or len(self.values) == 0:
                raise ValueError(f"{self.param_type.value}型参数 {self.name} 必须指定有效的values列表")
    
    def validate_value(self, value: Any) -> bool:
        """验证给定值是否符合参数配置"""
        try:
            if self.param_type == ParameterType.CONTINUOUS:
                val = float(value)
                return self.bounds[0] <= val <= self.bounds[1]
            
            elif self.param_type == ParameterType.DISCRETE:
                return value in self.values
            
            elif self.param_type == ParameterType.CATEGORICAL:
                return value in self.values
                
        except (ValueError, TypeError):
            return False
        
        return False
    
    def sample_random_value(self, rng: np.random.Generator) -> Any:
        """随机采样一个有效的参数值"""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.log_scale:
                log_low = np.log(self.bounds[0])
                log_high = np.log(self.bounds[1])
                log_val = rng.uniform(log_low, log_high)
                return np.exp(log_val)
            else:
                return rng.uniform(self.bounds[0], self.bounds[1])
        
        elif self.param_type in [ParameterType.DISCRETE, ParameterType.CATEGORICAL]:
            return rng.choice(self.values)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'param_type': self.param_type.value,
            'bounds': self.bounds,
            'values': self.values,
            'log_scale': self.log_scale,
            'constraints': self.constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterConfig':
        """从字典创建ParameterConfig实例"""
        data = data.copy()
        data['param_type'] = ParameterType(data['param_type'])
        return cls(**data)


@dataclass
class OptimizationResult:
    """
    单次优化结果数据类
    
    记录一次参数评估的完整信息，支持多目标优化
    """
    parameters: Dict[str, Any]  # 参数组合
    objective_value: float  # 主要优化目标值（AUROC）
    metrics: Dict[str, float]  # 所有性能指标（AUROC, AUPRC, F1等）
    iteration: int  # 迭代次数
    timestamp: datetime  # 评估时间戳
    evaluation_time: float  # 评估耗时（秒）
    fold_results: Optional[Dict[str, List[float]]] = None  # 各折的详细结果
    error_info: Optional[str] = None  # 错误信息（如果评估失败）
    
    # 多目标优化支持
    objective_values: Optional[Dict[str, float]] = None  # 多个目标函数值
    is_pareto_optimal: bool = False  # 是否为帕累托最优解
    dominance_count: int = 0  # 被支配的解的数量
    dominated_by: List[int] = field(default_factory=list)  # 支配此解的解的索引
    
    def __post_init__(self):
        """初始化后验证结果数据"""
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters必须是字典类型")
        
        if not isinstance(self.metrics, dict):
            raise ValueError("metrics必须是字典类型")
        
        if self.evaluation_time < 0:
            raise ValueError("evaluation_time不能为负数")
        
        # 如果没有设置多目标值，使用单目标值初始化
        if self.objective_values is None:
            self.objective_values = {'primary': self.objective_value}
    
    def is_better_than(self, other: 'OptimizationResult', maximize: bool = True) -> bool:
        """比较两个结果的优劣（默认最大化目标函数）"""
        if maximize:
            return self.objective_value > other.objective_value
        else:
            return self.objective_value < other.objective_value
    
    def dominates(self, other: 'OptimizationResult', objectives: List[str], 
                  maximize: Dict[str, bool]) -> bool:
        """
        检查当前解是否支配另一个解（帕累托支配关系）
        
        Args:
            other: 另一个优化结果
            objectives: 目标函数名称列表
            maximize: 每个目标是否最大化的字典
            
        Returns:
            True如果当前解支配另一个解
        """
        if self.objective_values is None or other.objective_values is None:
            return False
        
        at_least_one_better = False
        
        for obj in objectives:
            if obj not in self.objective_values or obj not in other.objective_values:
                continue
                
            self_val = self.objective_values[obj]
            other_val = other.objective_values[obj]
            
            if maximize.get(obj, True):
                if self_val < other_val:
                    return False  # 在某个目标上更差，不能支配
                elif self_val > other_val:
                    at_least_one_better = True
            else:
                if self_val > other_val:
                    return False  # 在某个目标上更差，不能支配
                elif self_val < other_val:
                    at_least_one_better = True
        
        return at_least_one_better
    
    def get_objective_vector(self, objectives: List[str]) -> List[float]:
        """获取目标函数向量"""
        if self.objective_values is None:
            return [self.objective_value]
        
        return [self.objective_values.get(obj, 0.0) for obj in objectives]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'parameters': self.parameters,
            'objective_value': self.objective_value,
            'metrics': self.metrics,
            'iteration': self.iteration,
            'timestamp': self.timestamp.isoformat(),
            'evaluation_time': self.evaluation_time,
            'fold_results': self.fold_results,
            'error_info': self.error_info,
            'objective_values': self.objective_values,
            'is_pareto_optimal': self.is_pareto_optimal,
            'dominance_count': self.dominance_count,
            'dominated_by': self.dominated_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """从字典创建OptimizationResult实例"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # 向后兼容性：为缺失的新MoCo参数添加默认值
        if 'parameters' in data and isinstance(data['parameters'], dict):
            parameters = data['parameters']
            
            # 添加新MoCo参数的默认值
            moco_defaults = {
                'moco_tau1': 0.2,      # DoubleTau MoCo正样本温度系数
                'moco_tau2': 0.3,      # DoubleTau MoCo负样本温度系数  
                'enable_view_0': 'true' # 是否启用MoCo第0视图
            }
            
            for param_name, default_value in moco_defaults.items():
                if param_name not in parameters:
                    parameters[param_name] = default_value
        
        return cls(**data)


@dataclass
class OptimizationHistory:
    """
    优化历史数据类
    
    记录整个优化过程的历史信息，支持多目标优化
    """
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
    objectives: List[str] = field(default_factory=lambda: ['primary'])  # 目标函数名称列表
    maximize_objectives: Dict[str, bool] = field(default_factory=lambda: {'primary': True})  # 每个目标是否最大化
    pareto_front: List[OptimizationResult] = field(default_factory=list)  # 帕累托前沿
    objective_weights: Optional[Dict[str, float]] = None  # 目标权重（用于加权求和）
    
    def add_result(self, result: OptimizationResult, maximize: bool = True):
        """添加新的优化结果"""
        self.results.append(result)
        self.total_iterations = len(self.results)
        
        # 更新最佳结果（单目标）
        if self.best_result is None or result.is_better_than(self.best_result, maximize):
            self.best_result = result
        
        # 更新帕累托前沿（多目标）
        if len(self.objectives) > 1:
            self._update_pareto_front(result)
    
    def _update_pareto_front(self, new_result: OptimizationResult):
        """更新帕累托前沿"""
        # 检查新结果是否被现有前沿中的解支配
        is_dominated = False
        dominated_solutions = []
        
        for front_result in self.pareto_front:
            if front_result.dominates(new_result, self.objectives, self.maximize_objectives):
                is_dominated = True
                break
            elif new_result.dominates(front_result, self.objectives, self.maximize_objectives):
                dominated_solutions.append(front_result)
        
        # 如果新结果不被支配，将其加入前沿
        if not is_dominated:
            # 移除被新结果支配的解
            for dominated in dominated_solutions:
                if dominated in self.pareto_front:
                    self.pareto_front.remove(dominated)
                    dominated.is_pareto_optimal = False
            
            # 添加新结果到前沿
            self.pareto_front.append(new_result)
            new_result.is_pareto_optimal = True
    
    def compute_pareto_front(self) -> List[OptimizationResult]:
        """重新计算完整的帕累托前沿"""
        if len(self.objectives) <= 1:
            return [self.best_result] if self.best_result else []
        
        pareto_front = []
        
        for i, result_i in enumerate(self.results):
            is_dominated = False
            
            for j, result_j in enumerate(self.results):
                if i != j and result_j.dominates(result_i, self.objectives, self.maximize_objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                result_i.is_pareto_optimal = True
                pareto_front.append(result_i)
            else:
                result_i.is_pareto_optimal = False
        
        self.pareto_front = pareto_front
        return pareto_front
    
    def get_weighted_objective_value(self, result: OptimizationResult) -> float:
        """计算加权目标函数值"""
        if self.objective_weights is None or result.objective_values is None:
            return result.objective_value
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for obj_name, weight in self.objective_weights.items():
            if obj_name in result.objective_values:
                obj_value = result.objective_values[obj_name]
                
                # 如果目标是最小化，取负值
                if not self.maximize_objectives.get(obj_name, True):
                    obj_value = -obj_value
                
                weighted_sum += weight * obj_value
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def set_objectives(self, objectives: List[str], maximize: Dict[str, bool], 
                      weights: Optional[Dict[str, float]] = None):
        """设置多目标优化配置"""
        self.objectives = objectives
        self.maximize_objectives = maximize
        self.objective_weights = weights
        
        # 重新计算帕累托前沿
        if len(objectives) > 1:
            self.compute_pareto_front()
    
    def get_pareto_front_metrics(self) -> Dict[str, Any]:
        """获取帕累托前沿的统计信息"""
        if not self.pareto_front:
            return {}
        
        metrics = {
            'front_size': len(self.pareto_front),
            'coverage_ratio': len(self.pareto_front) / len(self.results) if self.results else 0.0
        }
        
        # 计算每个目标的范围
        for obj in self.objectives:
            obj_values = []
            for result in self.pareto_front:
                if result.objective_values and obj in result.objective_values:
                    obj_values.append(result.objective_values[obj])
            
            if obj_values:
                metrics[f'{obj}_min'] = min(obj_values)
                metrics[f'{obj}_max'] = max(obj_values)
                metrics[f'{obj}_range'] = max(obj_values) - min(obj_values)
        
        return metrics
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """获取最佳参数组合"""
        return self.best_result.parameters if self.best_result else None
    
    def get_best_objective_value(self) -> Optional[float]:
        """获取最佳目标函数值"""
        return self.best_result.objective_value if self.best_result else None
    
    def get_convergence_curve(self) -> List[float]:
        """获取收敛曲线（历史最佳值序列）"""
        if not self.results:
            return []
        
        convergence = []
        current_best = self.results[0].objective_value
        
        for result in self.results:
            if result.objective_value > current_best:  # 假设最大化
                current_best = result.objective_value
            convergence.append(current_best)
        
        return convergence
    
    def get_parameter_history(self, param_name: str) -> List[Any]:
        """获取特定参数的历史值"""
        return [result.parameters.get(param_name) for result in self.results]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'results': [result.to_dict() for result in self.results],
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'parameter_space': self.parameter_space,
            'acquisition_function': self.acquisition_function,
            'task_type': self.task_type,
            'total_iterations': self.total_iterations,
            'total_time': self.total_time,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'objectives': self.objectives,
            'maximize_objectives': self.maximize_objectives,
            'pareto_front': [result.to_dict() for result in self.pareto_front],
            'objective_weights': self.objective_weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationHistory':
        """从字典创建OptimizationHistory实例"""
        data = data.copy()
        
        # 转换结果列表
        if 'results' in data:
            data['results'] = [OptimizationResult.from_dict(r) for r in data['results']]
        
        # 转换最佳结果
        if data.get('best_result'):
            data['best_result'] = OptimizationResult.from_dict(data['best_result'])
        
        # 转换帕累托前沿
        if 'pareto_front' in data:
            data['pareto_front'] = [OptimizationResult.from_dict(r) for r in data['pareto_front']]
        
        # 转换时间戳
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # 设置默认值
        if 'objectives' not in data:
            data['objectives'] = ['primary']
        if 'maximize_objectives' not in data:
            data['maximize_objectives'] = {'primary': True}
        if 'pareto_front' not in data:
            data['pareto_front'] = []
        
        # 向后兼容性：为历史数据添加新字段的默认值
        if 'parameter_space' not in data:
            data['parameter_space'] = {}
        if 'acquisition_function' not in data:
            data['acquisition_function'] = "EI"
        if 'task_type' not in data:
            data['task_type'] = "LDA"
        if 'total_iterations' not in data:
            data['total_iterations'] = len(data.get('results', []))
        if 'total_time' not in data:
            data['total_time'] = 0.0
        if 'objective_weights' not in data:
            data['objective_weights'] = None
        
        return cls(**data)


class ParameterSpace:
    """
    参数空间管理器
    
    管理所有可优化参数的定义、验证和采样
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterConfig] = {}
        self.constraints: List[str] = []
        self._rng = np.random.default_rng()
    
    def add_parameter(self, config: ParameterConfig):
        """添加参数配置"""
        if config.name in self.parameters:
            raise ValueError(f"参数 {config.name} 已存在")
        
        self.parameters[config.name] = config
    
    def add_continuous_parameter(self, name: str, low: float, high: float, 
                                log_scale: bool = False, constraints: Optional[List[str]] = None):
        """添加连续型参数"""
        config = ParameterConfig(
            name=name,
            param_type=ParameterType.CONTINUOUS,
            bounds=(low, high),
            log_scale=log_scale,
            constraints=constraints
        )
        self.add_parameter(config)
    
    def add_discrete_parameter(self, name: str, values: List[Union[int, float]], 
                              constraints: Optional[List[str]] = None):
        """添加离散型参数"""
        config = ParameterConfig(
            name=name,
            param_type=ParameterType.DISCRETE,
            values=values,
            constraints=constraints
        )
        self.add_parameter(config)
    
    def add_categorical_parameter(self, name: str, categories: List[str], 
                                 constraints: Optional[List[str]] = None):
        """添加分类型参数"""
        config = ParameterConfig(
            name=name,
            param_type=ParameterType.CATEGORICAL,
            values=categories,
            constraints=constraints
        )
        self.add_parameter(config)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数组合是否有效"""
        # 检查是否包含所有必需参数
        for param_name in self.parameters:
            if param_name not in parameters:
                return False
        
        # 检查每个参数值是否有效
        for param_name, value in parameters.items():
            if param_name not in self.parameters:
                return False
            
            if not self.parameters[param_name].validate_value(value):
                return False
        
        # 检查约束条件
        return self._check_constraints(parameters)
    
    def _check_constraints(self, parameters: Dict[str, Any]) -> bool:
        """检查参数约束条件"""
        # 模型结构约束：hidden层应该递减
        if all(key in parameters for key in ['dimensions', 'hidden1', 'hidden2']):
            dims = int(parameters['dimensions'])
            h1 = int(parameters['hidden1'])
            h2 = int(parameters['hidden2'])
            if not (dims >= h1 >= h2):
                return False
        
        # 解码器约束：解码器维度应该大于等于编码器最后一层
        if all(key in parameters for key in ['hidden2', 'decoder1']):
            h2 = int(parameters['hidden2'])
            d1 = int(parameters['decoder1'])
            if d1 < h2:
                return False
        
        # 注意力头数约束：应该能整除对应的隐藏维度
        attention_constraints = [
            ('hidden1', 'gat_heads'),
            ('hidden2', 'gt_heads'),
            ('hidden2', 'fusion_heads')
        ]
        
        for hidden_key, heads_key in attention_constraints:
            if all(key in parameters for key in [hidden_key, heads_key]):
                hidden_dim = int(parameters[hidden_key])
                heads = int(parameters[heads_key])
                if hidden_dim % heads != 0:
                    return False
        
        # 学习率和权重衰减的合理性约束
        if all(key in parameters for key in ['lr', 'weight_decay']):
            lr = float(parameters['lr'])
            wd = float(parameters['weight_decay'])
            if wd > lr * 10:
                return False
        
        # 损失权重约束：至少有一个权重大于0
        if all(key in parameters for key in ['alpha', 'beta', 'gamma']):
            alpha = float(parameters['alpha'])
            beta = float(parameters['beta'])
            gamma = float(parameters['gamma'])
            if not (alpha > 0 or beta > 0 or gamma > 0):
                return False
        
        # 批大小和MoCo队列大小的关系约束
        if all(key in parameters for key in ['batch', 'moco_K']):
            batch = int(parameters['batch'])
            moco_K = int(parameters['moco_K'])
            if moco_K < batch * 4:
                return False
        
        # MoCo特定约束
        # 1. DoubleTau温度约束：tau2应该大于等于tau1
        if all(key in parameters for key in ['moco_tau1', 'moco_tau2']):
            tau1 = float(parameters['moco_tau1'])
            tau2 = float(parameters['moco_tau2'])
            if tau2 < tau1:
                return False
        
        # 2. MoCo动量系数范围约束
        if 'moco_momentum' in parameters:
            momentum = float(parameters['moco_momentum'])
            if not (0.9 <= momentum <= 0.9999):
                return False
        
        # 3. 所有MoCo温度参数应该为正值
        temp_params = ['moco_t', 'moco_tau1', 'moco_tau2']
        for param in temp_params:
            if param in parameters:
                temp = float(parameters[param])
                if temp <= 0:
                    return False
        
        return True
    
    def sample_random_parameters(self, seed: Optional[int] = None, max_attempts: int = 100) -> Dict[str, Any]:
        """
        随机采样参数组合
        
        Args:
            seed: 随机种子
            max_attempts: 最大尝试次数，确保生成满足约束的参数
            
        Returns:
            满足约束条件的参数字典
            
        Raises:
            RuntimeError: 当无法在最大尝试次数内生成有效参数时
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        for attempt in range(max_attempts):
            parameters = {}
            
            # 首先采样基础参数
            for param_name, config in self.parameters.items():
                parameters[param_name] = config.sample_random_value(self._rng)
            
            # 应用约束修复
            parameters = self._apply_constraint_aware_sampling(parameters)
            
            # 检查是否满足约束
            if self.validate_parameters(parameters):
                return parameters
        
        # 如果无法生成有效参数，使用修复策略
        parameters = {}
        for param_name, config in self.parameters.items():
            parameters[param_name] = config.sample_random_value(self._rng)
        
        # 强制修复参数
        fixed_parameters = self.suggest_parameter_fix(parameters)
        
        if self.validate_parameters(fixed_parameters):
            return fixed_parameters
        else:
            raise RuntimeError(f"无法在{max_attempts}次尝试内生成满足约束的参数组合")
    
    def _apply_constraint_aware_sampling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用约束感知的采样策略
        
        在采样过程中考虑约束条件，生成更可能满足约束的参数
        """
        # 确保隐藏层维度递减
        if all(key in parameters for key in ['dimensions', 'hidden1', 'hidden2']):
            dimensions = int(parameters['dimensions'])
            
            # 重新采样hidden1，确保小于dimensions
            hidden1_config = self.parameters['hidden1']
            max_hidden1 = min(hidden1_config.bounds[1], dimensions)
            if max_hidden1 > hidden1_config.bounds[0]:
                parameters['hidden1'] = self._rng.uniform(hidden1_config.bounds[0], max_hidden1)
            
            # 重新采样hidden2，确保小于hidden1
            hidden1 = int(parameters['hidden1'])
            hidden2_config = self.parameters['hidden2']
            max_hidden2 = min(hidden2_config.bounds[1], hidden1)
            if max_hidden2 > hidden2_config.bounds[0]:
                parameters['hidden2'] = self._rng.uniform(hidden2_config.bounds[0], max_hidden2)
        
        # 确保注意力头数能整除隐藏维度
        attention_mappings = [
            ('hidden1', 'gat_heads'),
            ('hidden2', 'gt_heads'),
            ('hidden2', 'fusion_heads')
        ]
        
        for hidden_key, heads_key in attention_mappings:
            if all(key in parameters for key in [hidden_key, heads_key]):
                hidden_dim = int(parameters[hidden_key])
                heads_config = self.parameters[heads_key]
                
                # 找到能整除hidden_dim的头数
                valid_heads = [h for h in heads_config.values if hidden_dim % h == 0]
                if valid_heads:
                    parameters[heads_key] = self._rng.choice(valid_heads)
                else:
                    # 如果没有合适的头数，调整隐藏维度
                    preferred_heads = self._rng.choice(heads_config.values)
                    adjusted_dim = (hidden_dim // preferred_heads) * preferred_heads
                    if adjusted_dim >= self.parameters[hidden_key].bounds[0]:
                        parameters[hidden_key] = adjusted_dim
                        parameters[heads_key] = preferred_heads
        
        # 确保学习率和权重衰减的合理关系
        if all(key in parameters for key in ['lr', 'weight_decay']):
            lr = float(parameters['lr'])
            weight_decay = float(parameters['weight_decay'])
            
            # 如果权重衰减太大，重新采样一个较小的值
            if weight_decay > lr * 10:
                wd_config = self.parameters['weight_decay']
                max_wd = min(wd_config.bounds[1], lr * 5)
                if max_wd > wd_config.bounds[0]:
                    if wd_config.log_scale:
                        log_low = np.log(wd_config.bounds[0])
                        log_high = np.log(max_wd)
                        parameters['weight_decay'] = np.exp(self._rng.uniform(log_low, log_high))
                    else:
                        parameters['weight_decay'] = self._rng.uniform(wd_config.bounds[0], max_wd)
        
        # 确保至少有一个损失权重大于0.1
        if all(key in parameters for key in ['alpha', 'beta', 'gamma']):
            alpha = float(parameters['alpha'])
            beta = float(parameters['beta'])
            gamma = float(parameters['gamma'])
            
            if alpha < 0.1 and beta < 0.1 and gamma < 0.1:
                # 随机选择一个权重设为较大值
                weight_to_boost = self._rng.choice(['alpha', 'beta', 'gamma'])
                config = self.parameters[weight_to_boost]
                parameters[weight_to_boost] = self._rng.uniform(0.5, config.bounds[1])
        
        # 确保MoCo队列大小合理
        if all(key in parameters for key in ['batch', 'moco_K']):
            batch = int(parameters['batch'])
            moco_K = int(parameters['moco_K'])
            
            if moco_K < batch * 4:
                moco_config = self.parameters['moco_K']
                valid_moco_K = [k for k in moco_config.values if k >= batch * 4]
                if valid_moco_K:
                    parameters['moco_K'] = self._rng.choice(valid_moco_K)
        
        # 确保DoubleTau温度约束
        if all(key in parameters for key in ['moco_tau1', 'moco_tau2']):
            tau1 = float(parameters['moco_tau1'])
            tau2 = float(parameters['moco_tau2'])
            
            if tau2 < tau1:
                # 重新采样tau2，确保大于等于tau1
                tau2_config = self.parameters['moco_tau2']
                max_tau2 = tau2_config.bounds[1]
                min_tau2 = max(tau2_config.bounds[0], tau1)
                if max_tau2 > min_tau2:
                    parameters['moco_tau2'] = self._rng.uniform(min_tau2, max_tau2)
                else:
                    # 如果无法满足约束，调整tau1
                    parameters['moco_tau1'] = tau2_config.bounds[0]
                    parameters['moco_tau2'] = tau2_config.bounds[0] + 0.01
        
        # 确保MoCo动量系数在合理范围内
        if 'moco_momentum' in parameters:
            momentum = float(parameters['moco_momentum'])
            momentum_config = self.parameters['moco_momentum']
            
            # 确保在0.9-0.9999范围内
            if momentum < 0.9:
                parameters['moco_momentum'] = 0.9
            elif momentum > 0.9999:
                parameters['moco_momentum'] = 0.9999
        
        return parameters
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """获取所有连续型参数的边界（用于优化算法）"""
        bounds = []
        for config in self.parameters.values():
            if config.param_type == ParameterType.CONTINUOUS:
                if config.log_scale:
                    bounds.append((np.log(config.bounds[0]), np.log(config.bounds[1])))
                else:
                    bounds.append(config.bounds)
        return bounds
    
    def get_parameter_names(self) -> List[str]:
        """获取所有参数名称"""
        return list(self.parameters.keys())
    
    def get_continuous_parameter_names(self) -> List[str]:
        """获取连续型参数名称"""
        return [name for name, config in self.parameters.items() 
                if config.param_type == ParameterType.CONTINUOUS]
    
    def get_discrete_parameter_names(self) -> List[str]:
        """获取离散型参数名称"""
        return [name for name, config in self.parameters.items() 
                if config.param_type == ParameterType.DISCRETE]
    
    def get_categorical_parameter_names(self) -> List[str]:
        """获取分类型参数名称"""
        return [name for name, config in self.parameters.items() 
                if config.param_type == ParameterType.CATEGORICAL]
    
    def validate_parameters_detailed(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        详细验证参数组合
        
        Returns:
            (is_valid, error_messages): 验证结果和错误信息列表
        """
        errors = []
        
        # 检查是否包含所有必需参数
        for param_name in self.parameters:
            if param_name not in parameters:
                errors.append(f"缺少必需参数: {param_name}")
        
        # 检查每个参数值是否有效
        for param_name, value in parameters.items():
            if param_name not in self.parameters:
                errors.append(f"未知参数: {param_name}")
                continue
            
            if not self.parameters[param_name].validate_value(value):
                config = self.parameters[param_name]
                if config.param_type == ParameterType.CONTINUOUS:
                    errors.append(f"参数 {param_name} 值 {value} 超出范围 {config.bounds}")
                else:
                    errors.append(f"参数 {param_name} 值 {value} 不在允许的值列表中: {config.values}")
        
        # 检查约束条件
        constraint_errors = self._check_constraints_detailed(parameters)
        errors.extend(constraint_errors)
        
        return len(errors) == 0, errors
    
    def _check_constraints_detailed(self, parameters: Dict[str, Any]) -> List[str]:
        """详细检查参数约束条件，返回错误信息列表"""
        errors = []
        
        # 模型结构约束：hidden层应该递减
        if all(key in parameters for key in ['dimensions', 'hidden1', 'hidden2']):
            dims = int(parameters['dimensions'])
            h1 = int(parameters['hidden1'])
            h2 = int(parameters['hidden2'])
            if not (dims >= h1 >= h2):
                errors.append(f"隐藏层维度应该递减: dimensions({dims}) >= hidden1({h1}) >= hidden2({h2})")
        
        # 解码器约束：解码器维度应该大于等于编码器最后一层
        if all(key in parameters for key in ['hidden2', 'decoder1']):
            h2 = int(parameters['hidden2'])
            d1 = int(parameters['decoder1'])
            if d1 < h2:
                errors.append(f"解码器维度({d1})应该大于等于hidden2({h2})")
        
        # 注意力头数约束：应该能整除对应的隐藏维度
        attention_constraints = [
            ('hidden1', 'gat_heads', 'GAT'),
            ('hidden2', 'gt_heads', 'GT'),
            ('hidden2', 'fusion_heads', 'Fusion')
        ]
        
        for hidden_key, heads_key, layer_name in attention_constraints:
            if all(key in parameters for key in [hidden_key, heads_key]):
                hidden_dim = int(parameters[hidden_key])
                heads = int(parameters[heads_key])
                if hidden_dim % heads != 0:
                    errors.append(f"{layer_name}层隐藏维度({hidden_dim})必须能被头数({heads})整除")
        
        # 学习率和权重衰减的合理性约束
        if all(key in parameters for key in ['lr', 'weight_decay']):
            lr = float(parameters['lr'])
            wd = float(parameters['weight_decay'])
            if wd > lr * 10:
                errors.append(f"权重衰减({wd:.6f})不应该比学习率({lr:.6f})大太多")
        
        # 损失权重约束：至少有一个权重大于0
        if all(key in parameters for key in ['alpha', 'beta', 'gamma']):
            alpha = float(parameters['alpha'])
            beta = float(parameters['beta'])
            gamma = float(parameters['gamma'])
            if not (alpha > 0 or beta > 0 or gamma > 0):
                errors.append(f"至少需要一个损失权重大于0: alpha={alpha}, beta={beta}, gamma={gamma}")
        
        # 批大小和MoCo队列大小的关系约束
        if all(key in parameters for key in ['batch', 'moco_K']):
            batch = int(parameters['batch'])
            moco_K = int(parameters['moco_K'])
            if moco_K < batch * 4:
                errors.append(f"MoCo队列大小({moco_K})应该至少是批大小({batch})的4倍")
        
        # MoCo特定约束检查
        # 1. DoubleTau温度约束：tau2应该大于等于tau1
        if all(key in parameters for key in ['moco_tau1', 'moco_tau2']):
            tau1 = float(parameters['moco_tau1'])
            tau2 = float(parameters['moco_tau2'])
            if tau2 < tau1:
                errors.append(f"DoubleTau MoCo中tau2({tau2})应该大于等于tau1({tau1})")
        
        # 2. MoCo动量系数范围约束
        if 'moco_momentum' in parameters:
            momentum = float(parameters['moco_momentum'])
            if not (0.9 <= momentum <= 0.9999):
                errors.append(f"MoCo动量系数({momentum})应该在0.9-0.9999范围内")
        
        # 3. 所有MoCo温度参数应该为正值
        temp_params = ['moco_t', 'moco_tau1', 'moco_tau2']
        for param in temp_params:
            if param in parameters:
                temp = float(parameters[param])
                if temp <= 0:
                    errors.append(f"MoCo温度参数{param}({temp})必须为正值")
        
        return errors
    
    def suggest_parameter_fix(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        建议参数修复
        
        尝试自动修复违反约束的参数
        """
        fixed_params = parameters.copy()
        
        # 首先修复基本参数空间问题
        for param_name, value in fixed_params.items():
            if param_name in self.parameters:
                config = self.parameters[param_name]
                
                # 修复连续型参数
                if config.param_type == ParameterType.CONTINUOUS:
                    val = float(value)
                    if val < config.bounds[0]:
                        fixed_params[param_name] = config.bounds[0]
                    elif val > config.bounds[1]:
                        fixed_params[param_name] = config.bounds[1]
                    else:
                        fixed_params[param_name] = val
                
                # 修复离散型参数
                elif config.param_type == ParameterType.DISCRETE:
                    if value not in config.values:
                        # 选择最接近的有效值
                        if isinstance(value, (int, float)):
                            closest = min(config.values, key=lambda x: abs(x - value))
                            fixed_params[param_name] = closest
                        else:
                            fixed_params[param_name] = config.values[0]
                
                # 修复分类型参数
                elif config.param_type == ParameterType.CATEGORICAL:
                    if value not in config.values:
                        fixed_params[param_name] = config.values[0]
        
        # 修复隐藏层递减约束
        if all(key in fixed_params for key in ['dimensions', 'hidden1', 'hidden2']):
            dimensions = int(fixed_params['dimensions'])
            hidden1 = int(fixed_params['hidden1'])
            hidden2 = int(fixed_params['hidden2'])
            
            if not (dimensions >= hidden1 >= hidden2):
                # 重新调整隐藏层大小，保持递减关系
                fixed_params['hidden1'] = min(hidden1, dimensions)
                fixed_params['hidden2'] = min(hidden2, int(fixed_params['hidden1']))
        
        # 修复解码器约束
        if all(key in fixed_params for key in ['hidden2', 'decoder1']):
            hidden2 = int(fixed_params['hidden2'])
            decoder1 = int(fixed_params['decoder1'])
            if decoder1 < hidden2:
                fixed_params['decoder1'] = max(decoder1, hidden2)
        
        # 修复注意力头数约束
        for hidden_key, heads_key in [('hidden1', 'gat_heads'), ('hidden2', 'gt_heads'), ('hidden2', 'fusion_heads')]:
            if all(key in fixed_params for key in [hidden_key, heads_key]):
                hidden_dim = int(fixed_params[hidden_key])
                heads = int(fixed_params[heads_key])
                
                # 找到最接近的可整除的头数
                possible_heads = [h for h in [2, 4, 8, 16] if hidden_dim % h == 0]
                if possible_heads:
                    # 选择最接近原始值的头数
                    fixed_params[heads_key] = min(possible_heads, key=lambda x: abs(x - heads))
                else:
                    # 如果没有可整除的头数，调整隐藏层维度
                    # 找到最接近的可被常用头数整除的维度
                    for target_heads in [2, 4, 8]:
                        if hidden_dim >= target_heads:
                            adjusted_dim = (hidden_dim // target_heads) * target_heads
                            if adjusted_dim > 0:
                                fixed_params[hidden_key] = adjusted_dim
                                fixed_params[heads_key] = target_heads
                                break
        
        # 修复学习率约束
        if all(key in fixed_params for key in ['lr', 'weight_decay']):
            lr = float(fixed_params['lr'])
            weight_decay = float(fixed_params['weight_decay'])
            if weight_decay > lr * 10:
                fixed_params['weight_decay'] = lr * 5
        
        # 修复损失权重约束
        if all(key in fixed_params for key in ['alpha', 'beta', 'gamma']):
            alpha = float(fixed_params['alpha'])
            beta = float(fixed_params['beta'])
            gamma = float(fixed_params['gamma'])
            if alpha <= 0 and beta <= 0 and gamma <= 0:
                # 至少设置一个权重为正值
                fixed_params['alpha'] = 1.0
        
        # 修复批大小和MoCo队列约束
        if all(key in fixed_params for key in ['batch', 'moco_K']):
            batch = int(fixed_params['batch'])
            moco_K = int(fixed_params['moco_K'])
            if moco_K < batch * 4:
                fixed_params['moco_K'] = batch * 4
        
        # MoCo特定参数修复
        # 1. 修复DoubleTau温度约束
        if all(key in fixed_params for key in ['moco_tau1', 'moco_tau2']):
            tau1 = float(fixed_params['moco_tau1'])
            tau2 = float(fixed_params['moco_tau2'])
            if tau2 < tau1:
                # 调整tau2使其大于等于tau1
                fixed_params['moco_tau2'] = max(tau2, tau1 + 0.01)
        
        # 2. 修复MoCo动量系数范围
        if 'moco_momentum' in fixed_params:
            momentum = float(fixed_params['moco_momentum'])
            if momentum < 0.9:
                fixed_params['moco_momentum'] = 0.9
            elif momentum > 0.9999:
                fixed_params['moco_momentum'] = 0.9999
        
        # 3. 修复MoCo温度参数为正值
        temp_params = ['moco_t', 'moco_tau1', 'moco_tau2']
        for param in temp_params:
            if param in fixed_params:
                temp = float(fixed_params[param])
                if temp <= 0:
                    fixed_params[param] = 0.01  # 设置为最小正值
        
        return fixed_params
    
    def add_constraint(self, constraint_name: str, constraint_description: str = ""):
        """添加约束描述"""
        if constraint_name not in self.constraints:
            self.constraints.append(constraint_name)
    
    def remove_parameter(self, name: str):
        """移除参数"""
        if name in self.parameters:
            del self.parameters[name]
    
    def clear_parameters(self):
        """清空所有参数"""
        self.parameters.clear()
        self.constraints.clear()
    
    def get_parameter_count(self) -> int:
        """获取参数总数"""
        return len(self.parameters)
    
    def get_parameter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取特定参数的详细信息"""
        if name in self.parameters:
            return self.parameters[name].to_dict()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'parameters': {name: config.to_dict() for name, config in self.parameters.items()},
            'constraints': self.constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSpace':
        """从字典创建ParameterSpace实例"""
        space = cls()
        
        if 'parameters' in data:
            for name, config_data in data['parameters'].items():
                config = ParameterConfig.from_dict(config_data)
                space.add_parameter(config)
        
        if 'constraints' in data:
            space.constraints = data['constraints']
        
        return space


def create_default_parameter_space(task_type: str = "LDA") -> ParameterSpace:
    """
    创建默认的参数空间配置
    
    根据现有的parms_setting.py中的参数定义创建标准的参数空间
    """
    space = ParameterSpace()
    
    # 连续型参数
    space.add_continuous_parameter('dimensions', 128, 512)
    space.add_continuous_parameter('hidden1', 64, 256)
    space.add_continuous_parameter('hidden2', 32, 128)
    space.add_continuous_parameter('decoder1', 256, 1024)
    space.add_continuous_parameter('lr', 1e-5, 1e-2, log_scale=True)
    space.add_continuous_parameter('dropout', 0.0, 0.5)
    space.add_continuous_parameter('weight_decay', 1e-6, 1e-2, log_scale=True)
    space.add_continuous_parameter('alpha', 0.1, 2.0)
    space.add_continuous_parameter('beta', 0.1, 2.0)
    space.add_continuous_parameter('gamma', 0.1, 2.0)
    
    # MoCo连续型参数
    space.add_continuous_parameter('moco_momentum', 0.9, 0.9999)
    space.add_continuous_parameter('moco_t', 0.01, 1.0)
    space.add_continuous_parameter('moco_tau1', 0.01, 1.0)
    space.add_continuous_parameter('moco_tau2', 0.01, 1.0)
    
    # 离散型参数
    space.add_discrete_parameter('gat_heads', [2, 4, 8, 16])
    space.add_discrete_parameter('gt_heads', [2, 4, 8, 16])
    space.add_discrete_parameter('fusion_heads', [2, 4, 8, 16])
    space.add_discrete_parameter('batch', [16, 25, 32, 64])
    space.add_discrete_parameter('moco_K', [1024, 2048, 4096, 8192])
    
    # 分类型参数
    space.add_categorical_parameter('fusion_strategy', 
                                   ['self_attention', 'co_attention', 'hybrid', 'transformer_multihead'])
    space.add_categorical_parameter('feature_type', ['normal', 'uniform', 'one_hot'])
    space.add_categorical_parameter('moco_type', ['basic', 'double_tau'])
    space.add_categorical_parameter('enable_view_0', ['true', 'false'])
    
    return space


if __name__ == "__main__":
    # 测试代码
    print("测试核心数据结构...")
    
    # 测试参数空间
    space = create_default_parameter_space()
    print(f"创建参数空间，包含 {len(space.parameters)} 个参数")
    
    # 测试随机采样
    params = space.sample_random_parameters(seed=42)
    print(f"随机采样参数: {params}")
    
    # 测试参数验证
    is_valid = space.validate_parameters(params)
    print(f"参数验证结果: {is_valid}")
    
    # 测试详细验证
    is_valid_detailed, errors = space.validate_parameters_detailed(params)
    print(f"详细验证结果: {is_valid_detailed}")
    if errors:
        print(f"验证错误: {errors}")
    
    # 测试无效参数和修复功能
    invalid_params = params.copy()
    invalid_params['hidden1'] = 600  # 违反递减约束
    invalid_params['gat_heads'] = 3   # 违反整除约束
    invalid_params['alpha'] = 0       # 违反权重约束
    invalid_params['beta'] = 0
    invalid_params['gamma'] = 0
    
    print(f"\n测试无效参数...")
    is_valid_invalid, invalid_errors = space.validate_parameters_detailed(invalid_params)
    print(f"无效参数验证结果: {is_valid_invalid}")
    print(f"错误信息: {invalid_errors}")
    
    # 测试参数修复
    fixed_params = space.suggest_parameter_fix(invalid_params)
    print(f"\n修复后参数:")
    print(f"  hidden1: {invalid_params['hidden1']} -> {fixed_params['hidden1']}")
    print(f"  gat_heads: {invalid_params['gat_heads']} -> {fixed_params['gat_heads']}")
    print(f"  alpha: {invalid_params['alpha']} -> {fixed_params['alpha']}")
    
    # 验证修复后的参数
    is_fixed_valid, fixed_errors = space.validate_parameters_detailed(fixed_params)
    print(f"修复后验证结果: {is_fixed_valid}")
    if fixed_errors:
        print(f"修复后仍有错误: {fixed_errors}")
    
    # 测试参数空间信息查询
    print(f"\n参数空间信息:")
    print(f"  总参数数: {space.get_parameter_count()}")
    print(f"  连续型参数: {space.get_continuous_parameter_names()}")
    print(f"  离散型参数: {space.get_discrete_parameter_names()}")
    print(f"  分类型参数: {space.get_categorical_parameter_names()}")
    
    # 测试优化结果
    result = OptimizationResult(
        parameters=params,
        objective_value=0.85,
        metrics={'AUROC': 0.85, 'AUPRC': 0.78, 'F1': 0.72},
        iteration=1,
        timestamp=datetime.now(),
        evaluation_time=120.5
    )
    print(f"\n创建优化结果: iteration={result.iteration}, objective={result.objective_value}")
    
    # 测试优化历史
    history = OptimizationHistory()
    history.add_result(result)
    print(f"优化历史: {history.total_iterations} 次迭代, 最佳值={history.get_best_objective_value()}")
    
    print("\n核心数据结构测试完成!")