"""
贝叶斯优化器（BayesianOptimizer）

主要的优化协调器类，实现完整的贝叶斯优化流程：
- 建议参数 -> 评估 -> 更新模型的优化循环
- 收敛检测和停止条件
- 错误处理和恢复机制
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
import warnings

# 导入核心组件
from autodl_core import ParameterSpace, OptimizationHistory, OptimizationResult
from gaussian_process import GaussianProcess, create_default_gaussian_process
from acquisition_function import AcquisitionFunction, create_acquisition_function, AcquisitionOptimizer
from task_evaluator import TaskEvaluator, create_task_evaluator
from state_manager import StateManager, create_default_state_manager, CheckpointError


class ConvergenceError(Exception):
    """收敛相关错误"""
    pass


class OptimizationError(Exception):
    """优化过程错误"""
    pass


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    协调整个优化过程，管理高斯过程模型、采集函数和任务评估器
    """
    
    def __init__(self, 
                 parameter_space: ParameterSpace,
                 task_evaluator: TaskEvaluator,
                 acquisition_function: Optional[AcquisitionFunction] = None,
                 gaussian_process: Optional[GaussianProcess] = None,
                 state_manager: Optional[StateManager] = None,
                 n_initial_points: int = 10,
                 random_state: Optional[int] = None,
                 maximize: bool = True,
                 # 多目标优化参数
                 objectives: Optional[List[str]] = None,
                 maximize_objectives: Optional[Dict[str, bool]] = None,
                 objective_weights: Optional[Dict[str, float]] = None):
        """
        初始化贝叶斯优化器
        
        Args:
            parameter_space: 参数空间管理器
            task_evaluator: 任务评估器
            acquisition_function: 采集函数，默认使用EI
            gaussian_process: 高斯过程模型，默认创建标准配置
            state_manager: 状态管理器，默认创建标准配置
            n_initial_points: 初始随机采样点数
            random_state: 随机种子
            maximize: 是否最大化目标函数，默认True（单目标时使用）
            objectives: 多目标优化的目标函数名称列表
            maximize_objectives: 每个目标是否最大化的字典
            objective_weights: 目标权重字典（用于加权求和）
        """
        self.parameter_space = parameter_space
        self.task_evaluator = task_evaluator
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.maximize = maximize
        
        # 多目标优化配置
        self.objectives = objectives or ['primary']
        self.maximize_objectives = maximize_objectives or {'primary': maximize}
        self.objective_weights = objective_weights
        self.is_multi_objective = len(self.objectives) > 1
        
        # 创建或使用提供的组件
        self.acquisition_function = acquisition_function or create_acquisition_function('EI', xi=0.01)
        self.gaussian_process = gaussian_process or create_default_gaussian_process(random_state)
        self.state_manager = state_manager or create_default_state_manager()
        
        # 创建采集函数优化器
        self.acquisition_optimizer = AcquisitionOptimizer(method='L-BFGS-B', n_restarts=10)
        
        # 初始化优化历史
        self.history = OptimizationHistory(
            parameter_space=parameter_space.to_dict(),
            acquisition_function=self.acquisition_function.function_type,
            task_type=task_evaluator.task_type,
            objectives=self.objectives,
            maximize_objectives=self.maximize_objectives,
            objective_weights=self.objective_weights
        )
        
        # 优化状态
        self.is_initialized = False
        self.current_iteration = 0
        self.start_time: Optional[datetime] = None
        self.last_checkpoint_time: Optional[datetime] = None
        
        # 收敛检测参数
        self.convergence_window = 10  # 收敛检测窗口大小
        self.convergence_threshold = 1e-4  # 收敛阈值
        self.patience = 20  # 早停耐心值
        self.no_improvement_count = 0  # 无改进计数
        
        # 错误处理参数
        self.max_consecutive_failures = 5  # 最大连续失败次数
        self.consecutive_failures = 0  # 当前连续失败次数
        self.failed_parameters: List[Dict[str, Any]] = []  # 失败的参数记录
        
        # 设置日志
        self.logger = logging.getLogger(f"BayesianOptimizer_{task_evaluator.task_type}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"BayesianOptimizer初始化完成，任务类型: {task_evaluator.task_type}")
        self.logger.info(f"参数空间包含 {parameter_space.get_parameter_count()} 个参数")
        self.logger.info(f"采集函数: {self.acquisition_function.function_type}")
        self.logger.info(f"初始采样点数: {n_initial_points}")
    
    def optimize(self, 
                 n_iterations: int = 100,
                 checkpoint_freq: int = 10,
                 time_limit: Optional[float] = None,
                 target_value: Optional[float] = None,
                 resume_from_checkpoint: Optional[str] = None) -> OptimizationHistory:
        """
        执行贝叶斯优化
        
        Args:
            n_iterations: 最大迭代次数
            checkpoint_freq: 检查点保存频率
            time_limit: 时间限制（秒），None表示无限制
            target_value: 目标值，达到后停止优化
            resume_from_checkpoint: 从检查点恢复的路径
            
        Returns:
            优化历史记录
            
        Raises:
            OptimizationError: 当优化过程出现严重错误时
        """
        try:
            # 恢复或初始化优化状态
            if resume_from_checkpoint:
                self._resume_from_checkpoint(resume_from_checkpoint)
            else:
                self._initialize_optimization()
            
            self.logger.info(f"开始贝叶斯优化，最大迭代次数: {n_iterations}")
            if time_limit:
                self.logger.info(f"时间限制: {time_limit:.1f} 秒")
            if target_value:
                self.logger.info(f"目标值: {target_value}")
            
            # 主优化循环
            while self.current_iteration < n_iterations:
                iteration_start_time = time.time()
                
                try:
                    # 检查停止条件
                    if self._should_stop(time_limit, target_value):
                        break
                    
                    # 执行一次优化迭代
                    self._run_single_iteration()
                    
                    # 更新迭代计数
                    self.current_iteration += 1
                    
                    # 记录迭代时间
                    iteration_time = time.time() - iteration_start_time
                    self.history.total_time += iteration_time
                    
                    # 检查收敛
                    if self._check_convergence():
                        self.logger.info(f"优化在第 {self.current_iteration} 次迭代后收敛")
                        break
                    
                    # 保存检查点
                    if checkpoint_freq > 0 and self.current_iteration % checkpoint_freq == 0:
                        self._save_checkpoint()
                    
                    # 重置连续失败计数
                    self.consecutive_failures = 0
                    
                except Exception as e:
                    self._handle_iteration_error(e)
                    
                    # 检查是否应该终止
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        error_msg = f"连续失败 {self.consecutive_failures} 次，终止优化"
                        self.logger.error(error_msg)
                        raise OptimizationError(error_msg)
            
            # 优化完成
            self._finalize_optimization()
            
            self.logger.info(f"优化完成，总迭代次数: {self.current_iteration}")
            self.logger.info(f"最佳目标值: {self.history.get_best_objective_value():.6f}")
            self.logger.info(f"总耗时: {self.history.total_time:.1f} 秒")
            
            return self.history
            
        except Exception as e:
            self.logger.error(f"优化过程出现严重错误: {str(e)}")
            # 尝试保存当前状态
            try:
                self._save_emergency_checkpoint()
            except:
                pass
            raise OptimizationError(f"优化失败: {str(e)}") from e
    
    def suggest_next_parameters(self) -> Dict[str, Any]:
        """
        建议下一个要评估的参数组合
        
        Returns:
            参数字典
            
        Raises:
            RuntimeError: 当无法生成有效参数建议时
        """
        if not self.is_initialized:
            raise RuntimeError("优化器尚未初始化，请先调用optimize()或_initialize_optimization()")
        
        try:
            # 如果还在初始采样阶段
            if len(self.history.results) < self.n_initial_points:
                return self._suggest_initial_parameters()
            
            # 使用采集函数建议参数
            return self._suggest_acquisition_parameters()
            
        except Exception as e:
            self.logger.error(f"参数建议失败: {str(e)}")
            # 回退到随机采样
            return self._suggest_fallback_parameters()
    
    def update_model(self, parameters: Dict[str, Any], objective_value: float, 
                    metrics: Dict[str, float], evaluation_time: float = 0.0,
                    objective_values: Optional[Dict[str, float]] = None):
        """
        使用新的评估结果更新模型
        
        Args:
            parameters: 参数组合
            objective_value: 主要目标函数值
            metrics: 详细指标
            evaluation_time: 评估耗时
            objective_values: 多目标函数值字典
        """
        try:
            # 如果是多目标优化但没有提供多目标值，从metrics中提取
            if self.is_multi_objective and objective_values is None:
                objective_values = self._extract_objective_values(metrics)
            
            # 创建优化结果
            result = OptimizationResult(
                parameters=parameters,
                objective_value=objective_value,
                metrics=metrics,
                iteration=self.current_iteration + 1,
                timestamp=datetime.now(),
                evaluation_time=evaluation_time,
                objective_values=objective_values
            )
            
            # 添加到历史记录
            self.history.add_result(result, maximize=self.maximize)
            
            # 更新高斯过程模型
            if self.is_multi_objective and self.objective_weights:
                # 多目标：使用加权目标值更新模型
                weighted_value = self.history.get_weighted_objective_value(result)
                self._update_gaussian_process(parameters, weighted_value)
            else:
                # 单目标：使用主要目标值更新模型
                self._update_gaussian_process(parameters, objective_value)
            
            # 更新无改进计数
            self._update_improvement_tracking(result)
            
            if self.is_multi_objective:
                pareto_size = len(self.history.pareto_front)
                self.logger.info(f"模型更新完成，帕累托前沿大小: {pareto_size}")
            else:
                self.logger.info(f"模型更新完成，当前最佳值: {self.history.get_best_objective_value():.6f}")
            
        except Exception as e:
            self.logger.error(f"模型更新失败: {str(e)}")
            raise
    
    def _extract_objective_values(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """从metrics中提取多目标函数值"""
        objective_values = {}
        
        for obj_name in self.objectives:
            if obj_name in metrics:
                objective_values[obj_name] = metrics[obj_name]
            elif obj_name == 'primary':
                # 默认使用AUROC作为主要目标
                objective_values[obj_name] = metrics.get('AUROC', 0.0)
            else:
                # 尝试映射常见的目标名称
                mapping = {
                    'auroc': 'AUROC',
                    'auprc': 'AUPRC', 
                    'f1': 'F1',
                    'precision': 'precision',
                    'recall': 'recall'
                }
                mapped_name = mapping.get(obj_name.lower(), obj_name)
                objective_values[obj_name] = metrics.get(mapped_name, 0.0)
        
        return objective_values
    
    def get_best_parameters(self) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """
        获取当前最佳参数组合和目标值
        
        Returns:
            (best_parameters, best_objective_value): 最佳参数和目标值
        """
        if self.history.best_result:
            return self.history.best_result.parameters, self.history.best_result.objective_value
        else:
            return None, None
    
    def get_pareto_front(self) -> List[OptimizationResult]:
        """
        获取帕累托前沿（多目标优化）
        
        Returns:
            帕累托最优解列表
        """
        if not self.is_multi_objective:
            return [self.history.best_result] if self.history.best_result else []
        
        return self.history.pareto_front.copy()
    
    def set_objective_weights(self, weights: Dict[str, float]):
        """
        设置目标权重（用于加权求和）
        
        Args:
            weights: 目标权重字典，键为目标名称，值为权重
        """
        if not self.is_multi_objective:
            self.logger.warning("单目标优化不需要设置权重")
            return
        
        # 验证权重
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("权重总和必须大于0")
        
        # 归一化权重
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        self.objective_weights = normalized_weights
        self.history.objective_weights = normalized_weights
        
        self.logger.info(f"目标权重已更新: {normalized_weights}")
    
    def compute_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """
        计算帕累托前沿的超体积指标
        
        Args:
            reference_point: 参考点，如果未提供则自动计算
            
        Returns:
            超体积值
        """
        if not self.is_multi_objective or not self.history.pareto_front:
            return 0.0
        
        try:
            import numpy as np
            
            # 获取帕累托前沿的目标值矩阵
            front_matrix = []
            for result in self.history.pareto_front:
                obj_vector = result.get_objective_vector(self.objectives)
                # 对于最小化目标，取负值
                for i, obj_name in enumerate(self.objectives):
                    if not self.maximize_objectives.get(obj_name, True):
                        obj_vector[i] = -obj_vector[i]
                front_matrix.append(obj_vector)
            
            front_matrix = np.array(front_matrix)
            
            # 计算参考点
            if reference_point is None:
                ref_point = []
                for i, obj_name in enumerate(self.objectives):
                    min_val = np.min(front_matrix[:, i])
                    ref_point.append(min_val - 0.1 * abs(min_val))
                reference_point = ref_point
            else:
                ref_point = [reference_point.get(obj, 0.0) for obj in self.objectives]
            
            # 简化的超体积计算（适用于2-3个目标）
            if len(self.objectives) == 2:
                return self._compute_hypervolume_2d(front_matrix, ref_point)
            elif len(self.objectives) == 3:
                return self._compute_hypervolume_3d(front_matrix, ref_point)
            else:
                self.logger.warning("超体积计算仅支持2-3个目标")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"超体积计算失败: {e}")
            return 0.0
    
    def _compute_hypervolume_2d(self, front: np.ndarray, ref_point: List[float]) -> float:
        """计算2D超体积"""
        if len(front) == 0:
            return 0.0
        
        # 按第一个目标排序
        sorted_front = front[np.argsort(front[:, 0])]
        
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for point in sorted_front:
            x, y = point[0], point[1]
            if x > prev_x and y > ref_point[1]:
                hypervolume += (x - prev_x) * (y - ref_point[1])
                prev_x = x
        
        return hypervolume
    
    def _compute_hypervolume_3d(self, front: np.ndarray, ref_point: List[float]) -> float:
        """计算3D超体积（简化版本）"""
        if len(front) == 0:
            return 0.0
        
        # 简化计算：使用包围盒近似
        max_vals = np.max(front, axis=0)
        volume = 1.0
        
        for i in range(3):
            if max_vals[i] > ref_point[i]:
                volume *= (max_vals[i] - ref_point[i])
            else:
                return 0.0
        
        return volume * len(front) / 10.0  # 近似修正因子
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        获取优化状态信息
        
        Returns:
            包含优化状态的字典
        """
        status = {
            'is_initialized': self.is_initialized,
            'current_iteration': self.current_iteration,
            'total_evaluations': len(self.history.results),
            'best_objective_value': self.history.get_best_objective_value(),
            'total_time': self.history.total_time,
            'consecutive_failures': self.consecutive_failures,
            'no_improvement_count': self.no_improvement_count,
            'convergence_detected': self._check_convergence() if len(self.history.results) > self.convergence_window else False,
            'is_multi_objective': self.is_multi_objective,
            'objectives': self.objectives
        }
        
        if self.start_time:
            status['elapsed_time'] = (datetime.now() - self.start_time).total_seconds()
        
        if self.history.best_result:
            status['best_parameters'] = self.history.best_result.parameters
            status['best_metrics'] = self.history.best_result.metrics
        
        # 多目标优化状态
        if self.is_multi_objective:
            status['pareto_front_size'] = len(self.history.pareto_front)
            status['objective_weights'] = self.objective_weights
            status['pareto_metrics'] = self.history.get_pareto_front_metrics()
            
            if self.history.pareto_front:
                # 添加帕累托前沿的代表性解
                status['pareto_solutions'] = []
                for result in self.history.pareto_front[:5]:  # 最多显示5个解
                    status['pareto_solutions'].append({
                        'parameters': result.parameters,
                        'objective_values': result.objective_values,
                        'iteration': result.iteration
                    })
        
        return status
    
    def _initialize_optimization(self):
        """初始化优化过程"""
        self.logger.info("初始化贝叶斯优化器...")
        
        # 设置随机种子
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 重置状态
        self.current_iteration = 0
        self.start_time = datetime.now()
        self.history.start_time = self.start_time
        self.consecutive_failures = 0
        self.no_improvement_count = 0
        self.failed_parameters.clear()
        
        # 标记为已初始化
        self.is_initialized = True
        
        self.logger.info("优化器初始化完成")
    
    def _run_single_iteration(self):
        """执行单次优化迭代"""
        iteration_start = time.time()
        
        # 建议下一个参数组合
        parameters = self.suggest_next_parameters()
        
        self.logger.info(f"第 {self.current_iteration + 1} 次迭代，评估参数: {parameters}")
        
        # 评估参数
        evaluation_start = time.time()
        metrics = self.task_evaluator.evaluate_parameters(parameters)
        evaluation_time = time.time() - evaluation_start
        
        # 提取目标值
        objective_value = self.task_evaluator.get_objective_value(metrics)
        
        # 检查评估是否成功
        if 'error' in metrics:
            self.logger.warning(f"参数评估失败: {metrics['error']}")
            self.failed_parameters.append(parameters)
            raise RuntimeError(f"参数评估失败: {metrics['error']}")
        
        # 提取多目标值（如果适用）
        objective_values = None
        if self.is_multi_objective:
            objective_values = self._extract_objective_values(metrics)
        
        # 更新模型
        self.update_model(parameters, objective_value, metrics, evaluation_time, objective_values)
        
        iteration_time = time.time() - iteration_start
        
        if self.is_multi_objective:
            pareto_size = len(self.history.pareto_front)
            self.logger.info(f"第 {self.current_iteration + 1} 次迭代完成，"
                            f"目标值: {objective_value:.6f}, 帕累托前沿大小: {pareto_size}, 耗时: {iteration_time:.1f}s")
        else:
            self.logger.info(f"第 {self.current_iteration + 1} 次迭代完成，"
                            f"目标值: {objective_value:.6f}, 耗时: {iteration_time:.1f}s")
    
    def _suggest_initial_parameters(self) -> Dict[str, Any]:
        """建议初始采样参数"""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            try:
                # 使用不同的随机种子避免重复
                seed = self.random_state + attempt if self.random_state else None
                parameters = self.parameter_space.sample_random_parameters(seed=seed)
                
                # 检查是否与已评估的参数过于相似
                if not self._is_too_similar_to_evaluated(parameters):
                    return parameters
                    
            except Exception as e:
                self.logger.warning(f"初始参数采样失败 (尝试 {attempt + 1}): {e}")
                continue
        
        # 如果所有尝试都失败，使用回退策略
        return self._suggest_fallback_parameters()
    
    def _suggest_acquisition_parameters(self) -> Dict[str, Any]:
        """使用采集函数建议参数"""
        try:
            # 获取参数边界（仅连续型参数）
            bounds = self.parameter_space.get_bounds()
            
            if not bounds:
                # 如果没有连续型参数，使用随机采样
                return self.parameter_space.sample_random_parameters()
            
            # 获取当前最佳值
            best_value = self.history.get_best_objective_value()
            if best_value is None:
                best_value = 0.0
            
            # 优化采集函数
            best_continuous_params, _ = self.acquisition_optimizer.optimize(
                self.acquisition_function, 
                self.gaussian_process, 
                bounds, 
                best_value
            )
            
            # 构建完整的参数字典
            parameters = self._construct_full_parameters(best_continuous_params)
            
            # 验证参数有效性
            if not self.parameter_space.validate_parameters(parameters):
                self.logger.warning("采集函数建议的参数无效，使用修复策略")
                parameters = self.parameter_space.suggest_parameter_fix(parameters)
            
            return parameters
            
        except Exception as e:
            self.logger.warning(f"采集函数优化失败: {e}")
            return self._suggest_fallback_parameters()
    
    def _construct_full_parameters(self, continuous_params: np.ndarray) -> Dict[str, Any]:
        """从连续型参数构建完整的参数字典"""
        parameters = {}
        continuous_names = self.parameter_space.get_continuous_parameter_names()
        
        # 设置连续型参数
        for i, name in enumerate(continuous_names):
            if i < len(continuous_params):
                config = self.parameter_space.parameters[name]
                value = continuous_params[i]
                
                # 处理对数尺度参数
                if config.log_scale:
                    value = np.exp(value)
                
                parameters[name] = float(value)
        
        # 随机采样离散型和分类型参数
        discrete_names = self.parameter_space.get_discrete_parameter_names()
        categorical_names = self.parameter_space.get_categorical_parameter_names()
        
        for name in discrete_names + categorical_names:
            config = self.parameter_space.parameters[name]
            parameters[name] = np.random.choice(config.values)
        
        return parameters
    
    def _suggest_fallback_parameters(self) -> Dict[str, Any]:
        """回退参数建议策略"""
        self.logger.info("使用回退参数建议策略")
        
        # 尝试多次随机采样
        for attempt in range(10):
            try:
                parameters = self.parameter_space.sample_random_parameters()
                
                # 确保不与失败的参数过于相似
                if not self._is_too_similar_to_failed(parameters):
                    return parameters
                    
            except Exception as e:
                self.logger.warning(f"回退采样失败 (尝试 {attempt + 1}): {e}")
                continue
        
        # 最后的回退：使用固定的安全参数
        return self._get_safe_parameters()
    
    def _get_safe_parameters(self) -> Dict[str, Any]:
        """获取安全的默认参数组合"""
        safe_params = {}
        
        for name, config in self.parameter_space.parameters.items():
            if config.param_type.value == 'continuous':
                # 使用中位数
                low, high = config.bounds
                if config.log_scale:
                    safe_params[name] = np.exp((np.log(low) + np.log(high)) / 2)
                else:
                    safe_params[name] = (low + high) / 2
            else:
                # 使用第一个值
                safe_params[name] = config.values[0]
        
        return safe_params
    
    def _update_gaussian_process(self, parameters: Dict[str, Any], objective_value: float):
        """更新高斯过程模型"""
        try:
            # 将参数转换为特征向量
            X_new = self._parameters_to_array(parameters).reshape(1, -1)
            y_new = np.array([objective_value])
            
            # 更新高斯过程
            if self.gaussian_process.is_fitted:
                self.gaussian_process.update(X_new, y_new)
            else:
                # 收集所有历史数据进行初始拟合
                if len(self.history.results) >= 2:  # 至少需要2个点
                    X_all = []
                    y_all = []
                    
                    for result in self.history.results:
                        X_all.append(self._parameters_to_array(result.parameters))
                        y_all.append(result.objective_value)
                    
                    X_all = np.array(X_all)
                    y_all = np.array(y_all)
                    
                    self.gaussian_process.fit(X_all, y_all)
                    
        except Exception as e:
            self.logger.error(f"高斯过程更新失败: {e}")
            # 不抛出异常，允许优化继续
    
    def _parameters_to_array(self, parameters: Dict[str, Any]) -> np.ndarray:
        """将参数字典转换为数值数组"""
        features = []
        
        # 按固定顺序处理参数，确保一致性
        param_names = sorted(self.parameter_space.get_parameter_names())
        
        for name in param_names:
            config = self.parameter_space.parameters[name]
            
            if name in parameters:
                value = parameters[name]
                
                if config.param_type.value == 'continuous':
                    if config.log_scale:
                        features.append(np.log(float(value)))
                    else:
                        features.append(float(value))
                        
                elif config.param_type.value == 'discrete':
                    # 使用归一化的索引编码
                    try:
                        idx = config.values.index(value)
                        # 归一化到[0,1]范围
                        normalized_idx = idx / (len(config.values) - 1) if len(config.values) > 1 else 0.0
                        features.append(normalized_idx)
                    except ValueError:
                        features.append(0.0)
                        
                elif config.param_type.value == 'categorical':
                    # 使用归一化的索引编码
                    try:
                        idx = config.values.index(value)
                        # 归一化到[0,1]范围
                        normalized_idx = idx / (len(config.values) - 1) if len(config.values) > 1 else 0.0
                        features.append(normalized_idx)
                    except ValueError:
                        features.append(0.0)
            else:
                # 如果参数缺失，使用默认值
                if config.param_type.value == 'continuous':
                    # 使用中位数
                    low, high = config.bounds
                    if config.log_scale:
                        features.append((np.log(low) + np.log(high)) / 2)
                    else:
                        features.append((low + high) / 2)
                else:
                    # 使用第一个值的编码
                    features.append(0.0)
        
        return np.array(features)
    
    def _is_too_similar_to_evaluated(self, parameters: Dict[str, Any], threshold: float = 0.1) -> bool:
        """检查参数是否与已评估的参数过于相似"""
        if not self.history.results:
            return False
        
        current_array = self._parameters_to_array(parameters)
        
        for result in self.history.results:
            existing_array = self._parameters_to_array(result.parameters)
            
            # 计算归一化的欧几里得距离
            try:
                distance = np.linalg.norm(current_array - existing_array)
                normalized_distance = distance / np.sqrt(len(current_array))
                
                if normalized_distance < threshold:
                    return True
            except:
                continue
        
        return False
    
    def _is_too_similar_to_failed(self, parameters: Dict[str, Any], threshold: float = 0.05) -> bool:
        """检查参数是否与失败的参数过于相似"""
        if not self.failed_parameters:
            return False
        
        current_array = self._parameters_to_array(parameters)
        
        for failed_params in self.failed_parameters:
            try:
                failed_array = self._parameters_to_array(failed_params)
                distance = np.linalg.norm(current_array - failed_array)
                normalized_distance = distance / np.sqrt(len(current_array))
                
                if normalized_distance < threshold:
                    return True
            except:
                continue
        
        return False
    
    def _update_improvement_tracking(self, result: OptimizationResult):
        """更新改进跟踪"""
        if len(self.history.results) <= 1:
            self.no_improvement_count = 0
            return
        
        # 检查是否有改进
        previous_best = self.history.results[-2].objective_value
        current_best = self.history.get_best_objective_value()
        
        if self.maximize:
            has_improvement = current_best > previous_best + self.convergence_threshold
        else:
            has_improvement = current_best < previous_best - self.convergence_threshold
        
        if has_improvement:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
    
    def _check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self.history.results) < self.convergence_window:
            return False
        
        # 获取最近的目标值
        recent_values = [result.objective_value for result in self.history.results[-self.convergence_window:]]
        
        # 检查方差是否足够小
        variance = np.var(recent_values)
        if variance < self.convergence_threshold ** 2:
            return True
        
        # 检查是否长时间无改进
        if self.no_improvement_count >= self.patience:
            self.logger.info(f"连续 {self.no_improvement_count} 次迭代无显著改进，判定为收敛")
            return True
        
        return False
    
    def _should_stop(self, time_limit: Optional[float], target_value: Optional[float]) -> bool:
        """检查是否应该停止优化"""
        # 检查时间限制
        if time_limit and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed >= time_limit:
                self.logger.info(f"达到时间限制 {time_limit:.1f} 秒，停止优化")
                return True
        
        # 检查目标值
        if target_value is not None:
            current_best = self.history.get_best_objective_value()
            if current_best is not None:
                if (self.maximize and current_best >= target_value) or \
                   (not self.maximize and current_best <= target_value):
                    self.logger.info(f"达到目标值 {target_value}，停止优化")
                    return True
        
        return False
    
    def _handle_iteration_error(self, error: Exception):
        """处理迭代过程中的错误"""
        self.consecutive_failures += 1
        self.logger.error(f"第 {self.current_iteration + 1} 次迭代失败 "
                         f"(连续失败 {self.consecutive_failures} 次): {str(error)}")
        
        # 如果是参数相关错误，尝试调整采集函数参数
        if self.consecutive_failures >= 3:
            self._adjust_acquisition_function()
    
    def _adjust_acquisition_function(self):
        """调整采集函数参数以增加探索"""
        try:
            if hasattr(self.acquisition_function, 'xi'):
                # 增加EI或PI的探索参数
                old_xi = self.acquisition_function.xi
                new_xi = min(old_xi * 1.5, 0.1)
                self.acquisition_function.update_xi(new_xi)
                self.logger.info(f"调整采集函数探索参数: xi {old_xi:.4f} -> {new_xi:.4f}")
                
            elif hasattr(self.acquisition_function, 'kappa'):
                # 增加UCB的探索参数
                old_kappa = self.acquisition_function.kappa
                new_kappa = min(old_kappa * 1.2, 5.0)
                self.acquisition_function.update_kappa(new_kappa)
                self.logger.info(f"调整采集函数探索参数: kappa {old_kappa:.4f} -> {new_kappa:.4f}")
                
        except Exception as e:
            self.logger.warning(f"调整采集函数参数失败: {e}")
    
    def _save_checkpoint(self):
        """保存检查点"""
        try:
            optimizer_state = {
                'history': self.history,
                'parameter_space': self.parameter_space,
                'gaussian_process': self.gaussian_process,
                'acquisition_function': self.acquisition_function.get_info(),
                'config': {
                    'n_initial_points': self.n_initial_points,
                    'random_state': self.random_state,
                    'maximize': self.maximize,
                    'convergence_window': self.convergence_window,
                    'convergence_threshold': self.convergence_threshold,
                    'patience': self.patience,
                    'current_iteration': self.current_iteration,
                    'consecutive_failures': self.consecutive_failures,
                    'no_improvement_count': self.no_improvement_count
                }
            }
            
            checkpoint_path = self.state_manager.save_state(optimizer_state, self.current_iteration)
            self.last_checkpoint_time = datetime.now()
            self.logger.info(f"检查点保存成功: {checkpoint_path}")
            
        except Exception as e:
            self.logger.warning(f"保存检查点失败: {e}")
    
    def _save_emergency_checkpoint(self):
        """保存紧急检查点"""
        try:
            optimizer_state = {
                'history': self.history,
                'parameter_space': self.parameter_space,
                'gaussian_process': self.gaussian_process,
                'acquisition_function': self.acquisition_function.get_info(),
                'config': {
                    'current_iteration': self.current_iteration,
                    'emergency_save': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            emergency_path = self.state_manager.save_state(
                optimizer_state, 
                self.current_iteration, 
                f"emergency_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.logger.info(f"紧急检查点保存成功: {emergency_path}")
            
        except Exception as e:
            self.logger.error(f"保存紧急检查点失败: {e}")
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """从检查点恢复优化状态"""
        try:
            self.logger.info(f"从检查点恢复: {checkpoint_path}")
            
            # 加载状态
            optimizer_state = self.state_manager.load_state(checkpoint_path)
            
            # 恢复组件
            self.history = optimizer_state['history']
            self.parameter_space = optimizer_state['parameter_space']
            self.gaussian_process = optimizer_state['gaussian_process']
            
            # 恢复配置
            config = optimizer_state.get('config', {})
            self.current_iteration = config.get('current_iteration', len(self.history.results))
            self.consecutive_failures = config.get('consecutive_failures', 0)
            self.no_improvement_count = config.get('no_improvement_count', 0)
            
            # 重新创建采集函数
            acq_info = optimizer_state.get('acquisition_function', {})
            if acq_info:
                self.acquisition_function = create_acquisition_function(
                    acq_info.get('function_type', 'EI'),
                    **acq_info.get('parameters', {})
                )
            
            # 标记为已初始化
            self.is_initialized = True
            self.start_time = datetime.now()
            
            self.logger.info(f"成功恢复优化状态，当前迭代: {self.current_iteration}")
            self.logger.info(f"历史记录包含 {len(self.history.results)} 个结果")
            
        except Exception as e:
            error_msg = f"从检查点恢复失败: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg) from e
    
    def _finalize_optimization(self):
        """完成优化过程"""
        self.history.end_time = datetime.now()
        self.history.total_iterations = self.current_iteration
        
        # 保存最终检查点
        try:
            self._save_checkpoint()
        except Exception as e:
            self.logger.warning(f"保存最终检查点失败: {e}")
        
        # 清理资源
        try:
            self.task_evaluator.cleanup()
        except Exception as e:
            self.logger.warning(f"清理任务评估器资源失败: {e}")


def create_bayesian_optimizer(task_type: str = "LDA",
                             data_config: Optional[Dict[str, Any]] = None,
                             acquisition_function_type: str = "EI",
                             n_initial_points: int = 10,
                             random_state: Optional[int] = None,
                             checkpoint_dir: str = "checkpoints",
                             # 多目标优化参数
                             objectives: Optional[List[str]] = None,
                             maximize_objectives: Optional[Dict[str, bool]] = None,
                             objective_weights: Optional[Dict[str, float]] = None) -> BayesianOptimizer:
    """
    创建贝叶斯优化器的工厂函数
    
    Args:
        task_type: 任务类型 ('LDA', 'MDA', 'LMI')
        data_config: 数据配置
        acquisition_function_type: 采集函数类型 ('EI', 'PI', 'UCB', 'ES')
        n_initial_points: 初始采样点数
        random_state: 随机种子
        checkpoint_dir: 检查点目录
        objectives: 多目标优化的目标函数名称列表
        maximize_objectives: 每个目标是否最大化的字典
        objective_weights: 目标权重字典
        
    Returns:
        配置好的BayesianOptimizer实例
    """
    from autodl_core import create_default_parameter_space
    
    # 创建组件
    parameter_space = create_default_parameter_space()
    task_evaluator = create_task_evaluator(task_type, data_config, use_real_training=False)
    acquisition_function = create_acquisition_function(acquisition_function_type)
    gaussian_process = create_default_gaussian_process(random_state)
    state_manager = create_default_state_manager(checkpoint_dir)
    
    # 创建优化器
    optimizer = BayesianOptimizer(
        parameter_space=parameter_space,
        task_evaluator=task_evaluator,
        acquisition_function=acquisition_function,
        gaussian_process=gaussian_process,
        state_manager=state_manager,
        n_initial_points=n_initial_points,
        random_state=random_state,
        maximize=True,
        objectives=objectives,
        maximize_objectives=maximize_objectives,
        objective_weights=objective_weights
    )
    
    return optimizer


def create_multi_objective_optimizer(task_type: str = "LDA",
                                   objectives: List[str] = None,
                                   objective_weights: Optional[Dict[str, float]] = None,
                                   data_config: Optional[Dict[str, Any]] = None,
                                   acquisition_function_type: str = "EI",
                                   n_initial_points: int = 10,
                                   random_state: Optional[int] = None,
                                   checkpoint_dir: str = "checkpoints") -> BayesianOptimizer:
    """
    创建多目标贝叶斯优化器的便捷函数
    
    Args:
        task_type: 任务类型
        objectives: 目标函数名称列表，如 ['AUROC', 'AUPRC', 'F1']
        objective_weights: 目标权重字典
        data_config: 数据配置
        acquisition_function_type: 采集函数类型
        n_initial_points: 初始采样点数
        random_state: 随机种子
        checkpoint_dir: 检查点目录
        
    Returns:
        配置好的多目标BayesianOptimizer实例
    """
    if objectives is None:
        objectives = ['AUROC', 'AUPRC', 'F1']
    
    # 默认所有目标都是最大化
    maximize_objectives = {obj: True for obj in objectives}
    
    # 如果没有提供权重，使用均等权重
    if objective_weights is None:
        weight = 1.0 / len(objectives)
        objective_weights = {obj: weight for obj in objectives}
    
    return create_bayesian_optimizer(
        task_type=task_type,
        data_config=data_config,
        acquisition_function_type=acquisition_function_type,
        n_initial_points=n_initial_points,
        random_state=random_state,
        checkpoint_dir=checkpoint_dir,
        objectives=objectives,
        maximize_objectives=maximize_objectives,
        objective_weights=objective_weights
    )


if __name__ == "__main__":
    # 测试代码
    print("测试贝叶斯优化器...")
    
    try:
        # 测试单目标优化
        print("=== 单目标优化测试 ===")
        optimizer = create_bayesian_optimizer(
            task_type="LDA",
            acquisition_function_type="EI",
            n_initial_points=3,
            random_state=42
        )
        
        print(f"优化器创建成功")
        print(f"参数空间: {optimizer.parameter_space.get_parameter_count()} 个参数")
        print(f"采集函数: {optimizer.acquisition_function.function_type}")
        print(f"是否多目标: {optimizer.is_multi_objective}")
        
        # 测试短期优化
        print("\n开始短期单目标优化测试...")
        history = optimizer.optimize(n_iterations=5, checkpoint_freq=2)
        
        print(f"单目标优化完成:")
        print(f"  总迭代次数: {history.total_iterations}")
        print(f"  最佳目标值: {history.get_best_objective_value():.6f}")
        print(f"  总耗时: {history.total_time:.1f} 秒")
        
        # 测试多目标优化
        print("\n=== 多目标优化测试 ===")
        multi_optimizer = create_multi_objective_optimizer(
            task_type="LDA",
            objectives=['AUROC', 'AUPRC', 'F1'],
            objective_weights={'AUROC': 0.5, 'AUPRC': 0.3, 'F1': 0.2},
            n_initial_points=3,
            random_state=42
        )
        
        print(f"多目标优化器创建成功")
        print(f"目标函数: {multi_optimizer.objectives}")
        print(f"目标权重: {multi_optimizer.objective_weights}")
        print(f"是否多目标: {multi_optimizer.is_multi_objective}")
        
        # 测试短期多目标优化
        print("\n开始短期多目标优化测试...")
        multi_history = multi_optimizer.optimize(n_iterations=5, checkpoint_freq=2)
        
        print(f"多目标优化完成:")
        print(f"  总迭代次数: {multi_history.total_iterations}")
        print(f"  帕累托前沿大小: {len(multi_history.pareto_front)}")
        print(f"  总耗时: {multi_history.total_time:.1f} 秒")
        
        # 显示帕累托前沿
        if multi_history.pareto_front:
            print("\n帕累托前沿解:")
            for i, result in enumerate(multi_history.pareto_front[:3]):  # 显示前3个解
                obj_vals = result.objective_values
                print(f"  解 {i+1}: AUROC={obj_vals['AUROC']:.4f}, "
                      f"AUPRC={obj_vals['AUPRC']:.4f}, F1={obj_vals['F1']:.4f}")
        
        # 测试超体积计算
        try:
            hypervolume = multi_optimizer.compute_hypervolume()
            print(f"  超体积: {hypervolume:.6f}")
        except Exception as e:
            print(f"  超体积计算失败: {e}")
        
        # 测试优化状态
        status = multi_optimizer.get_optimization_status()
        print(f"\n多目标优化状态:")
        print(f"  帕累托前沿大小: {status['pareto_front_size']}")
        print(f"  目标权重: {status['objective_weights']}")
        
        print("\n贝叶斯优化器测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()