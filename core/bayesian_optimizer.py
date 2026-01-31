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

# 导入新的输出系统
from unified_log_manager import UnifiedLogManager, get_global_log_manager, init_global_log_manager
from structured_tag_processor import get_global_tag_processor


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
        
        # 处理目标权重归一化
        if objective_weights:
            total_weight = sum(objective_weights.values())
            if total_weight > 0:
                self.objective_weights = {k: v / total_weight for k, v in objective_weights.items()}
            else:
                self.objective_weights = objective_weights
        else:
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
        
        # 获取日志管理器
        self.log_manager = get_global_log_manager()
        if not self.log_manager:
            # 如果没有全局日志管理器，创建一个临时的
            self.log_manager = init_global_log_manager(f"bayesian_optimizer_{task_evaluator.task_type}")
        
        # 记录初始化信息
        self._log_initialization_info()
    
    def _log_initialization_info(self):
        """记录详细的初始化信息"""
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "========== 贝叶斯优化器初始化开始 ==========", 
                                     component_name)
        
        # 基本配置信息
        self.log_manager.log_structured(logging.INFO, "CONFIG", {
            "task_type": self.task_evaluator.task_type,
            "parameter_count": self.parameter_space.get_parameter_count(),
            "acquisition_function": self.acquisition_function.function_type,
            "n_initial_points": self.n_initial_points,
            "random_state": self.random_state,
            "maximize": self.maximize,
            "is_multi_objective": self.is_multi_objective
        }, component_name)
        
        # 参数空间详细信息
        param_details = {}
        for name, config in self.parameter_space.parameters.items():
            if config.param_type.value == 'continuous':
                param_details[name] = f"{config.param_type.value}[{config.bounds[0]:.4f}, {config.bounds[1]:.4f}]"
            else:
                param_details[name] = f"{config.param_type.value}{config.values}"
        
        self.log_manager.log_structured(logging.INFO, "PARAMS", param_details, component_name)
        
        # 多目标优化配置
        if self.is_multi_objective:
            self.log_manager.log_structured(logging.INFO, "MULTI_OBJECTIVE", {
                "objectives": self.objectives,
                "maximize_objectives": self.maximize_objectives,
                "objective_weights": self.objective_weights
            }, component_name)
        
        # 收敛检测配置
        self.log_manager.log_structured(logging.INFO, "CONVERGENCE", {
            "convergence_window": self.convergence_window,
            "convergence_threshold": self.convergence_threshold,
            "patience": self.patience,
            "max_consecutive_failures": self.max_consecutive_failures
        }, component_name)
        
        # 采集函数详细信息
        acq_info = self.acquisition_function.get_info()
        self.log_manager.log_structured(logging.INFO, "ACQUISITION", acq_info, component_name)
        
        # 高斯过程配置
        gp_info = {
            "kernel_type": str(type(self.gaussian_process.kernel).__name__),
            "is_fitted": self.gaussian_process.is_fitted,
            "n_restarts_optimizer": getattr(self.gaussian_process, 'n_restarts_optimizer', 'N/A')
        }
        self.log_manager.log_structured(logging.INFO, "GAUSSIAN_PROCESS", gp_info, component_name)
        
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "贝叶斯优化器初始化完成", 
                                     component_name)
    
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
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        try:
            # 恢复或初始化优化状态
            if resume_from_checkpoint:
                self._resume_from_checkpoint(resume_from_checkpoint)
            else:
                self._initialize_optimization()
            
            # 详细的优化开始信息
            self.log_manager.log_with_tag(logging.INFO, "OPTIMIZATION", 
                                         "========== 贝叶斯优化开始 ==========", 
                                         component_name)
            
            # 优化配置信息
            optimization_config = {
                "max_iterations": n_iterations,
                "checkpoint_frequency": checkpoint_freq,
                "time_limit_seconds": time_limit,
                "target_value": target_value,
                "resume_from_checkpoint": resume_from_checkpoint is not None,
                "current_iteration": self.current_iteration,
                "total_evaluations": len(self.history.results)
            }
            self.log_manager.log_structured(logging.INFO, "CONFIG", optimization_config, component_name)
            
            # 时间和目标信息
            if time_limit:
                time_limit_hours = time_limit / 3600
                self.log_manager.log_with_tag(logging.INFO, "CONFIG", 
                                             f"时间限制: {time_limit:.1f}秒 ({time_limit_hours:.2f}小时)", 
                                             component_name)
            if target_value:
                self.log_manager.log_with_tag(logging.INFO, "CONFIG", 
                                             f"目标值: {target_value:.6f}", 
                                             component_name)
            
            # 当前状态信息
            if self.history.best_result:
                self.log_manager.log_structured(logging.INFO, "CURRENT_STATE", {
                    "best_objective_value": self.history.get_best_objective_value(),
                    "total_evaluations": len(self.history.results),
                    "consecutive_failures": self.consecutive_failures,
                    "no_improvement_count": self.no_improvement_count
                }, component_name)
            
            # 主优化循环
            self.log_manager.log_with_tag(logging.INFO, "OPTIMIZATION", 
                                         "开始主优化循环", component_name)
            
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
                    
                    # 详细的迭代完成信息
                    iteration_info = {
                        "iteration": self.current_iteration,
                        "iteration_time": iteration_time,
                        "total_time": self.history.total_time,
                        "current_best": self.history.get_best_objective_value(),
                        "consecutive_failures": self.consecutive_failures,
                        "no_improvement_count": self.no_improvement_count
                    }
                    
                    if self.is_multi_objective:
                        iteration_info["pareto_front_size"] = len(self.history.pareto_front)
                    
                    self.log_manager.log_structured(logging.INFO, "ITERATION_COMPLETE", 
                                                   iteration_info, component_name)
                    
                    # 检查收敛
                    if self._check_convergence():
                        self.log_manager.log_with_tag(logging.INFO, "CONVERGENCE", 
                                                     f"优化在第 {self.current_iteration} 次迭代后收敛", 
                                                     component_name)
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
                        self.log_manager.log_with_tag(logging.ERROR, "ERROR", error_msg, component_name)
                        raise OptimizationError(error_msg)
            
            # 优化完成
            self._finalize_optimization()
            
            # 详细的完成信息
            completion_info = {
                "total_iterations": self.current_iteration,
                "total_evaluations": len(self.history.results),
                "best_objective_value": self.history.get_best_objective_value(),
                "total_time_seconds": self.history.total_time,
                "total_time_hours": self.history.total_time / 3600,
                "average_iteration_time": self.history.total_time / max(self.current_iteration, 1),
                "convergence_detected": self._check_convergence(),
                "final_consecutive_failures": self.consecutive_failures
            }
            
            if self.is_multi_objective:
                completion_info.update({
                    "pareto_front_size": len(self.history.pareto_front),
                    "pareto_coverage": len(self.history.pareto_front) / max(len(self.history.results), 1)
                })
            
            self.log_manager.log_structured(logging.INFO, "OPTIMIZATION_COMPLETE", 
                                           completion_info, component_name)
            
            self.log_manager.log_with_tag(logging.INFO, "OPTIMIZATION", 
                                         "========== 贝叶斯优化完成 ==========", 
                                         component_name)
            
            return self.history
            
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"优化过程出现严重错误: {str(e)}", 
                                         component_name)
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
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        if not self.is_initialized:
            raise RuntimeError("优化器尚未初始化，请先调用optimize()或_initialize_optimization()")
        
        self.log_manager.log_with_tag(logging.INFO, "SUGGESTION", 
                                     "开始生成参数建议", component_name)
        
        suggestion_start_time = time.time()
        
        try:
            # 如果还在初始采样阶段
            if len(self.history.results) < self.n_initial_points:
                self.log_manager.log_with_tag(logging.INFO, "SUGGESTION", 
                                             f"初始采样阶段: {len(self.history.results)}/{self.n_initial_points}", 
                                             component_name)
                parameters = self._suggest_initial_parameters()
                suggestion_type = "初始随机采样"
            else:
                # 使用采集函数建议参数
                self.log_manager.log_with_tag(logging.INFO, "SUGGESTION", 
                                             "使用采集函数生成参数建议", component_name)
                
                # 记录当前高斯过程状态
                gp_status = {
                    "is_fitted": self.gaussian_process.is_fitted,
                    "n_training_samples": len(self.history.results),
                    "best_observed_value": self.history.get_best_objective_value()
                }
                self.log_manager.log_structured(logging.INFO, "GP_STATUS", gp_status, component_name)
                
                parameters = self._suggest_acquisition_parameters()
                suggestion_type = f"采集函数({self.acquisition_function.function_type})"
            
            suggestion_time = time.time() - suggestion_start_time
            
            # 调试参数维度
            self._debug_parameter_dimensions(parameters)
            
            # 验证参数有效性
            is_valid, validation_errors = self.task_evaluator.validate_parameters(parameters)
            
            # 详细的参数建议信息
            suggestion_info = {
                "suggestion_type": suggestion_type,
                "suggestion_time": suggestion_time,
                "parameter_count": len(parameters),
                "is_valid": is_valid,
                "validation_errors": validation_errors if not is_valid else None,
                "similar_to_previous": self._is_too_similar_to_evaluated(parameters),
                "similar_to_failed": self._is_too_similar_to_failed(parameters)
            }
            
            self.log_manager.log_structured(logging.INFO, "SUGGESTION_INFO", 
                                           suggestion_info, component_name)
            
            # 格式化参数显示
            formatted_params = {}
            for key, value in parameters.items():
                if isinstance(value, float):
                    if abs(value) < 0.001:
                        formatted_params[key] = f"{value:.2e}"
                    else:
                        formatted_params[key] = f"{value:.6f}"
                else:
                    formatted_params[key] = str(value)
            
            self.log_manager.log_structured(logging.INFO, "SUGGESTED_PARAMS", 
                                           formatted_params, component_name)
            
            # 如果使用采集函数，记录采集函数值
            if suggestion_type != "初始随机采样":
                try:
                    feature_vector = self._parameters_to_array(parameters).reshape(1, -1)
                    best_value = self.history.get_best_objective_value() or 0.0
                    acquisition_value = self.acquisition_function.evaluate(
                        feature_vector, self.gaussian_process, best_value
                    )[0]
                    
                    self.log_manager.log_structured(logging.INFO, "ACQUISITION_VALUE", {
                        "acquisition_function": self.acquisition_function.function_type,
                        "acquisition_value": acquisition_value,
                        "best_observed_value": best_value,
                        "exploration_exploitation_balance": getattr(self.acquisition_function, 'xi', 'N/A')
                    }, component_name)
                    
                except Exception as e:
                    self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                                 f"无法计算采集函数值: {e}", component_name)
            
            self.log_manager.log_with_tag(logging.INFO, "SUGGESTION", 
                                         f"参数建议生成完成 (耗时: {suggestion_time:.4f}秒)", 
                                         component_name)
            
            return parameters
            
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"参数建议失败: {str(e)}", component_name)
            # 回退到随机采样
            self.log_manager.log_with_tag(logging.INFO, "FALLBACK", 
                                         "使用回退策略生成参数", component_name)
            fallback_params = self._suggest_fallback_parameters()
            self._debug_parameter_dimensions(fallback_params)
            
            fallback_time = time.time() - suggestion_start_time
            self.log_manager.log_structured(logging.INFO, "FALLBACK_PARAMS", {
                "fallback_reason": str(e),
                "fallback_time": fallback_time,
                "parameters": {k: (f"{v:.6f}" if isinstance(v, float) else str(v)) 
                              for k, v in fallback_params.items()}
            }, component_name)
            
            return fallback_params
    
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
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        self.log_manager.log_with_tag(logging.INFO, "MODEL_UPDATE", 
                                     "开始更新贝叶斯优化模型", component_name)
        
        update_start_time = time.time()
        
        try:
            # 如果是多目标优化但没有提供多目标值，从metrics中提取
            if self.is_multi_objective and objective_values is None:
                objective_values = self._extract_objective_values(metrics)
                self.log_manager.log_with_tag(logging.INFO, "MODEL_UPDATE", 
                                             f"从指标中提取多目标值: {objective_values}", 
                                             component_name)
            
            # 记录更新前的状态
            pre_update_state = {
                "total_evaluations": len(self.history.results),
                "current_best": self.history.get_best_objective_value(),
                "gp_fitted": self.gaussian_process.is_fitted,
                "no_improvement_count": self.no_improvement_count
            }
            
            if self.is_multi_objective:
                pre_update_state["pareto_front_size"] = len(self.history.pareto_front)
            
            self.log_manager.log_structured(logging.INFO, "PRE_UPDATE_STATE", 
                                           pre_update_state, component_name)
            
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
            
            # 记录新结果的详细信息
            result_info = {
                "iteration": result.iteration,
                "objective_value": objective_value,
                "evaluation_time": evaluation_time,
                "parameter_count": len(parameters),
                "metric_count": len(metrics)
            }
            
            if objective_values:
                result_info["objective_values"] = objective_values
            
            # 显示主要指标
            main_metrics = {}
            for key in ['AUROC', 'AUPRC', 'F1', 'precision', 'recall', 'loss']:
                if key in metrics:
                    main_metrics[key] = metrics[key]
            
            if main_metrics:
                result_info["main_metrics"] = main_metrics
            
            self.log_manager.log_structured(logging.INFO, "NEW_RESULT", 
                                           result_info, component_name)
            
            # 添加到历史记录
            self.history.add_result(result, maximize=self.maximize)
            
            # 更新高斯过程模型
            gp_update_start = time.time()
            
            if self.is_multi_objective and self.objective_weights:
                # 多目标：使用加权目标值更新模型
                weighted_value = self.history.get_weighted_objective_value(result)
                self.log_manager.log_structured(logging.INFO, "WEIGHTED_OBJECTIVE", {
                    "original_value": objective_value,
                    "weighted_value": weighted_value,
                    "weights": self.objective_weights
                }, component_name)
                self._update_gaussian_process(parameters, weighted_value)
            else:
                # 单目标：使用主要目标值更新模型
                self._update_gaussian_process(parameters, objective_value)
            
            gp_update_time = time.time() - gp_update_start
            
            # 更新无改进计数
            old_no_improvement = self.no_improvement_count
            self._update_improvement_tracking(result)
            
            # 记录更新后的状态
            post_update_state = {
                "total_evaluations": len(self.history.results),
                "current_best": self.history.get_best_objective_value(),
                "gp_fitted": self.gaussian_process.is_fitted,
                "gp_update_time": gp_update_time,
                "no_improvement_count": self.no_improvement_count,
                "improvement_detected": self.no_improvement_count < old_no_improvement
            }
            
            if self.is_multi_objective:
                post_update_state.update({
                    "pareto_front_size": len(self.history.pareto_front),
                    "is_pareto_optimal": result.is_pareto_optimal if hasattr(result, 'is_pareto_optimal') else False
                })
            
            self.log_manager.log_structured(logging.INFO, "POST_UPDATE_STATE", 
                                           post_update_state, component_name)
            
            # 分析改进情况
            if len(self.history.results) > 1:
                previous_best = self.history.results[-2].objective_value
                improvement = objective_value - previous_best
                improvement_percent = (improvement / abs(previous_best)) * 100 if previous_best != 0 else 0
                
                improvement_analysis = {
                    "previous_best": previous_best,
                    "current_value": objective_value,
                    "absolute_improvement": improvement,
                    "relative_improvement_percent": improvement_percent,
                    "is_improvement": improvement > 0 if self.maximize else improvement < 0,
                    "improvement_magnitude": abs(improvement)
                }
                
                self.log_manager.log_structured(logging.INFO, "IMPROVEMENT_ANALYSIS", 
                                               improvement_analysis, component_name)
                
                if improvement_analysis["is_improvement"]:
                    self.log_manager.log_with_tag(logging.INFO, "SUCCESS", 
                                                 f"发现改进! 提升: {improvement:+.6f} ({improvement_percent:+.2f}%)", 
                                                 component_name)
            
            total_update_time = time.time() - update_start_time
            
            # 最终更新摘要
            update_summary = {
                "total_update_time": total_update_time,
                "gp_update_time": gp_update_time,
                "history_size": len(self.history.results),
                "consecutive_failures_reset": self.consecutive_failures == 0
            }
            
            if self.is_multi_objective:
                update_summary["pareto_front_size"] = len(self.history.pareto_front)
                self.log_manager.log_with_tag(logging.INFO, "MODEL_UPDATE", 
                                             f"模型更新完成，帕累托前沿大小: {len(self.history.pareto_front)}", 
                                             component_name)
            else:
                current_best = self.history.get_best_objective_value()
                update_summary["current_best_value"] = current_best
                self.log_manager.log_with_tag(logging.INFO, "MODEL_UPDATE", 
                                             f"模型更新完成，当前最佳值: {current_best:.6f}", 
                                             component_name)
            
            self.log_manager.log_structured(logging.INFO, "UPDATE_SUMMARY", 
                                           update_summary, component_name)
            
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"模型更新失败: {str(e)}", component_name)
            # 记录失败的详细信息
            failure_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "parameters": {k: (f"{v:.6f}" if isinstance(v, float) else str(v)) 
                              for k, v in parameters.items()},
                "objective_value": objective_value,
                "metrics_count": len(metrics)
            }
            self.log_manager.log_structured(logging.ERROR, "UPDATE_FAILURE", 
                                           failure_info, component_name)
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
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "========== 初始化贝叶斯优化过程 ==========", 
                                     component_name)
        
        # 设置随机种子
        if self.random_state is not None:
            np.random.seed(self.random_state)
            self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                         f"设置随机种子: {self.random_state}", 
                                         component_name)
        
        # 重置状态
        self.current_iteration = 0
        self.start_time = datetime.now()
        self.history.start_time = self.start_time
        self.consecutive_failures = 0
        self.no_improvement_count = 0
        self.failed_parameters.clear()
        
        # 记录初始化状态
        init_state = {
            "start_time": self.start_time.isoformat(),
            "current_iteration": self.current_iteration,
            "consecutive_failures": self.consecutive_failures,
            "no_improvement_count": self.no_improvement_count,
            "failed_parameters_count": len(self.failed_parameters),
            "history_results_count": len(self.history.results)
        }
        
        self.log_manager.log_structured(logging.INFO, "INIT_STATE", init_state, component_name)
        
        # 验证组件状态
        component_status = {
            "parameter_space_valid": self.parameter_space is not None,
            "task_evaluator_valid": self.task_evaluator is not None,
            "acquisition_function_valid": self.acquisition_function is not None,
            "gaussian_process_valid": self.gaussian_process is not None,
            "state_manager_valid": self.state_manager is not None
        }
        
        self.log_manager.log_structured(logging.INFO, "COMPONENT_STATUS", 
                                       component_status, component_name)
        
        # 检查组件完整性
        missing_components = [name for name, valid in component_status.items() if not valid]
        if missing_components:
            error_msg = f"缺少必要组件: {missing_components}"
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", error_msg, component_name)
            raise RuntimeError(error_msg)
        
        # 标记为已初始化
        self.is_initialized = True
        
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "贝叶斯优化过程初始化完成", component_name)
    
    def _run_single_iteration(self):
        """执行单次优化迭代"""
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        iteration_start = time.time()
        
        self.log_manager.log_with_tag(logging.INFO, "ITERATION", 
                                     f"========== 第 {self.current_iteration + 1} 次迭代开始 ==========", 
                                     component_name)
        
        # 建议下一个参数组合
        suggestion_start = time.time()
        parameters = self.suggest_next_parameters()
        suggestion_time = time.time() - suggestion_start
        
        self.log_manager.log_structured(logging.INFO, "ITERATION_PARAMS", {
            "iteration": self.current_iteration + 1,
            "suggestion_time": suggestion_time,
            "parameter_count": len(parameters)
        }, component_name)
        
        # 格式化参数显示
        formatted_params = []
        for key, value in parameters.items():
            if isinstance(value, float):
                if abs(value) < 0.001:
                    formatted_params.append(f"{key}={value:.2e}")
                else:
                    formatted_params.append(f"{key}={value:.6f}")
            else:
                formatted_params.append(f"{key}={value}")
        
        param_str = ", ".join(formatted_params)
        self.log_manager.log_with_tag(logging.INFO, "ITERATION", 
                                     f"评估参数: {param_str}", component_name)
        
        # 评估参数
        self.log_manager.log_with_tag(logging.INFO, "EVALUATION", 
                                     "开始参数评估...", component_name)
        evaluation_start = time.time()
        metrics = self.task_evaluator.evaluate_parameters(parameters)
        evaluation_time = time.time() - evaluation_start
        
        # 提取目标值
        objective_value = self.task_evaluator.get_objective_value(metrics)
        
        # 记录评估结果
        evaluation_info = {
            "evaluation_time": evaluation_time,
            "objective_value": objective_value,
            "metrics_count": len(metrics)
        }
        
        # 添加主要指标
        main_metrics = {}
        for key in ['AUROC', 'AUPRC', 'F1', 'precision', 'recall', 'loss']:
            if key in metrics:
                main_metrics[key] = metrics[key]
        
        if main_metrics:
            evaluation_info["main_metrics"] = main_metrics
        
        self.log_manager.log_structured(logging.INFO, "EVALUATION_RESULT", 
                                       evaluation_info, component_name)
        
        # 检查评估是否成功
        if 'error' in metrics:
            error_msg = f"参数评估失败: {metrics['error']}"
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", error_msg, component_name)
            self.failed_parameters.append(parameters)
            raise RuntimeError(error_msg)
        
        # 提取多目标值（如果适用）
        objective_values = None
        if self.is_multi_objective:
            objective_values = self._extract_objective_values(metrics)
            self.log_manager.log_structured(logging.INFO, "MULTI_OBJECTIVE_VALUES", 
                                           objective_values, component_name)
        
        # 更新模型
        self.update_model(parameters, objective_value, metrics, evaluation_time, objective_values)
        
        iteration_time = time.time() - iteration_start
        
        # 迭代完成摘要
        iteration_summary = {
            "iteration": self.current_iteration + 1,
            "total_iteration_time": iteration_time,
            "suggestion_time": suggestion_time,
            "evaluation_time": evaluation_time,
            "model_update_time": iteration_time - suggestion_time - evaluation_time,
            "objective_value": objective_value,
            "current_best": self.history.get_best_objective_value()
        }
        
        if self.is_multi_objective:
            iteration_summary["pareto_front_size"] = len(self.history.pareto_front)
            self.log_manager.log_with_tag(logging.INFO, "ITERATION", 
                                         f"第 {self.current_iteration + 1} 次迭代完成，"
                                         f"目标值: {objective_value:.6f}, 帕累托前沿大小: {len(self.history.pareto_front)}, "
                                         f"耗时: {iteration_time:.1f}s", 
                                         component_name)
        else:
            self.log_manager.log_with_tag(logging.INFO, "ITERATION", 
                                         f"第 {self.current_iteration + 1} 次迭代完成，"
                                         f"目标值: {objective_value:.6f}, 耗时: {iteration_time:.1f}s", 
                                         component_name)
        
        self.log_manager.log_structured(logging.INFO, "ITERATION_SUMMARY", 
                                       iteration_summary, component_name)
    
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
            
            # 使用混合参数优化策略
            return self._optimize_mixed_parameters(bounds, best_value)
            
        except Exception as e:
            self.logger.warning(f"采集函数优化失败: {e}")
            return self._suggest_fallback_parameters()
    
    def _optimize_mixed_parameters(self, bounds: List[Tuple[float, float]], best_value: float) -> Dict[str, Any]:
        """优化混合参数类型（连续型+离散型+分类型）"""
        best_params = None
        best_acquisition_value = -np.inf
        max_attempts = 20
        
        # 获取离散型和分类型参数的所有可能组合
        discrete_names = self.parameter_space.get_discrete_parameter_names()
        categorical_names = self.parameter_space.get_categorical_parameter_names()
        
        # 如果离散/分类参数太多，随机采样一些组合
        discrete_categorical_combinations = self._sample_discrete_categorical_combinations(
            discrete_names, categorical_names, max_combinations=10
        )
        
        for combination in discrete_categorical_combinations:
            try:
                # 为当前离散/分类参数组合优化连续型参数
                continuous_params = self._optimize_continuous_for_combination(
                    bounds, best_value, combination
                )
                
                if continuous_params is not None:
                    # 构建完整参数
                    full_params = self._construct_full_parameters_with_combination(
                        continuous_params, combination
                    )
                    
                    # 评估采集函数值
                    feature_vector = self._parameters_to_array(full_params).reshape(1, -1)
                    acquisition_value = self.acquisition_function.evaluate(
                        feature_vector, self.gaussian_process, best_value
                    )[0]
                    
                    if acquisition_value > best_acquisition_value:
                        best_acquisition_value = acquisition_value
                        best_params = full_params
                        
            except Exception as e:
                self.logger.debug(f"优化组合失败: {e}")
                continue
        
        if best_params is None:
            # 如果所有尝试都失败，使用回退策略
            return self._suggest_fallback_parameters()
        
        return best_params
    
    def _sample_discrete_categorical_combinations(self, discrete_names: List[str], 
                                                categorical_names: List[str], 
                                                max_combinations: int = 10) -> List[Dict[str, Any]]:
        """采样离散型和分类型参数的组合"""
        combinations = []
        
        # 如果有历史数据，优先使用表现好的组合
        if self.history.results:
            # 获取前几个最佳结果的离散/分类参数组合
            sorted_results = sorted(self.history.results, 
                                  key=lambda r: r.objective_value, reverse=True)
            
            for result in sorted_results[:max_combinations//2]:
                combination = {}
                for name in discrete_names + categorical_names:
                    if name in result.parameters:
                        combination[name] = result.parameters[name]
                
                if combination and combination not in combinations:
                    combinations.append(combination)
        
        # 补充随机组合
        while len(combinations) < max_combinations:
            combination = {}
            
            for name in discrete_names + categorical_names:
                config = self.parameter_space.parameters[name]
                combination[name] = np.random.choice(config.values)
            
            if combination not in combinations:
                combinations.append(combination)
        
        return combinations
    
    def _optimize_continuous_for_combination(self, bounds: List[Tuple[float, float]], 
                                           best_value: float, 
                                           discrete_categorical_params: Dict[str, Any]) -> Optional[np.ndarray]:
        """为给定的离散/分类参数组合优化连续型参数"""
        try:
            # 创建一个特殊的采集函数，固定离散/分类参数
            def objective(continuous_array):
                try:
                    full_params = self._construct_full_parameters_with_combination(
                        continuous_array, discrete_categorical_params
                    )
                    feature_vector = self._parameters_to_array(full_params).reshape(1, -1)
                    acquisition_value = self.acquisition_function.evaluate(
                        feature_vector, self.gaussian_process, best_value
                    )[0]
                    return -acquisition_value  # 最小化负值
                except Exception as e:
                    return np.inf
            
            # 多次随机重启优化
            best_x = None
            best_value_opt = np.inf
            
            for _ in range(5):  # 减少重启次数以提高效率
                # 随机初始点
                x0 = np.array([
                    np.random.uniform(low, high) 
                    for low, high in bounds
                ])
                
                try:
                    from scipy.optimize import minimize
                    result = minimize(
                        objective, x0, method='L-BFGS-B', 
                        bounds=bounds, options={'maxiter': 100}
                    )
                    
                    if result.success and result.fun < best_value_opt:
                        best_value_opt = result.fun
                        best_x = result.x
                        
                except Exception:
                    continue
            
            return best_x
            
        except Exception as e:
            self.logger.debug(f"连续参数优化失败: {e}")
            return None
    
    def _construct_full_parameters_with_combination(self, continuous_params: np.ndarray, 
                                                  discrete_categorical_params: Dict[str, Any]) -> Dict[str, Any]:
        """使用给定的离散/分类参数组合构建完整参数"""
        parameters = discrete_categorical_params.copy()
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
        
        return parameters
    
    def _construct_full_parameters(self, continuous_params: np.ndarray) -> Dict[str, Any]:
        """从连续型参数构建完整的参数字典（向后兼容方法）"""
        return self._construct_full_parameters_with_combination(continuous_params, {})
    
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
        
        # 向后兼容性：确保新MoCo参数有合理的默认值
        moco_defaults = {
            'moco_tau1': 0.2,
            'moco_tau2': 0.3,
            'enable_view_0': 'true'
        }
        
        for param_name, default_value in moco_defaults.items():
            if param_name not in safe_params:
                safe_params[param_name] = default_value
        
        return safe_params
    
    def _debug_parameter_dimensions(self, parameters: Dict[str, Any]) -> None:
        """调试参数维度信息"""
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        try:
            feature_array = self._parameters_to_array(parameters)
            expected_dim = len(self.parameter_space.get_parameter_names())
            actual_dim = len(feature_array)
            
            debug_info = {
                "expected_dimension": expected_dim,
                "actual_dimension": actual_dim,
                "dimension_match": actual_dim == expected_dim,
                "parameter_space_params": sorted(self.parameter_space.get_parameter_names()),
                "input_params": sorted(parameters.keys()),
                "feature_array_shape": feature_array.shape if hasattr(feature_array, 'shape') else len(feature_array)
            }
            
            self.log_manager.log_structured(logging.DEBUG, "PARAM_DIMENSIONS", 
                                           debug_info, component_name)
            
            if actual_dim != expected_dim:
                missing_params = set(self.parameter_space.get_parameter_names()) - set(parameters.keys())
                extra_params = set(parameters.keys()) - set(self.parameter_space.get_parameter_names())
                
                dimension_mismatch = {
                    "missing_parameters": list(missing_params),
                    "extra_parameters": list(extra_params),
                    "dimension_difference": actual_dim - expected_dim
                }
                
                self.log_manager.log_structured(logging.WARNING, "DIMENSION_MISMATCH", 
                                               dimension_mismatch, component_name)
                
                if missing_params:
                    self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                                 f"缺失参数: {missing_params}", component_name)
                if extra_params:
                    self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                                 f"多余参数: {extra_params}", component_name)
                    
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"参数维度调试失败: {e}", component_name)
    
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
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        if len(self.history.results) < self.convergence_window:
            return False
        
        # 获取最近的目标值
        recent_values = [result.objective_value for result in self.history.results[-self.convergence_window:]]
        
        # 计算统计信息
        variance = np.var(recent_values)
        mean_value = np.mean(recent_values)
        min_value = np.min(recent_values)
        max_value = np.max(recent_values)
        range_value = max_value - min_value
        
        # 收敛检测分析
        convergence_analysis = {
            "window_size": self.convergence_window,
            "recent_values_count": len(recent_values),
            "variance": variance,
            "mean_value": mean_value,
            "min_value": min_value,
            "max_value": max_value,
            "range": range_value,
            "convergence_threshold": self.convergence_threshold,
            "variance_converged": variance < self.convergence_threshold ** 2,
            "no_improvement_count": self.no_improvement_count,
            "patience": self.patience,
            "patience_exceeded": self.no_improvement_count >= self.patience
        }
        
        self.log_manager.log_structured(logging.DEBUG, "CONVERGENCE_ANALYSIS", 
                                       convergence_analysis, component_name)
        
        # 检查方差是否足够小
        if variance < self.convergence_threshold ** 2:
            self.log_manager.log_with_tag(logging.INFO, "CONVERGENCE", 
                                         f"方差收敛检测: 方差({variance:.2e}) < 阈值²({self.convergence_threshold**2:.2e})", 
                                         component_name)
            return True
        
        # 检查是否长时间无改进
        if self.no_improvement_count >= self.patience:
            self.log_manager.log_with_tag(logging.INFO, "CONVERGENCE", 
                                         f"耐心收敛检测: 连续 {self.no_improvement_count} 次迭代无显著改进 >= 耐心值({self.patience})", 
                                         component_name)
            return True
        
        # 记录当前收敛状态
        if len(self.history.results) % 10 == 0:  # 每10次迭代记录一次
            convergence_status = {
                "variance_ratio": variance / (self.convergence_threshold ** 2),
                "patience_ratio": self.no_improvement_count / self.patience,
                "recent_improvement": max_value - min_value,
                "convergence_progress": min(variance / (self.convergence_threshold ** 2), 
                                          self.no_improvement_count / self.patience)
            }
            
            self.log_manager.log_structured(logging.DEBUG, "CONVERGENCE_STATUS", 
                                           convergence_status, component_name)
        
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
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        self.consecutive_failures += 1
        
        # 详细的错误信息
        error_info = {
            "iteration": self.current_iteration + 1,
            "consecutive_failures": self.consecutive_failures,
            "max_consecutive_failures": self.max_consecutive_failures,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_parameters_count": len(self.failed_parameters)
        }
        
        self.log_manager.log_structured(logging.ERROR, "ITERATION_ERROR", 
                                       error_info, component_name)
        
        self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                     f"第 {self.current_iteration + 1} 次迭代失败 "
                                     f"(连续失败 {self.consecutive_failures} 次): {str(error)}", 
                                     component_name)
        
        # 如果是参数相关错误，尝试调整采集函数参数
        if self.consecutive_failures >= 3:
            self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                         f"连续失败达到 {self.consecutive_failures} 次，尝试调整采集函数参数", 
                                         component_name)
            self._adjust_acquisition_function()
        
        # 记录错误恢复策略
        recovery_strategy = {
            "will_adjust_acquisition": self.consecutive_failures >= 3,
            "will_terminate": self.consecutive_failures >= self.max_consecutive_failures,
            "remaining_attempts": self.max_consecutive_failures - self.consecutive_failures
        }
        
        self.log_manager.log_structured(logging.INFO, "ERROR_RECOVERY", 
                                       recovery_strategy, component_name)
    
    def _adjust_acquisition_function(self):
        """调整采集函数参数以增加探索"""
        component_name = f"BayesianOptimizer_{self.task_evaluator.task_type}"
        
        self.log_manager.log_with_tag(logging.INFO, "ACQUISITION_ADJUST", 
                                     "开始调整采集函数参数以增加探索", 
                                     component_name)
        
        try:
            adjustment_made = False
            old_params = {}
            new_params = {}
            
            if hasattr(self.acquisition_function, 'xi'):
                # 增加EI或PI的探索参数
                old_xi = self.acquisition_function.xi
                new_xi = min(old_xi * 1.5, 0.1)
                self.acquisition_function.update_xi(new_xi)
                
                old_params['xi'] = old_xi
                new_params['xi'] = new_xi
                adjustment_made = True
                
            elif hasattr(self.acquisition_function, 'kappa'):
                # 增加UCB的探索参数
                old_kappa = self.acquisition_function.kappa
                new_kappa = min(old_kappa * 1.2, 5.0)
                self.acquisition_function.update_kappa(new_kappa)
                
                old_params['kappa'] = old_kappa
                new_params['kappa'] = new_kappa
                adjustment_made = True
            
            if adjustment_made:
                adjustment_info = {
                    "acquisition_function": self.acquisition_function.function_type,
                    "old_parameters": old_params,
                    "new_parameters": new_params,
                    "adjustment_reason": f"连续失败 {self.consecutive_failures} 次"
                }
                
                self.log_manager.log_structured(logging.INFO, "ACQUISITION_ADJUSTED", 
                                               adjustment_info, component_name)
                
                self.log_manager.log_with_tag(logging.INFO, "ACQUISITION_ADJUST", 
                                             f"采集函数参数已调整: {old_params} -> {new_params}", 
                                             component_name)
            else:
                self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                             f"采集函数 {self.acquisition_function.function_type} 不支持参数调整", 
                                             component_name)
                
        except Exception as e:
            self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                         f"调整采集函数参数失败: {e}", component_name)
    
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