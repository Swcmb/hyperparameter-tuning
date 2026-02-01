#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
贝叶斯超参数优化系统主入口脚本

本脚本集成了所有组件，提供完整的贝叶斯优化流程，支持：
- 命令行参数解析和配置文件加载
- 多种任务类型（LDA、MDA、LMI）的自动优化
- 多种采集函数（EI、PI、UCB）的选择
- 状态保存和恢复功能
- 实时进度监控和日志记录
- 结果分析和可视化报告生成
- 多目标优化支持
- MoCo（Momentum Contrast）对比学习参数优化

支持的MoCo参数：
- moco_momentum: MoCo动量更新系数 (0.9-0.9999)
- moco_t: MoCo基础温度系数 (0.01-1.0)
- moco_tau1: DoubleTau MoCo正样本温度系数 (0.01-1.0)
- moco_tau2: DoubleTau MoCo负样本温度系数 (0.01-1.0)，需满足 tau2 >= tau1
- moco_K: MoCo队列大小 [1024, 2048, 4096, 8192]
- moco_type: MoCo类型 ['basic', 'double_tau']
- enable_view_0: 是否启用MoCo第0视图 ['true', 'false']

MoCo参数约束：
- 动量系数应在合理范围内（0.9-0.9999）
- 所有温度参数必须为正值
- DoubleTau模式下，tau2必须大于等于tau1
- 队列大小与批次大小应保持合理比例
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心组件
from autodl_core import (
    create_default_parameter_space, OptimizationHistory, OptimizationResult
)
from bayesian_optimizer import create_bayesian_optimizer, create_multi_objective_optimizer
from task_evaluator import create_task_evaluator
from state_manager import create_default_state_manager
from result_analyzer import create_result_analyzer_from_checkpoint
from visualizer import create_visualizer_from_checkpoint
from report_generator import ReportGenerator, ReportConfig

# 导入新的输出系统
from unified_log_manager import UnifiedLogManager, get_global_log_manager, init_global_log_manager
from structured_tag_processor import get_global_tag_processor

import log_output_manager


class AutoDLOptimizer:
    """
    贝叶斯超参数优化系统主类
    
    集成所有组件，提供完整的优化流程
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化优化器
        
        Args:
            config: 配置字典，包含所有优化参数
        """
        self.config = config
        
        # 初始化统一日志管理器
        run_name = config.get('run_name', f"autodl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.log_manager = init_global_log_manager(
            run_name=run_name,
            log_level=config.get('log_level', logging.INFO),
            enable_console=True,
            enable_file=True,
            log_dir=config.get('log_dir', 'logs')
        )
        
        # 记录详细的初始化信息
        self._log_initialization_start(config)
        
        # 核心组件
        self.parameter_space = None
        self.optimizer = None
        self.task_evaluator = None
        self.state_manager = None
        self.history = None
        
        # 分析和报告组件
        self.result_analyzer = None
        self.visualizer = None
        self.report_generator = None
        
        # 运行状态
        self.start_time = None
        self.is_running = False
        self.current_iteration = 0
        
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "AutoDL优化器初始化完成", "AutoDLOptimizer")
    
    def _log_initialization_start(self, config: Dict[str, Any]):
        """记录简化的初始化开始信息"""
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "AutoDL贝叶斯超参数优化系统启动", 
                                     "AutoDLOptimizer")
        
        # 简化的系统信息输出
        import sys
        import torch
        
        # 只输出关键的系统信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                         f"GPU: {gpu_name}", 
                                         "AutoDLOptimizer")
        else:
            self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                         "CPU模式", 
                                         "AutoDLOptimizer")
        
        # 简化的配置信息
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     f"任务: {config.get('task_type', 'LDA')} | 迭代: {config.get('max_iterations', 50)}", 
                                     "AutoDLOptimizer")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('autodl')
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"autodl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def initialize_components(self):
        """初始化所有组件"""
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "开始初始化系统组件", 
                                     "AutoDLOptimizer")
        
        # 1. 创建参数空间
        task_type = self.config.get('task_type', 'LDA')
        self.parameter_space = create_default_parameter_space(task_type)
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     f"参数空间已创建 ({len(self.parameter_space.parameters)}个参数)", "AutoDLOptimizer")
        
        # 2. 创建任务评估器
        data_config = {
            'pos_file': self.config.get('pos_file'),
            'neg_file': self.config.get('neg_file')
        }
        
        # 如果数据配置为空，设置默认值
        if not data_config.get('pos_file') or not data_config.get('neg_file'):
            default_data = {
                "LDA": ("dataset1/LDA.edgelist", "dataset1/non_LDA.edgelist"),
                "MDA": ("dataset1/MDA.edgelist", "dataset1/non_MDA.edgelist"),
                "LMI": ("dataset1/LMI.edgelist", "dataset1/non_LMI.edgelist")
            }
            
            if task_type in default_data:
                data_config['pos_file'], data_config['neg_file'] = default_data[task_type]
        
        self.task_evaluator = create_task_evaluator(
            task_type=task_type,
            data_config=data_config,
            use_real_training=self.config.get('use_real_training', True)
        )
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "任务评估器已创建", "AutoDLOptimizer")
        
        # 3. 创建状态管理器
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        self.state_manager = create_default_state_manager(checkpoint_dir=checkpoint_dir)
        
        # 4. 创建优化器
        self._create_optimizer()
        optimizer_type = "多目标" if len(self.config.get('objectives', ['AUROC'])) > 1 else "单目标"
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     f"{optimizer_type}优化器已创建", "AutoDLOptimizer")
        
        # 5. 初始化历史记录
        self.history = self.optimizer.history
        
        # 6. 尝试恢复状态
        if self.config.get('resume', False):
            resume_success = self._try_resume_state()
            if resume_success:
                self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                             f"状态恢复成功，已完成 {self.current_iteration} 次迭代", "AutoDLOptimizer")
            else:
                self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                             "从头开始优化", "AutoDLOptimizer")
        
        self.log_manager.log_with_tag(logging.INFO, "INIT", 
                                     "系统初始化完成", 
                                     "AutoDLOptimizer")
    
    def _create_optimizer(self):
        """创建贝叶斯优化器"""
        # 检查是否为多目标优化
        objectives = self.config.get('objectives', ['AUROC'])
        task_type = self.config.get('task_type', 'LDA')
        
        # 准备数据配置
        data_config = {
            'pos_file': self.config.get('pos_file'),
            'neg_file': self.config.get('neg_file')
        }
        
        if len(objectives) > 1:
            # 多目标优化
            self.log_manager.log_with_tag(logging.INFO, "OPTIMIZER_CREATION", 
                                         f"创建多目标优化器，目标函数: {objectives}", "AutoDLOptimizer")
            
            maximize_objectives = {}
            for obj in objectives:
                # 默认所有目标都是最大化（AUROC、AUPRC、F1等）
                maximize_objectives[obj] = self.config.get(f'maximize_{obj.lower()}', True)
            
            objective_weights = self.config.get('objective_weights')
            
            self.optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=objectives,
                objective_weights=objective_weights,
                data_config=data_config,
                acquisition_function_type=self.config.get('acquisition_function', 'EI'),
                n_initial_points=self.config.get('n_initial_points', 10),
                random_state=self.config.get('random_seed', 42),
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints')
            )
            
            # 设置历史记录的多目标配置
            self.history.set_objectives(objectives, maximize_objectives, objective_weights)
            
        else:
            # 单目标优化
            self.log_manager.log_with_tag(logging.INFO, "OPTIMIZER_CREATION", 
                                         f"创建单目标优化器，目标函数: {objectives[0]}", "AutoDLOptimizer")
            
            self.optimizer = create_bayesian_optimizer(
                task_type=task_type,
                data_config=data_config,
                acquisition_function_type=self.config.get('acquisition_function', 'EI'),
                n_initial_points=self.config.get('n_initial_points', 10),
                random_state=self.config.get('random_seed', 42),
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints')
            )
    
    def _try_resume_state(self):
        """尝试恢复之前的优化状态"""
        try:
            checkpoint_name = self.config.get('checkpoint_name', 'latest')
            state_data = self.state_manager.load_state(checkpoint_name)
            
            if state_data:
                # 恢复历史记录
                if 'history' in state_data:
                    self.history = OptimizationHistory.from_dict(state_data['history'])
                    self.current_iteration = self.history.total_iterations
                    self.log_manager.log_with_tag(logging.INFO, "RESUME", 
                                                 f"恢复优化状态，已完成 {self.current_iteration} 次迭代", 
                                                 "AutoDLOptimizer")
                
                # 恢复优化器状态
                if 'optimizer_state' in state_data:
                    self.optimizer.load_state(state_data['optimizer_state'])
                    self.log_manager.log_with_tag(logging.INFO, "RESUME", 
                                                 "优化器状态恢复完成", "AutoDLOptimizer")
                
                return True
                
        except Exception as e:
            self.log_manager.log_with_tag(logging.WARNING, "RESUME", 
                                         f"状态恢复失败: {e}", "AutoDLOptimizer")
            self.log_manager.log_with_tag(logging.INFO, "RESUME", 
                                         "将从头开始优化", "AutoDLOptimizer")
        
        return False
    
    def run_optimization(self):
        """运行完整的贝叶斯优化流程"""
        self.log_manager.log_with_tag(logging.INFO, "OPTIMIZATION", 
                                     "开始贝叶斯超参数优化", 
                                     "AutoDLOptimizer")
        
        self.start_time = datetime.now()
        self.is_running = True
        
        # 简化的优化开始信息
        max_iterations = self.config.get('max_iterations', 50)
        self.log_manager.log_with_tag(logging.INFO, "OPTIMIZATION", 
                                     f"最大迭代: {max_iterations} | 目标: {self.config.get('objectives', ['AUROC'])[0]}", 
                                     "AutoDLOptimizer")
        
        # 初始化优化器
        if not self.optimizer.is_initialized:
            self.optimizer._initialize_optimization()
        
        try:
            max_time = self.config.get('max_time_hours', 24) * 3600  # 转换为秒
            
            # 主优化循环
            while (self.current_iteration < max_iterations and 
                   self.is_running and 
                   (time.time() - self.start_time.timestamp()) < max_time):
                
                self.current_iteration += 1
                
                self.log_manager.log_with_tag(logging.INFO, "ITERATION", 
                                             f"第 {self.current_iteration} 次迭代开始", 
                                             "AutoDLOptimizer")
                
                try:
                    # 获取下一个参数建议
                    suggested_params = self.optimizer.suggest_next_parameters()
                    formatted_params = self._format_parameters(suggested_params)
                    
                    self.log_manager.log_with_tag(logging.INFO, "ITERATION", 
                                                 f"建议参数: {formatted_params}", "AutoDLOptimizer")
                    
                    # 评估参数
                    evaluation_start = time.time()
                    metrics = self.task_evaluator.evaluate_parameters(suggested_params)
                    evaluation_time = time.time() - evaluation_start
                    
                    # 构建评估结果
                    evaluation_result = {
                        'objective_value': self.task_evaluator.get_objective_value(metrics),
                        'metrics': metrics,
                        'fold_results': None,
                        'objective_values': None
                    }
                    
                    # 如果是多目标优化，提取多目标值
                    if len(self.history.objectives) > 1:
                        evaluation_result['objective_values'] = self.task_evaluator.get_multi_objective_values(
                            metrics, self.history.objectives
                        )
                    
                    # 记录评估结果
                    self.log_manager.log_with_tag(logging.INFO, "EVALUATION", 
                                                 f"目标值: {evaluation_result['objective_value']:.6f} | AUROC: {metrics.get('AUROC', 0):.4f}", 
                                                 "AutoDLOptimizer")
                    
                    # 创建优化结果
                    result = OptimizationResult(
                        parameters=suggested_params,
                        objective_value=evaluation_result['objective_value'],
                        metrics=evaluation_result['metrics'],
                        iteration=self.current_iteration,
                        timestamp=datetime.now(),
                        evaluation_time=evaluation_time,
                        fold_results=evaluation_result.get('fold_results'),
                        objective_values=evaluation_result.get('objective_values')
                    )
                    
                    # 更新优化器和历史记录
                    self.optimizer.update_model(
                        parameters=suggested_params,
                        objective_value=evaluation_result['objective_value'],
                        metrics=evaluation_result['metrics'],
                        evaluation_time=evaluation_time,
                        objective_values=evaluation_result.get('objective_values')
                    )
                    self.history.add_result(result)
                    
                    # 记录迭代结果
                    self._log_iteration_result(result)
                    
                    # 保存状态
                    if self.current_iteration % self.config.get('save_frequency', 1) == 0:
                        self._save_state()
                    
                    # 检查收敛条件
                    if self._check_convergence():
                        self.log_manager.log_with_tag(logging.INFO, "CONVERGENCE", 
                                                     "检测到收敛，提前结束优化", "AutoDLOptimizer")
                        break
                        
                except Exception as e:
                    self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                                 f"第 {self.current_iteration} 次迭代失败: {e}", 
                                                 "AutoDLOptimizer")
                    # 记录错误但继续优化
                    continue
            
            # 优化完成
            self._finalize_optimization()
            
        except KeyboardInterrupt:
            self.log_manager.log_with_tag(logging.WARNING, "INTERRUPT", 
                                         "用户中断优化", "AutoDLOptimizer")
            self._save_state()
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"优化过程出现严重错误: {e}", "AutoDLOptimizer")
            raise
        finally:
            self.is_running = False
    
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """格式化参数显示"""
        formatted = []
        for key, value in params.items():
            if isinstance(value, float):
                if value < 0.001:
                    formatted.append(f"{key}={value:.2e}")
                else:
                    formatted.append(f"{key}={value:.4f}")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)
    
    def _log_iteration_result(self, result: OptimizationResult):
        """记录简化的迭代结果"""
        component_name = "AutoDLOptimizer"
        
        # 改进分析
        if self.history.best_result and len(self.history.results) > 1:
            previous_best = self.history.best_result.objective_value
            improvement = result.objective_value - previous_best
            
            if improvement > 0:
                self.log_manager.log_with_tag(logging.INFO, "SUCCESS", 
                                             f"发现新的最佳结果! 改进: {improvement:+.6f}", 
                                             component_name)
            else:
                self.log_manager.log_with_tag(logging.INFO, "ITERATION_RESULT", 
                                             f"当前最佳: {previous_best:.6f}, 本次: {result.objective_value:.6f}", 
                                             component_name)
        
        # 多目标优化分析
        if len(self.history.objectives) > 1:
            if getattr(result, 'is_pareto_optimal', False):
                self.log_manager.log_with_tag(logging.INFO, "SUCCESS", 
                                             "发现帕累托最优解!", component_name)
        
        self.log_manager.log_with_tag(logging.INFO, "ITERATION_RESULT", 
                                     f"第 {result.iteration} 次迭代完成 - 目标值: {result.objective_value:.6f}", 
                                     component_name)
    
    def _check_convergence(self) -> bool:
        """检查收敛条件"""
        if len(self.history.results) < 10:
            return False
        
        # 检查最近10次迭代的改进
        recent_results = self.history.results[-10:]
        recent_objectives = [r.objective_value for r in recent_results]
        
        improvement = max(recent_objectives) - min(recent_objectives)
        convergence_threshold = self.config.get('convergence_threshold', 0.001)
        
        if improvement < convergence_threshold:
            self.log_manager.log_with_tag(logging.INFO, "CONVERGENCE", 
                                         f"最近10次迭代改进 ({improvement:.6f}) 小于阈值 ({convergence_threshold})", 
                                         "AutoDLOptimizer")
            return True
        
        return False
    
    def _save_state(self):
        """保存当前优化状态"""
        try:
            state_data = {
                'history': self.history.to_dict() if self.history else {},
                'config': self.config,
                'current_iteration': self.current_iteration,
                'save_time': datetime.now().isoformat()
            }
            
            checkpoint_name = f"iteration_{self.current_iteration}"
            self.state_manager.save_state(state_data, checkpoint_name)
            
            # 同时保存为latest
            self.state_manager.save_state(state_data, "latest")
            
            self.log_manager.log_with_tag(logging.DEBUG, "CHECKPOINT", 
                                         f"状态已保存: {checkpoint_name}", "AutoDLOptimizer")
            
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"状态保存失败: {e}", "AutoDLOptimizer")
    
    def _finalize_optimization(self):
        """完成优化，生成最终报告"""
        component_name = "AutoDLOptimizer"
        
        self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                     "开始优化完成处理", 
                                     component_name)
        
        # 设置结束时间
        self.history.end_time = datetime.now()
        if self.history.start_time:
            self.history.total_time = (self.history.end_time - self.history.start_time).total_seconds()
        else:
            self.history.total_time = 0.0
        
        # 简化的完成统计
        self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                     f"总迭代: {self.current_iteration} | 运行时间: {self.history.total_time/3600:.2f}小时", 
                                     component_name)
        
        # 最佳结果分析
        if self.history.best_result:
            best_value = self.history.best_result.objective_value
            self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                         f"最佳目标值: {best_value:.6f} (第{self.history.best_result.iteration}次迭代)", 
                                         component_name)
            
            # 计算相对于第一次结果的改进
            if len(self.history.results) > 1:
                first_result = self.history.results[0].objective_value
                improvement = best_value - first_result
                improvement_percent = (improvement / abs(first_result)) * 100 if first_result != 0 else 0
                
                self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                             f"总体改进: {improvement:+.6f} ({improvement_percent:+.2f}%)", 
                                             component_name)
        
        # 多目标优化总结
        if len(self.history.objectives) > 1:
            try:
                pareto_size = len(self.history.pareto_front)
                self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                             f"帕累托前沿解数量: {pareto_size}", component_name)
            except Exception as e:
                self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                             f"多目标优化总结生成失败: {e}", component_name)
        
        # 保存最终状态
        try:
            self._save_state()
            self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                         "最终状态保存成功", component_name)
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"最终状态保存失败: {e}", component_name)
        
        # 生成分析报告
        if self.config.get('generate_report', True):
            try:
                self._generate_final_report()
                self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                             "分析报告生成完成", component_name)
            except Exception as e:
                self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                             f"报告生成失败: {e}", component_name)
        
        # 最终完成信息
        self.log_manager.log_with_tag(logging.INFO, "FINALIZATION", 
                                     "贝叶斯超参数优化完成", 
                                     component_name)
    
    def _generate_final_report(self):
        """生成最终分析报告"""
        component_name = "AutoDLOptimizer"
        
        self.log_manager.log_with_tag(logging.INFO, "REPORT_GENERATION", 
                                     "开始生成分析报告", 
                                     component_name)
        
        try:
            # 创建输出目录
            output_dir = Path(self.config.get('output_dir', 'results'))
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 创建结果分析器
            try:
                if checkpoint_path := self.state_manager.get_latest_checkpoint():
                    self.result_analyzer = create_result_analyzer_from_checkpoint(checkpoint_path)
                else:
                    from result_analyzer import ResultAnalyzer
                    self.result_analyzer = ResultAnalyzer(self.history, self.parameter_space)
            except Exception as e:
                from result_analyzer import ResultAnalyzer
                self.result_analyzer = ResultAnalyzer(self.history, self.parameter_space)
            
            # 创建可视化器
            try:
                if checkpoint_path := self.state_manager.get_latest_checkpoint():
                    from visualizer import create_visualizer_from_checkpoint
                    self.visualizer = create_visualizer_from_checkpoint(checkpoint_path)
                else:
                    from visualizer import Visualizer
                    self.visualizer = Visualizer(self.history, self.parameter_space)
            except Exception as e:
                from visualizer import Visualizer
                self.visualizer = Visualizer(self.history, self.parameter_space)
            
            # 创建报告生成器
            report_config = ReportConfig(
                title=f"{self.config.get('task_type', 'LDA')}任务贝叶斯超参数优化报告",
                author="AutoDL优化系统",
                include_charts=True,
                include_parameter_details=True,
                include_convergence_analysis=True,
                include_sensitivity_analysis=True
            )
            
            self.report_generator = ReportGenerator(
                optimization_history=self.history,
                parameter_space=self.parameter_space,
                result_analyzer=self.result_analyzer,
                visualizer=self.visualizer,
                config=report_config
            )
            
            # 生成JSON报告
            json_path = output_dir / f"optimization_report_{timestamp}.json"
            self.report_generator.save_json_report(str(json_path))
            self.log_manager.log_with_tag(logging.INFO, "REPORT_GENERATION", 
                                         f"JSON报告已保存: {json_path}", component_name)
            
            # 生成HTML报告
            if self.config.get('generate_html', True):
                html_path = output_dir / f"optimization_report_{timestamp}.html"
                self.report_generator.save_html_report(str(html_path))
                self.log_manager.log_with_tag(logging.INFO, "REPORT_GENERATION", 
                                             f"HTML报告已保存: {html_path}", component_name)
            
            # 生成可视化图表
            if self.config.get('generate_charts', True) and self.visualizer is not None:
                chart_dir = output_dir / f"charts_{timestamp}"
                chart_dir.mkdir(exist_ok=True)
                
                charts_generated = 0
                charts_failed = 0
                
                # 生成收敛曲线
                try:
                    convergence_path = str(chart_dir / "convergence.png")
                    self.visualizer.plot_convergence_curve(convergence_path)
                    charts_generated += 1
                except Exception as e:
                    charts_failed += 1
                    self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                                 f"收敛曲线生成失败: {e}", component_name)
                
                # 生成参数分布图
                try:
                    param_dist_path = str(chart_dir / "parameter_dist.png")
                    self.visualizer.plot_parameter_distributions(param_dist_path)
                    charts_generated += 1
                except Exception as e:
                    charts_failed += 1
                    self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                                 f"参数分布图生成失败: {e}", component_name)
                
                # 生成参数相关性热力图
                try:
                    param_corr_path = str(chart_dir / "parameter_corr.png")
                    self.visualizer.plot_parameter_correlation_heatmap(param_corr_path)
                    charts_generated += 1
                except Exception as e:
                    charts_failed += 1
                    self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                                 f"参数相关性图生成失败: {e}", component_name)
                
                # 多目标优化的帕累托前沿图
                if len(self.history.objectives) > 1:
                    try:
                        pareto_path = str(chart_dir / "pareto_front.png")
                        self.visualizer.plot_pareto_frontier(pareto_path)
                        charts_generated += 1
                    except Exception as e:
                        charts_failed += 1
                        self.log_manager.log_with_tag(logging.WARNING, "WARNING", 
                                                     f"帕累托前沿图生成失败: {e}", component_name)
                
                self.log_manager.log_with_tag(logging.INFO, "REPORT_GENERATION", 
                                             f"图表已保存: {chart_dir} (成功: {charts_generated}, 失败: {charts_failed})", 
                                             component_name)
            
        except Exception as e:
            self.log_manager.log_with_tag(logging.ERROR, "ERROR", 
                                         f"报告生成过程出现错误: {str(e)}", component_name)
            raise


def load_config_file(config_path: str) -> Dict[str, Any]:
    """从文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            else:
                # 支持Python配置文件
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                return config_module.config
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        return {}


def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        # 基本设置
        'task_type': 'LDA',
        'max_iterations': 50,
        'max_time_hours': 24,
        'random_seed': 42,
        
        # 优化设置
        'acquisition_function': 'EI',
        'acquisition_params': {},
        'objectives': ['AUROC'],
        'objective_weights': None,
        
        # 数据设置
        'data_path': None,
        'cv_folds': 5,
        
        # 状态管理
        'checkpoint_dir': 'checkpoints',
        'save_frequency': 1,
        'resume': False,
        'checkpoint_name': 'latest',
        
        # 收敛设置
        'convergence_threshold': 0.001,
        
        # 输出设置
        'output_dir': 'results',
        'log_dir': 'logs',
        'generate_report': True,
        'generate_html': True,
        'generate_charts': True
    }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='贝叶斯超参数优化系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python autodl.py --task_type LDA --max_iterations 30
  
  # 使用配置文件
  python autodl.py --config config.json
  
  # 多目标优化
  python autodl.py --objectives AUROC AUPRC F1 --objective_weights 0.5 0.3 0.2
  
  # 恢复之前的优化
  python autodl.py --resume --checkpoint_name iteration_20
  
  # 自定义采集函数
  python autodl.py --acquisition_function UCB --acquisition_params '{"beta": 2.0}'
  
  # MoCo参数优化（使用配置文件）
  python autodl.py --config moco_config.json
        """
    )
    
    # 基本参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--task_type', choices=['LDA', 'MDA', 'LMI'], 
                       help='任务类型')
    parser.add_argument('--max_iterations', type=int, 
                       help='最大迭代次数')
    parser.add_argument('--max_time_hours', type=float,
                       help='最大运行时间（小时）')
    parser.add_argument('--random_seed', type=int,
                       help='随机种子')
    
    # 优化参数
    parser.add_argument('--acquisition_function', 
                       choices=['EI', 'PI', 'UCB'],
                       help='采集函数类型')
    parser.add_argument('--acquisition_params', type=str,
                       help='采集函数参数（JSON格式）')
    parser.add_argument('--objectives', nargs='+',
                       help='目标函数列表（多目标优化）')
    parser.add_argument('--objective_weights', nargs='+', type=float,
                       help='目标函数权重')
    
    # 数据参数
    parser.add_argument('--data_path', type=str,
                       help='数据路径')
    parser.add_argument('--pos_file', type=str,
                       help='正样本文件路径')
    parser.add_argument('--neg_file', type=str,
                       help='负样本文件路径')
    parser.add_argument('--cv_folds', type=int,
                       help='交叉验证折数')
    
    # 状态管理
    parser.add_argument('--checkpoint_dir', type=str,
                       help='检查点目录')
    parser.add_argument('--save_frequency', type=int,
                       help='保存频率（每N次迭代）')
    parser.add_argument('--resume', action='store_true',
                       help='恢复之前的优化')
    parser.add_argument('--checkpoint_name', type=str,
                       help='要恢复的检查点名称')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str,
                       help='输出目录')
    parser.add_argument('--log_dir', type=str,
                       help='日志目录')
    parser.add_argument('--no_report', action='store_true',
                       help='不生成报告')
    parser.add_argument('--no_html', action='store_true',
                       help='不生成HTML报告')
    parser.add_argument('--no_charts', action='store_true',
                       help='不生成图表')
    

    
    return parser.parse_args()


def main():
    """主函数"""
    print("贝叶斯超参数优化系统 v1.0")
    print("=" * 60)
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建配置
    config = create_default_config()
    
    # 加载配置文件
    if args.config:
        file_config = load_config_file(args.config)
        config.update(file_config)
        print(f"已加载配置文件: {args.config}")
    
    # 应用命令行参数
    for key, value in vars(args).items():
        if value is not None:
            if key == 'acquisition_params' and isinstance(value, str):
                try:
                    config[key] = json.loads(value)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析采集函数参数: {value}")
            elif key in ['no_report', 'no_html', 'no_charts']:
                # 处理否定参数
                positive_key = key.replace('no_', 'generate_')
                if positive_key == 'generate_report':
                    config['generate_report'] = not value
                elif positive_key == 'generate_html':
                    config['generate_html'] = not value
                elif positive_key == 'generate_charts':
                    config['generate_charts'] = not value
            else:
                config[key] = value
    
    # 处理目标权重
    if args.objectives and args.objective_weights:
        if len(args.objectives) == len(args.objective_weights):
            config['objective_weights'] = dict(zip(args.objectives, args.objective_weights))
        else:
            print("警告: 目标函数数量与权重数量不匹配")
    
    # 显示配置信息
    print("\n当前配置:")
    for key, value in config.items():
        if key not in ['acquisition_params']:  # 跳过复杂参数
            print(f"  {key}: {value}")
    
    try:
        # 创建并运行优化器
        optimizer = AutoDLOptimizer(config)
        optimizer.initialize_components()
        optimizer.run_optimization()
        
        print("\n优化完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())