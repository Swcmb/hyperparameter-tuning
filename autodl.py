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
from log_output_manager import LogOutputManager


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
        self.logger = self._setup_logging()
        self.log_manager = LogOutputManager()
        
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
        
        self.logger.info("AutoDL优化器初始化完成")
    
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
        self.logger.info("开始初始化组件...")
        
        # 1. 创建参数空间
        task_type = self.config.get('task_type', 'LDA')
        self.parameter_space = create_default_parameter_space(task_type)
        self.logger.info(f"参数空间创建完成，包含 {len(self.parameter_space.parameters)} 个参数")
        
        # 2. 创建任务评估器
        self.task_evaluator = create_task_evaluator(
            task_type=task_type,
            data_path=self.config.get('data_path'),
            cv_folds=self.config.get('cv_folds', 5),
            random_seed=self.config.get('random_seed', 42)
        )
        self.logger.info(f"任务评估器创建完成，任务类型: {task_type}")
        
        # 3. 创建状态管理器
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        self.state_manager = create_default_state_manager(
            checkpoint_dir=checkpoint_dir,
            save_frequency=self.config.get('save_frequency', 1)
        )
        self.logger.info(f"状态管理器创建完成，检查点目录: {checkpoint_dir}")
        
        # 4. 创建优化器
        self._create_optimizer()
        
        # 5. 初始化历史记录
        self.history = OptimizationHistory()
        self.history.task_type = task_type
        self.history.acquisition_function = self.config.get('acquisition_function', 'EI')
        self.history.start_time = datetime.now()
        
        # 6. 尝试恢复状态
        if self.config.get('resume', False):
            self._try_resume_state()
        
        self.logger.info("所有组件初始化完成")
    
    def _create_optimizer(self):
        """创建贝叶斯优化器"""
        # 检查是否为多目标优化
        objectives = self.config.get('objectives', ['AUROC'])
        
        if len(objectives) > 1:
            # 多目标优化
            self.logger.info(f"创建多目标优化器，目标函数: {objectives}")
            
            maximize_objectives = {}
            for obj in objectives:
                # 默认所有目标都是最大化（AUROC、AUPRC、F1等）
                maximize_objectives[obj] = self.config.get(f'maximize_{obj.lower()}', True)
            
            objective_weights = self.config.get('objective_weights')
            
            self.optimizer = create_multi_objective_optimizer(
                parameter_space=self.parameter_space,
                task_evaluator=self.task_evaluator,
                objectives=objectives,
                maximize_objectives=maximize_objectives,
                objective_weights=objective_weights,
                acquisition_function=self.config.get('acquisition_function', 'EI'),
                acquisition_params=self.config.get('acquisition_params', {}),
                random_seed=self.config.get('random_seed', 42)
            )
            
            # 设置历史记录的多目标配置
            self.history.set_objectives(objectives, maximize_objectives, objective_weights)
            
        else:
            # 单目标优化
            self.logger.info(f"创建单目标优化器，目标函数: {objectives[0]}")
            
            self.optimizer = create_bayesian_optimizer(
                parameter_space=self.parameter_space,
                task_evaluator=self.task_evaluator,
                acquisition_function=self.config.get('acquisition_function', 'EI'),
                acquisition_params=self.config.get('acquisition_params', {}),
                random_seed=self.config.get('random_seed', 42)
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
                    self.logger.info(f"恢复优化状态，已完成 {self.current_iteration} 次迭代")
                
                # 恢复优化器状态
                if 'optimizer_state' in state_data:
                    self.optimizer.load_state(state_data['optimizer_state'])
                    self.logger.info("优化器状态恢复完成")
                
                return True
                
        except Exception as e:
            self.logger.warning(f"状态恢复失败: {e}")
            self.logger.info("将从头开始优化")
        
        return False
    
    def run_optimization(self):
        """运行完整的贝叶斯优化流程"""
        self.logger.info("开始贝叶斯超参数优化")
        self.start_time = datetime.now()
        self.is_running = True
        
        try:
            max_iterations = self.config.get('max_iterations', 50)
            max_time = self.config.get('max_time_hours', 24) * 3600  # 转换为秒
            
            self.logger.info(f"优化配置:")
            self.logger.info(f"  - 最大迭代次数: {max_iterations}")
            self.logger.info(f"  - 最大运行时间: {max_time/3600:.1f} 小时")
            self.logger.info(f"  - 任务类型: {self.config.get('task_type', 'LDA')}")
            self.logger.info(f"  - 采集函数: {self.config.get('acquisition_function', 'EI')}")
            
            # 主优化循环
            while (self.current_iteration < max_iterations and 
                   self.is_running and 
                   (time.time() - self.start_time.timestamp()) < max_time):
                
                iteration_start = time.time()
                self.current_iteration += 1
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"开始第 {self.current_iteration} 次迭代")
                self.logger.info(f"{'='*60}")
                
                try:
                    # 获取下一个参数建议
                    suggested_params = self.optimizer.suggest_parameters()
                    self.logger.info(f"建议参数: {self._format_parameters(suggested_params)}")
                    
                    # 评估参数
                    self.logger.info("开始参数评估...")
                    evaluation_result = self.task_evaluator.evaluate_parameters(suggested_params)
                    
                    # 创建优化结果
                    result = OptimizationResult(
                        parameters=suggested_params,
                        objective_value=evaluation_result['objective_value'],
                        metrics=evaluation_result['metrics'],
                        iteration=self.current_iteration,
                        timestamp=datetime.now(),
                        evaluation_time=time.time() - iteration_start,
                        fold_results=evaluation_result.get('fold_results'),
                        objective_values=evaluation_result.get('objective_values')
                    )
                    
                    # 更新优化器和历史记录
                    self.optimizer.update_with_result(result)
                    self.history.add_result(result)
                    
                    # 记录结果
                    self._log_iteration_result(result)
                    
                    # 保存状态
                    if self.current_iteration % self.config.get('save_frequency', 1) == 0:
                        self._save_state()
                    
                    # 检查收敛条件
                    if self._check_convergence():
                        self.logger.info("检测到收敛，提前结束优化")
                        break
                        
                except Exception as e:
                    self.logger.error(f"第 {self.current_iteration} 次迭代失败: {e}")
                    # 记录错误但继续优化
                    continue
            
            # 优化完成
            self._finalize_optimization()
            
        except KeyboardInterrupt:
            self.logger.info("用户中断优化")
            self._save_state()
        except Exception as e:
            self.logger.error(f"优化过程出现严重错误: {e}")
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
        """记录迭代结果"""
        self.logger.info(f"评估完成:")
        self.logger.info(f"  - 主要目标值: {result.objective_value:.4f}")
        self.logger.info(f"  - 评估时间: {result.evaluation_time:.1f} 秒")
        
        # 显示所有指标
        for metric, value in result.metrics.items():
            self.logger.info(f"  - {metric}: {value:.4f}")
        
        # 显示当前最佳结果
        if self.history.best_result:
            improvement = result.objective_value - self.history.best_result.objective_value
            if improvement > 0:
                self.logger.info(f"  ✓ 新的最佳结果! 改进: +{improvement:.4f}")
            else:
                self.logger.info(f"  - 当前最佳: {self.history.best_result.objective_value:.4f}")
        
        # 多目标优化信息
        if len(self.history.objectives) > 1:
            pareto_size = len(self.history.pareto_front)
            self.logger.info(f"  - 帕累托前沿大小: {pareto_size}")
            if result.is_pareto_optimal:
                self.logger.info(f"  ✓ 帕累托最优解!")
    
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
            self.logger.info(f"最近10次迭代改进 ({improvement:.6f}) 小于阈值 ({convergence_threshold})")
            return True
        
        return False
    
    def _save_state(self):
        """保存当前优化状态"""
        try:
            state_data = {
                'history': self.history.to_dict(),
                'optimizer_state': self.optimizer.get_state(),
                'config': self.config,
                'current_iteration': self.current_iteration,
                'save_time': datetime.now().isoformat()
            }
            
            checkpoint_name = f"iteration_{self.current_iteration}"
            self.state_manager.save_state(state_data, checkpoint_name)
            
            # 同时保存为latest
            self.state_manager.save_state(state_data, "latest")
            
            self.logger.debug(f"状态已保存: {checkpoint_name}")
            
        except Exception as e:
            self.logger.error(f"状态保存失败: {e}")
    
    def _finalize_optimization(self):
        """完成优化，生成最终报告"""
        self.history.end_time = datetime.now()
        self.history.total_time = (self.history.end_time - self.history.start_time).total_seconds()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("贝叶斯优化完成")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"总迭代次数: {self.current_iteration}")
        self.logger.info(f"总运行时间: {self.history.total_time/3600:.2f} 小时")
        
        if self.history.best_result:
            self.logger.info(f"最佳目标值: {self.history.best_result.objective_value:.4f}")
            self.logger.info(f"最佳参数: {self._format_parameters(self.history.best_result.parameters)}")
        
        # 多目标优化总结
        if len(self.history.objectives) > 1:
            pareto_metrics = self.history.get_pareto_front_metrics()
            self.logger.info(f"帕累托前沿大小: {pareto_metrics.get('front_size', 0)}")
            self.logger.info(f"覆盖率: {pareto_metrics.get('coverage_ratio', 0):.2%}")
        
        # 保存最终状态
        self._save_state()
        
        # 生成分析报告
        if self.config.get('generate_report', True):
            self._generate_final_report()
    
    def _generate_final_report(self):
        """生成最终分析报告"""
        try:
            self.logger.info("开始生成分析报告...")
            
            # 创建结果分析器
            self.result_analyzer = create_result_analyzer_from_checkpoint(
                checkpoint_path=None,  # 直接使用历史数据
                history=self.history,
                parameter_space=self.parameter_space
            )
            
            # 创建可视化器
            self.visualizer = create_visualizer_from_checkpoint(
                checkpoint_path=None,
                history=self.history,
                parameter_space=self.parameter_space
            )
            
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
                history=self.history,
                parameter_space=self.parameter_space,
                result_analyzer=self.result_analyzer,
                visualizer=self.visualizer,
                config=report_config
            )
            
            # 生成报告
            output_dir = Path(self.config.get('output_dir', 'results'))
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON报告
            json_path = output_dir / f"optimization_report_{timestamp}.json"
            self.report_generator.generate_json_report(str(json_path))
            self.logger.info(f"JSON报告已保存: {json_path}")
            
            # HTML报告
            if self.config.get('generate_html', True):
                html_path = output_dir / f"optimization_report_{timestamp}.html"
                self.report_generator.generate_html_report(str(html_path))
                self.logger.info(f"HTML报告已保存: {html_path}")
            
            # 可视化图表
            if self.config.get('generate_charts', True):
                chart_dir = output_dir / f"charts_{timestamp}"
                chart_dir.mkdir(exist_ok=True)
                
                # 生成各种图表
                self.visualizer.plot_convergence_curve(str(chart_dir / "convergence.png"))
                self.visualizer.plot_parameter_distribution(str(chart_dir / "parameter_dist.png"))
                self.visualizer.plot_parameter_correlation(str(chart_dir / "parameter_corr.png"))
                
                if len(self.history.objectives) > 1:
                    self.visualizer.plot_pareto_front(str(chart_dir / "pareto_front.png"))
                
                self.logger.info(f"可视化图表已保存: {chart_dir}")
            
            self.logger.info("分析报告生成完成")
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")


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