"""
报告生成器（ReportGenerator）

本模块实现了贝叶斯超参数优化结果的详细报告生成功能，包括：
- 实验配置和参数空间信息
- 最佳参数组合和性能指标
- 统计分析和参数敏感性分析
- 收敛性分析和优化历史
- 多种输出格式支持（JSON、HTML、PDF）
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
from datetime import datetime
from pathlib import Path
import warnings
from dataclasses import dataclass, asdict
import base64
import io

# 导入核心组件
from autodl_core import OptimizationHistory, OptimizationResult, ParameterSpace
from result_analyzer import ResultAnalyzer, ParameterSensitivityResult, ConvergenceAnalysisResult, StatisticalSummary
from visualizer import Visualizer

# 可选依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 配置中文字体支持和禁用字体警告
    import warnings
    import platform
    
    # 禁用字体相关的警告
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
    
    # 尝试配置中文字体
    try:
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun'] + plt.rcParams['font.sans-serif']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti'] + plt.rcParams['font.sans-serif']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
        
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        pass  # 如果字体配置失败，继续使用默认字体
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib未安装，将无法生成图表")

try:
    from weasyprint import HTML, CSS
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False
    warnings.warn("weasyprint未安装，将无法生成PDF报告")

try:
    from jinja2 import Template, Environment, FileSystemLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    warnings.warn("jinja2未安装，将使用简单的HTML模板")


@dataclass
class ReportConfig:
    """报告配置"""
    title: str = "贝叶斯超参数优化报告"
    author: str = "AutoDL系统"
    include_charts: bool = True
    include_parameter_details: bool = True
    include_convergence_analysis: bool = True
    include_sensitivity_analysis: bool = True
    include_best_parameters_analysis: bool = True
    chart_dpi: int = 300
    max_parameters_in_charts: int = 15
    language: str = "zh"  # 支持中文和英文


class ReportGenerator:
    """
    报告生成器
    
    生成包含实验配置、最佳参数、性能指标和统计分析的详细优化报告，
    支持JSON、HTML、PDF等多种输出格式。
    """
    
    def __init__(self, optimization_history: OptimizationHistory,
                 parameter_space: Optional[ParameterSpace] = None,
                 result_analyzer: Optional[ResultAnalyzer] = None,
                 visualizer: Optional[Visualizer] = None,
                 config: Optional[ReportConfig] = None):
        """
        初始化报告生成器
        
        Args:
            optimization_history: 优化历史记录
            parameter_space: 参数空间定义（可选）
            result_analyzer: 结果分析器（可选，如果不提供会自动创建）
            visualizer: 可视化器（可选，如果不提供会自动创建）
            config: 报告配置（可选）
        """
        self.history = optimization_history
        self.parameter_space = parameter_space
        self.config = config or ReportConfig()
        
        # 创建分析器和可视化器
        if result_analyzer is None:
            self.analyzer = ResultAnalyzer(optimization_history, parameter_space)
        else:
            self.analyzer = result_analyzer
            
        if visualizer is None and HAS_MATPLOTLIB:
            self.visualizer = Visualizer(optimization_history, parameter_space, self.analyzer)
        else:
            self.visualizer = visualizer
        
        # 报告数据缓存
        self._report_data: Optional[Dict[str, Any]] = None
        self._charts_data: Optional[Dict[str, str]] = None
    
    def generate_report_data(self) -> Dict[str, Any]:
        """
        生成完整的报告数据
        
        Returns:
            包含所有报告内容的字典
        """
        if self._report_data is not None:
            return self._report_data
        
        print("正在生成报告数据...")
        
        # 基本信息
        report_data = {
            'metadata': self._generate_metadata(),
            'experiment_configuration': self._generate_experiment_config(),
            'optimization_summary': self._generate_optimization_summary(),
            'statistical_summary': self._generate_statistical_summary(),
            'best_parameters': self._generate_best_parameters_section(),
            'convergence_analysis': self._generate_convergence_analysis(),
            'parameter_analysis': self._generate_parameter_analysis(),
            'performance_metrics': self._generate_performance_metrics(),
            'optimization_history': self._generate_optimization_history(),
            'recommendations': self._generate_recommendations()
        }
        
        # 添加图表数据（如果启用）
        if self.config.include_charts and self.visualizer:
            report_data['charts'] = self._generate_charts_data()
        
        self._report_data = report_data
        return report_data
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """生成报告元数据"""
        return {
            'title': self.config.title,
            'author': self.config.author,
            'generation_time': datetime.now().isoformat(),
            'language': self.config.language,
            'version': '1.0.0',
            'system_info': {
                'task_type': self.history.task_type,
                'acquisition_function': self.history.acquisition_function,
                'total_iterations': self.history.total_iterations,
                'total_time': self.history.total_time
            }
        }
    
    def _generate_experiment_config(self) -> Dict[str, Any]:
        """生成实验配置信息"""
        config = {
            'task_type': self.history.task_type,
            'acquisition_function': self.history.acquisition_function,
            'optimization_start_time': self.history.start_time.isoformat() if self.history.start_time else None,
            'optimization_end_time': self.history.end_time.isoformat() if self.history.end_time else None,
            'total_duration': self.history.total_time,
            'parameter_space_summary': {}
        }
        
        # 参数空间摘要
        if self.parameter_space:
            config['parameter_space_summary'] = {
                'total_parameters': self.parameter_space.get_parameter_count(),
                'continuous_parameters': len(self.parameter_space.get_continuous_parameter_names()),
                'discrete_parameters': len(self.parameter_space.get_discrete_parameter_names()),
                'categorical_parameters': len(self.parameter_space.get_categorical_parameter_names()),
                'parameter_details': {}
            }
            
            # 详细参数信息
            if self.config.include_parameter_details:
                for param_name, param_config in self.parameter_space.parameters.items():
                    config['parameter_space_summary']['parameter_details'][param_name] = {
                        'type': param_config.param_type.value,
                        'bounds': param_config.bounds,
                        'values': param_config.values,
                        'log_scale': param_config.log_scale,
                        'constraints': param_config.constraints
                    }
        
        return config
    
    def _generate_optimization_summary(self) -> Dict[str, Any]:
        """生成优化过程摘要"""
        summary = {
            'total_evaluations': len(self.history.results),
            'successful_evaluations': len([r for r in self.history.results if r.error_info is None]),
            'failed_evaluations': len([r for r in self.history.results if r.error_info is not None]),
            'success_rate': 0.0,
            'best_objective_value': None,
            'best_iteration': None,
            'improvement_over_baseline': None,
            'evaluation_times': {
                'total_time': sum(r.evaluation_time for r in self.history.results),
                'average_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0
            }
        }
        
        if self.history.results:
            successful_results = [r for r in self.history.results if r.error_info is None]
            summary['success_rate'] = len(successful_results) / len(self.history.results)
            
            if self.history.best_result:
                summary['best_objective_value'] = self.history.best_result.objective_value
                summary['best_iteration'] = self.history.best_result.iteration
                
                # 计算相对于初始结果的改进
                if len(self.history.results) > 1:
                    initial_value = self.history.results[0].objective_value
                    summary['improvement_over_baseline'] = (
                        self.history.best_result.objective_value - initial_value
                    ) / abs(initial_value) if initial_value != 0 else 0
            
            # 评估时间统计
            eval_times = [r.evaluation_time for r in self.history.results]
            summary['evaluation_times'].update({
                'average_time': sum(eval_times) / len(eval_times),
                'min_time': min(eval_times),
                'max_time': max(eval_times)
            })
        
        return summary
    
    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """生成统计摘要"""
        stats = self.analyzer.get_statistical_summary()
        return {
            'total_evaluations': stats.total_evaluations,
            'objective_statistics': {
                'best': stats.best_objective_value,
                'worst': stats.worst_objective_value,
                'mean': stats.mean_objective_value,
                'std': stats.std_objective_value,
                'median': stats.median_objective_value,
                'q25': stats.q25_objective_value,
                'q75': stats.q75_objective_value
            },
            'time_statistics': {
                'total_time': stats.total_time,
                'average_evaluation_time': stats.average_evaluation_time
            },
            'success_rate': stats.success_rate
        }
    
    def _generate_best_parameters_section(self) -> Dict[str, Any]:
        """生成最佳参数部分"""
        section = {
            'best_single_result': None,
            'top_k_analysis': None,
            'parameter_recommendations': {}
        }
        
        if self.history.best_result:
            section['best_single_result'] = {
                'iteration': self.history.best_result.iteration,
                'objective_value': self.history.best_result.objective_value,
                'parameters': self.history.best_result.parameters,
                'metrics': self.history.best_result.metrics,
                'evaluation_time': self.history.best_result.evaluation_time,
                'timestamp': self.history.best_result.timestamp.isoformat()
            }
        
        # 前k个最佳结果分析
        if self.config.include_best_parameters_analysis:
            top_k_analysis = self.analyzer.get_best_parameters_analysis(top_k=10)
            section['top_k_analysis'] = top_k_analysis
            
            # 参数推荐
            if 'parameter_statistics' in top_k_analysis:
                for param_name, stats in top_k_analysis['parameter_statistics'].items():
                    if stats['type'] == 'categorical':
                        section['parameter_recommendations'][param_name] = {
                            'type': 'categorical',
                            'recommended_value': stats['most_frequent'],
                            'confidence': stats['frequency'] / top_k_analysis['top_k'],
                            'reasoning': f"在前{top_k_analysis['top_k']}个最佳结果中出现{stats['frequency']}次"
                        }
                    else:
                        section['parameter_recommendations'][param_name] = {
                            'type': 'numerical',
                            'recommended_range': [
                                max(stats['mean'] - stats['std'], stats['min']),
                                min(stats['mean'] + stats['std'], stats['max'])
                            ],
                            'optimal_value': stats['mean'],
                            'reasoning': f"基于前{top_k_analysis['top_k']}个最佳结果的统计分析"
                        }
        
        return section
    
    def _generate_convergence_analysis(self) -> Dict[str, Any]:
        """生成收敛性分析"""
        if not self.config.include_convergence_analysis:
            return {}
        
        convergence_result = self.analyzer.analyze_convergence()
        
        return {
            'is_converged': convergence_result.is_converged,
            'convergence_iteration': convergence_result.convergence_iteration,
            'convergence_threshold': convergence_result.convergence_threshold,
            'improvement_rate': convergence_result.improvement_rate,
            'plateau_length': convergence_result.plateau_length,
            'final_improvement': convergence_result.final_improvement,
            'convergence_curve': self.history.get_convergence_curve(),
            'analysis_summary': self._generate_convergence_summary(convergence_result)
        }
    
    def _generate_convergence_summary(self, convergence_result: ConvergenceAnalysisResult) -> str:
        """生成收敛性分析摘要文本"""
        if convergence_result.is_converged:
            return (f"优化过程在第{convergence_result.convergence_iteration}次迭代后收敛，"
                   f"最终改进幅度为{convergence_result.final_improvement:.4f}，"
                   f"平台期长度为{convergence_result.plateau_length}次迭代。")
        else:
            return (f"优化过程尚未收敛，当前改进速率为{convergence_result.improvement_rate:.6f}/迭代，"
                   f"建议继续优化或调整收敛阈值。")
    
    def _generate_parameter_analysis(self) -> Dict[str, Any]:
        """生成参数分析"""
        if not self.config.include_sensitivity_analysis:
            return {}
        
        sensitivity_results = self.analyzer.analyze_parameter_sensitivity()
        importance_ranking = self.analyzer.get_parameter_importance_ranking()
        correlation_matrix = self.analyzer.analyze_parameter_correlations()
        
        analysis = {
            'parameter_sensitivity': [],
            'importance_ranking': importance_ranking,
            'parameter_correlations': correlation_matrix.to_dict() if not correlation_matrix.empty else {},
            'sensitivity_summary': self._generate_sensitivity_summary(sensitivity_results)
        }
        
        # 参数敏感性详细结果
        for result in sensitivity_results:
            analysis['parameter_sensitivity'].append({
                'parameter_name': result.parameter_name,
                'sensitivity_score': result.sensitivity_score,
                'correlation_coefficient': result.correlation_coefficient,
                'p_value': result.p_value,
                'mutual_information': result.mutual_information,
                'importance_rank': result.importance_rank,
                'analysis_method': result.analysis_method,
                'significance': 'high' if result.sensitivity_score > 0.1 else 
                              'medium' if result.sensitivity_score > 0.05 else 'low'
            })
        
        return analysis
    
    def _generate_sensitivity_summary(self, sensitivity_results: List[ParameterSensitivityResult]) -> str:
        """生成敏感性分析摘要文本"""
        if not sensitivity_results:
            return "未找到显著的参数敏感性。"
        
        high_impact = [r for r in sensitivity_results if r.sensitivity_score > 0.1]
        medium_impact = [r for r in sensitivity_results if 0.05 < r.sensitivity_score <= 0.1]
        
        summary = f"分析了{len(sensitivity_results)}个参数的敏感性。"
        
        if high_impact:
            high_names = [r.parameter_name for r in high_impact]
            summary += f"高影响参数（{len(high_impact)}个）：{', '.join(high_names)}。"
        
        if medium_impact:
            medium_names = [r.parameter_name for r in medium_impact]
            summary += f"中等影响参数（{len(medium_impact)}个）：{', '.join(medium_names)}。"
        
        most_important = sensitivity_results[0]
        summary += f"最重要的参数是{most_important.parameter_name}（敏感性得分：{most_important.sensitivity_score:.4f}）。"
        
        return summary
    
    def _generate_performance_metrics(self) -> Dict[str, Any]:
        """生成性能指标分析"""
        metrics_analysis = {
            'primary_metric': 'AUROC',
            'all_metrics': {},
            'metric_correlations': {},
            'best_metrics': {}
        }
        
        if not self.history.results:
            return metrics_analysis
        
        # 收集所有指标
        all_metric_names = set()
        for result in self.history.results:
            if result.metrics:
                all_metric_names.update(result.metrics.keys())
        
        # 分析每个指标
        for metric_name in all_metric_names:
            metric_values = []
            for result in self.history.results:
                if result.metrics and metric_name in result.metrics:
                    metric_values.append(result.metrics[metric_name])
            
            if metric_values:
                metrics_analysis['all_metrics'][metric_name] = {
                    'count': len(metric_values),
                    'mean': sum(metric_values) / len(metric_values),
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'std': (sum((x - sum(metric_values)/len(metric_values))**2 for x in metric_values) / len(metric_values))**0.5
                }
        
        # 最佳结果的指标
        if self.history.best_result and self.history.best_result.metrics:
            metrics_analysis['best_metrics'] = self.history.best_result.metrics
        
        return metrics_analysis
    
    def _generate_optimization_history(self) -> Dict[str, Any]:
        """生成优化历史"""
        history_data = {
            'total_iterations': len(self.history.results),
            'results_summary': [],
            'error_analysis': {
                'total_errors': 0,
                'error_types': {},
                'error_rate_by_iteration': []
            }
        }
        
        # 结果摘要（只包含关键信息）
        for i, result in enumerate(self.history.results[-20:], start=max(0, len(self.history.results)-20)):  # 只显示最后20个结果
            summary = {
                'iteration': result.iteration,
                'objective_value': result.objective_value,
                'evaluation_time': result.evaluation_time,
                'has_error': result.error_info is not None,
                'is_best': result == self.history.best_result
            }
            
            # 添加主要参数（前5个最重要的）
            if self.config.include_parameter_details:
                importance_ranking = self.analyzer.get_parameter_importance_ranking()
                top_params = [name for name, _ in importance_ranking[:5]]
                summary['key_parameters'] = {
                    param: result.parameters.get(param) for param in top_params 
                    if param in result.parameters
                }
            
            history_data['results_summary'].append(summary)
        
        # 错误分析
        error_count = 0
        for result in self.history.results:
            if result.error_info:
                error_count += 1
                # 简单的错误分类
                error_type = 'unknown'
                if 'timeout' in result.error_info.lower():
                    error_type = 'timeout'
                elif 'memory' in result.error_info.lower():
                    error_type = 'memory'
                elif 'cuda' in result.error_info.lower():
                    error_type = 'cuda'
                
                history_data['error_analysis']['error_types'][error_type] = (
                    history_data['error_analysis']['error_types'].get(error_type, 0) + 1
                )
        
        history_data['error_analysis']['total_errors'] = error_count
        
        return history_data
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """生成优化建议"""
        recommendations = {
            'parameter_tuning': [],
            'optimization_strategy': [],
            'next_steps': []
        }
        
        # 基于收敛性分析的建议
        convergence_result = self.analyzer.analyze_convergence()
        if not convergence_result.is_converged:
            if convergence_result.improvement_rate > 0.001:
                recommendations['optimization_strategy'].append(
                    "优化过程仍在改进中，建议继续运行更多迭代。"
                )
            else:
                recommendations['optimization_strategy'].append(
                    "改进速率较低，建议调整采集函数参数或尝试不同的采集策略。"
                )
        else:
            recommendations['optimization_strategy'].append(
                f"优化已在第{convergence_result.convergence_iteration}次迭代收敛，可以停止优化。"
            )
        
        # 基于参数敏感性的建议
        sensitivity_results = self.analyzer.analyze_parameter_sensitivity()
        high_impact_params = [r for r in sensitivity_results if r.sensitivity_score > 0.1]
        low_impact_params = [r for r in sensitivity_results if r.sensitivity_score < 0.02]
        
        if high_impact_params:
            param_names = [r.parameter_name for r in high_impact_params[:3]]
            recommendations['parameter_tuning'].append(
                f"重点关注高影响参数：{', '.join(param_names)}，这些参数对性能影响最大。"
            )
        
        if low_impact_params:
            param_names = [r.parameter_name for r in low_impact_params[:3]]
            recommendations['parameter_tuning'].append(
                f"参数{', '.join(param_names)}对性能影响较小，可以固定为默认值以减少搜索空间。"
            )
        
        # 基于成功率的建议
        stats = self.analyzer.get_statistical_summary()
        if stats.success_rate < 0.9:
            recommendations['optimization_strategy'].append(
                f"当前成功率为{stats.success_rate:.2%}，建议检查参数约束和评估函数的稳定性。"
            )
        
        # 下一步建议
        if self.history.best_result:
            best_value = self.history.best_result.objective_value
            if best_value < 0.8:
                recommendations['next_steps'].append(
                    "当前最佳性能较低，建议检查数据质量、模型架构或尝试不同的特征工程方法。"
                )
            elif best_value > 0.95:
                recommendations['next_steps'].append(
                    "已获得很好的性能，建议进行模型集成或在更大数据集上验证结果。"
                )
            else:
                recommendations['next_steps'].append(
                    "性能良好，可以考虑进一步精调高影响参数或尝试多目标优化。"
                )
        
        return recommendations
    
    def _generate_charts_data(self) -> Dict[str, str]:
        """生成图表数据（Base64编码的图片）"""
        if self._charts_data is not None:
            return self._charts_data
        
        if not self.visualizer or not HAS_MATPLOTLIB:
            return {}
        
        print("正在生成图表...")
        charts = {}
        
        try:
            # 1. 收敛曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            convergence_curve = self.history.get_convergence_curve()
            iterations = list(range(1, len(convergence_curve) + 1))
            ax.plot(iterations, convergence_curve, 'b-', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('目标函数值')
            ax.set_title('收敛曲线')
            ax.grid(True, alpha=0.3)
            charts['convergence_curve'] = self._fig_to_base64(fig)
            plt.close(fig)
            
            # 2. 参数重要性
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if importance_ranking:
                top_params = importance_ranking[:min(10, len(importance_ranking))]
                param_names = [name for name, _ in top_params]
                importance_scores = [score for _, score in top_params]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(param_names, importance_scores, alpha=0.8)
                ax.set_xlabel('重要性得分')
                ax.set_title('参数重要性排序')
                ax.grid(True, alpha=0.3, axis='x')
                
                # 添加数值标签
                for bar, score in zip(bars, importance_scores):
                    ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{score:.3f}', va='center', fontsize=9)
                
                charts['parameter_importance'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 3. 目标函数值分布
            obj_values = [r.objective_value for r in self.history.results]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(obj_values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('目标函数值')
            ax.set_ylabel('频次')
            ax.set_title('目标函数值分布')
            ax.grid(True, alpha=0.3)
            charts['objective_distribution'] = self._fig_to_base64(fig)
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成图表时出错: {e}")
        
        self._charts_data = charts
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图形转换为Base64编码的字符串"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.config.chart_dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64
    
    def save_json_report(self, filepath: str) -> None:
        """
        保存JSON格式报告
        
        Args:
            filepath: 保存路径
        """
        report_data = self.generate_report_data()
        
        # 移除图表数据（JSON中不需要）
        json_data = report_data.copy()
        if 'charts' in json_data:
            del json_data['charts']
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"JSON报告已保存到: {filepath}")
    
    def save_html_report(self, filepath: str) -> None:
        """
        保存HTML格式报告
        
        Args:
            filepath: 保存路径
        """
        report_data = self.generate_report_data()
        
        if HAS_JINJA2:
            html_content = self._generate_html_with_jinja2(report_data)
        else:
            html_content = self._generate_simple_html(report_data)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已保存到: {filepath}")
    
    def save_pdf_report(self, filepath: str) -> None:
        """
        保存PDF格式报告
        
        Args:
            filepath: 保存路径
        """
        if not HAS_WEASYPRINT:
            raise ImportError("需要安装weasyprint才能生成PDF报告: pip install weasyprint")
        
        # 先生成HTML
        html_filepath = filepath.replace('.pdf', '_temp.html')
        self.save_html_report(html_filepath)
        
        try:
            # 转换为PDF
            HTML(filename=html_filepath).write_pdf(filepath)
            print(f"PDF报告已保存到: {filepath}")
            
            # 删除临时HTML文件
            if os.path.exists(html_filepath):
                os.remove(html_filepath)
                
        except Exception as e:
            warnings.warn(f"生成PDF时出错: {e}")
            print(f"HTML版本已保存到: {html_filepath}")
    
    def _generate_html_with_jinja2(self, report_data: Dict[str, Any]) -> str:
        """使用Jinja2模板生成HTML"""
        template_str = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #2c3e50; border-left: 4px solid #3498db; padding-left: 10px; }
        .section h3 { color: #34495e; margin-top: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 3px solid #3498db; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; font-size: 0.9em; }
        .parameter-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .parameter-table th, .parameter-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .parameter-table th { background-color: #f2f2f2; }
        .chart-container { text-align: center; margin: 20px 0; }
        .chart-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .recommendation { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #27ae60; }
        .warning { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }
        .error { background: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #dc3545; }
        .footer { text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ metadata.title }}</h1>
        <p>生成时间: {{ metadata.generation_time[:19] }} | 作者: {{ metadata.author }}</p>
    </div>

    <!-- 实验配置 -->
    <div class="section">
        <h2>实验配置</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ experiment_configuration.task_type }}</div>
                <div class="metric-label">任务类型</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ experiment_configuration.acquisition_function }}</div>
                <div class="metric-label">采集函数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ experiment_configuration.parameter_space_summary.total_parameters }}</div>
                <div class="metric-label">参数总数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:.2f}".format(experiment_configuration.total_duration) }}s</div>
                <div class="metric-label">总耗时</div>
            </div>
        </div>
    </div>

    <!-- 优化摘要 -->
    <div class="section">
        <h2>优化摘要</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ optimization_summary.total_evaluations }}</div>
                <div class="metric-label">总评估次数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:.4f}".format(optimization_summary.best_objective_value or 0) }}</div>
                <div class="metric-label">最佳目标值</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ optimization_summary.best_iteration or 0 }}</div>
                <div class="metric-label">最佳迭代</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:.2%}".format(optimization_summary.success_rate) }}</div>
                <div class="metric-label">成功率</div>
            </div>
        </div>
    </div>

    <!-- 最佳参数 -->
    {% if best_parameters.best_single_result %}
    <div class="section">
        <h2>最佳参数组合</h2>
        <h3>最佳结果详情</h3>
        <p><strong>迭代次数:</strong> {{ best_parameters.best_single_result.iteration }}</p>
        <p><strong>目标函数值:</strong> {{ "{:.6f}".format(best_parameters.best_single_result.objective_value) }}</p>
        <p><strong>评估时间:</strong> {{ "{:.2f}".format(best_parameters.best_single_result.evaluation_time) }}秒</p>
        
        <h3>参数值</h3>
        <table class="parameter-table">
            <tr><th>参数名</th><th>值</th></tr>
            {% for param_name, param_value in best_parameters.best_single_result.parameters.items() %}
            <tr><td>{{ param_name }}</td><td>{{ param_value }}</td></tr>
            {% endfor %}
        </table>
        
        {% if best_parameters.best_single_result.metrics %}
        <h3>性能指标</h3>
        <div class="metric-grid">
            {% for metric_name, metric_value in best_parameters.best_single_result.metrics.items() %}
            <div class="metric-card">
                <div class="metric-value">{{ "{:.4f}".format(metric_value) }}</div>
                <div class="metric-label">{{ metric_name }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- 收敛性分析 -->
    {% if convergence_analysis %}
    <div class="section">
        <h2>收敛性分析</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "是" if convergence_analysis.is_converged else "否" }}</div>
                <div class="metric-label">是否收敛</div>
            </div>
            {% if convergence_analysis.convergence_iteration %}
            <div class="metric-card">
                <div class="metric-value">{{ convergence_analysis.convergence_iteration }}</div>
                <div class="metric-label">收敛迭代</div>
            </div>
            {% endif %}
            <div class="metric-card">
                <div class="metric-value">{{ "{:.6f}".format(convergence_analysis.improvement_rate) }}</div>
                <div class="metric-label">改进速率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:.4f}".format(convergence_analysis.final_improvement) }}</div>
                <div class="metric-label">最终改进</div>
            </div>
        </div>
        <p>{{ convergence_analysis.analysis_summary }}</p>
    </div>
    {% endif %}

    <!-- 参数分析 -->
    {% if parameter_analysis.parameter_sensitivity %}
    <div class="section">
        <h2>参数敏感性分析</h2>
        <p>{{ parameter_analysis.sensitivity_summary }}</p>
        
        <h3>参数重要性排序</h3>
        <table class="parameter-table">
            <tr><th>排名</th><th>参数名</th><th>敏感性得分</th><th>相关系数</th><th>显著性</th></tr>
            {% for param in parameter_analysis.parameter_sensitivity[:10] %}
            <tr>
                <td>{{ param.importance_rank }}</td>
                <td>{{ param.parameter_name }}</td>
                <td>{{ "{:.4f}".format(param.sensitivity_score) }}</td>
                <td>{{ "{:.4f}".format(param.correlation_coefficient) }}</td>
                <td>{{ param.significance }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    <!-- 图表 -->
    {% if charts %}
    <div class="section">
        <h2>可视化分析</h2>
        
        {% if charts.convergence_curve %}
        <div class="chart-container">
            <h3>收敛曲线</h3>
            <img src="data:image/png;base64,{{ charts.convergence_curve }}" alt="收敛曲线">
        </div>
        {% endif %}
        
        {% if charts.parameter_importance %}
        <div class="chart-container">
            <h3>参数重要性</h3>
            <img src="data:image/png;base64,{{ charts.parameter_importance }}" alt="参数重要性">
        </div>
        {% endif %}
        
        {% if charts.objective_distribution %}
        <div class="chart-container">
            <h3>目标函数值分布</h3>
            <img src="data:image/png;base64,{{ charts.objective_distribution }}" alt="目标函数值分布">
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- 优化建议 -->
    {% if recommendations %}
    <div class="section">
        <h2>优化建议</h2>
        
        {% if recommendations.parameter_tuning %}
        <h3>参数调优建议</h3>
        {% for rec in recommendations.parameter_tuning %}
        <div class="recommendation">{{ rec }}</div>
        {% endfor %}
        {% endif %}
        
        {% if recommendations.optimization_strategy %}
        <h3>优化策略建议</h3>
        {% for rec in recommendations.optimization_strategy %}
        <div class="recommendation">{{ rec }}</div>
        {% endfor %}
        {% endif %}
        
        {% if recommendations.next_steps %}
        <h3>下一步建议</h3>
        {% for rec in recommendations.next_steps %}
        <div class="recommendation">{{ rec }}</div>
        {% endfor %}
        {% endif %}
    </div>
    {% endif %}

    <div class="footer">
        <p>报告由AutoDL贝叶斯优化系统自动生成 | 版本: {{ metadata.version }}</p>
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        return template.render(**report_data)
    
    def _generate_simple_html(self, report_data: Dict[str, Any]) -> str:
        """生成简单的HTML报告（不使用Jinja2）"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='zh-CN'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<title>{}</title>".format(report_data['metadata']['title']),
            "<style>",
            "body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }",
            ".section { margin-bottom: 30px; }",
            ".section h2 { color: #2c3e50; border-left: 4px solid #3498db; padding-left: 10px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>{}</h1>".format(report_data['metadata']['title']),
            "<p>生成时间: {} | 作者: {}</p>".format(
                report_data['metadata']['generation_time'][:19],
                report_data['metadata']['author']
            )
        ]
        
        # 添加各个部分
        sections = [
            ('实验配置', 'experiment_configuration'),
            ('优化摘要', 'optimization_summary'),
            ('最佳参数', 'best_parameters'),
            ('收敛性分析', 'convergence_analysis'),
            ('参数分析', 'parameter_analysis'),
            ('优化建议', 'recommendations')
        ]
        
        for section_title, section_key in sections:
            if section_key in report_data and report_data[section_key]:
                html_parts.append(f"<div class='section'><h2>{section_title}</h2>")
                html_parts.append(f"<pre>{json.dumps(report_data[section_key], ensure_ascii=False, indent=2, default=str)}</pre>")
                html_parts.append("</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def generate_all_formats(self, output_dir: str, base_filename: str = "optimization_report") -> None:
        """
        生成所有格式的报告
        
        Args:
            output_dir: 输出目录
            base_filename: 基础文件名
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"正在生成所有格式的报告到: {output_dir}")
        
        # JSON报告
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        self.save_json_report(json_path)
        
        # HTML报告
        html_path = os.path.join(output_dir, f"{base_filename}.html")
        self.save_html_report(html_path)
        
        # PDF报告（如果支持）
        if HAS_WEASYPRINT:
            pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
            try:
                self.save_pdf_report(pdf_path)
            except Exception as e:
                warnings.warn(f"PDF生成失败: {e}")
        
        print("所有格式报告生成完成!")


def create_report_generator_from_checkpoint(checkpoint_path: str, 
                                          config: Optional[ReportConfig] = None) -> Optional[ReportGenerator]:
    """
    从检查点文件创建报告生成器
    
    Args:
        checkpoint_path: 检查点文件路径
        config: 报告配置
        
    Returns:
        报告生成器实例，如果加载失败则返回None
    """
    try:
        from state_manager import StateManager
        
        # 加载状态数据
        state_manager = StateManager()
        state_data = state_manager.load_state(checkpoint_path)
        
        if 'optimization_history' not in state_data:
            warnings.warn("检查点文件中没有优化历史数据")
            return None
        
        # 创建优化历史对象
        history = OptimizationHistory.from_dict(state_data['optimization_history'])
        
        # 创建参数空间对象（如果存在）
        parameter_space = None
        if 'parameter_space' in state_data:
            parameter_space = ParameterSpace.from_dict(state_data['parameter_space'])
        
        return ReportGenerator(history, parameter_space, config=config)
        
    except Exception as e:
        warnings.warn(f"从检查点创建报告生成器失败: {e}")
        return None


if __name__ == "__main__":
    # 测试代码
    print("测试报告生成器...")
    
    # 创建模拟数据
    from autodl_core import create_default_parameter_space
    import numpy as np
    
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    history.start_time = datetime.now()
    
    # 添加模拟优化结果
    np.random.seed(42)
    for i in range(50):
        params = parameter_space.sample_random_parameters(seed=42+i)
        
        # 模拟目标函数值
        obj_value = 0.7 + 0.2 * np.random.random()
        if params.get('lr', 0.001) < 0.001:
            obj_value += 0.05
        if params.get('dimensions', 256) > 300:
            obj_value += 0.03
        
        result = OptimizationResult(
            parameters=params,
            objective_value=obj_value,
            metrics={'AUROC': obj_value, 'AUPRC': obj_value - 0.02, 'F1': obj_value - 0.05},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=120.0 + 30 * np.random.random(),
            error_info=None if np.random.random() > 0.1 else "模拟错误"
        )
        
        history.add_result(result)
    
    history.end_time = datetime.now()
    history.total_time = 6000.0
    
    # 创建报告生成器
    config = ReportConfig(
        title="测试优化报告",
        author="测试系统",
        include_charts=True
    )
    
    generator = ReportGenerator(history, parameter_space, config=config)
    
    print(f"创建了包含 {len(history.results)} 个结果的报告生成器")
    
    # 测试报告生成
    output_dir = "test_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成JSON报告...")
    generator.save_json_report(os.path.join(output_dir, "test_report.json"))
    
    print("生成HTML报告...")
    generator.save_html_report(os.path.join(output_dir, "test_report.html"))
    
    if HAS_WEASYPRINT:
        print("生成PDF报告...")
        try:
            generator.save_pdf_report(os.path.join(output_dir, "test_report.pdf"))
        except Exception as e:
            print(f"PDF生成失败: {e}")
    
    print("生成所有格式报告...")
    generator.generate_all_formats(output_dir, "comprehensive_report")
    
    print(f"报告生成器测试完成! 结果保存在: {output_dir}")