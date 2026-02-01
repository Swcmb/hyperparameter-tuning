"""
增强报告生成器（Enhanced Report Generator）

修复图表生成问题，确保：
1. 图表保存为单独的文件
2. 生成热力图
3. 图表中的文字使用英文
4. 提供完整的可视化分析
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import base64
import io

# 导入核心组件
from autodl_core import OptimizationHistory, OptimizationResult, ParameterSpace
from result_analyzer import ResultAnalyzer
from visualizer import Visualizer
from report_generator import ReportGenerator, ReportConfig

# 可选依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 配置英文字体和样式
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib未安装，将无法生成图表")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("seaborn未安装，热力图功能受限")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("numpy未安装，数值计算功能受限")


class EnhancedReportGenerator(ReportGenerator):
    """
    增强报告生成器
    
    在原有功能基础上增加：
    - 图表保存为单独文件
    - 生成热力图
    - 英文图表标签
    - 完整的可视化分析
    """
    
    def __init__(self, optimization_history: OptimizationHistory,
                 parameter_space: Optional[ParameterSpace] = None,
                 result_analyzer: Optional[ResultAnalyzer] = None,
                 visualizer: Optional[Visualizer] = None,
                 config: Optional[ReportConfig] = None):
        """
        初始化增强报告生成器
        """
        super().__init__(optimization_history, parameter_space, result_analyzer, visualizer, config)
        
        # 图表保存目录
        self.charts_dir: Optional[str] = None
        self.chart_files: Dict[str, str] = {}
    
    def generate_enhanced_report(self, output_dir: str, base_filename: str = "optimization_report") -> None:
        """
        生成增强版报告，包含单独的图表文件
        
        Args:
            output_dir: 输出目录
            base_filename: 基础文件名
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图表子目录
        self.charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
        
        print(f"正在生成增强版报告到: {output_dir}")
        
        # 1. 生成所有图表文件
        self._generate_all_charts()
        
        # 2. 生成报告数据
        report_data = self.generate_report_data()
        
        # 3. 添加图表文件路径信息
        report_data['chart_files'] = self.chart_files
        
        # 4. 生成各种格式的报告
        self._save_enhanced_json_report(output_dir, base_filename, report_data)
        self._save_enhanced_html_report(output_dir, base_filename, report_data)
        
        print("增强版报告生成完成!")
        print(f"报告文件: {output_dir}")
        print(f"图表文件: {self.charts_dir}")
    
    def _generate_all_charts(self) -> None:
        """生成所有图表并保存为单独文件"""
        if not self.visualizer or not HAS_MATPLOTLIB:
            print("跳过图表生成：matplotlib或visualizer不可用")
            return
        
        print("正在生成图表文件...")
        
        try:
            # 1. 收敛曲线
            print("  - 生成收敛曲线...")
            convergence_path = os.path.join(self.charts_dir, "convergence_curve.png")
            self._plot_convergence_curve_english(convergence_path)
            self.chart_files['convergence_curve'] = "charts/convergence_curve.png"
            
            # 2. 参数重要性
            print("  - 生成参数重要性图...")
            importance_path = os.path.join(self.charts_dir, "parameter_importance.png")
            self._plot_parameter_importance_english(importance_path)
            self.chart_files['parameter_importance'] = "charts/parameter_importance.png"
            
            # 3. 目标函数值分布
            print("  - 生成目标函数值分布...")
            distribution_path = os.path.join(self.charts_dir, "objective_distribution.png")
            self._plot_objective_distribution_english(distribution_path)
            self.chart_files['objective_distribution'] = "charts/objective_distribution.png"
            
            # 4. 参数相关性热力图
            print("  - 生成参数相关性热力图...")
            correlation_path = os.path.join(self.charts_dir, "parameter_correlation_heatmap.png")
            self._plot_parameter_correlation_heatmap_english(correlation_path)
            self.chart_files['parameter_correlation_heatmap'] = "charts/parameter_correlation_heatmap.png"
            
            # 5. 性能热力图
            print("  - 生成性能热力图...")
            performance_heatmap_path = os.path.join(self.charts_dir, "performance_heatmap.png")
            self._plot_performance_heatmap_english(performance_heatmap_path)
            self.chart_files['performance_heatmap'] = "charts/performance_heatmap.png"
            
            # 6. 参数分布图
            print("  - 生成参数分布图...")
            param_dist_path = os.path.join(self.charts_dir, "parameter_distributions.png")
            self._plot_parameter_distributions_english(param_dist_path)
            self.chart_files['parameter_distributions'] = "charts/parameter_distributions.png"
            
            # 7. 参数演化图
            print("  - 生成参数演化图...")
            evolution_path = os.path.join(self.charts_dir, "parameter_evolution.png")
            self._plot_parameter_evolution_english(evolution_path)
            self.chart_files['parameter_evolution'] = "charts/parameter_evolution.png"
            
            # 8. 3D优化景观图（如果可能）
            print("  - 生成3D优化景观图...")
            landscape_path = os.path.join(self.charts_dir, "optimization_landscape_3d.png")
            self._plot_optimization_landscape_3d_english(landscape_path)
            self.chart_files['optimization_landscape_3d'] = "charts/optimization_landscape_3d.png"
            
        except Exception as e:
            warnings.warn(f"生成图表时出错: {e}")
    
    def _plot_convergence_curve_english(self, save_path: str) -> None:
        """生成英文版收敛曲线"""
        try:
            convergence_curve = self.history.get_convergence_curve()
            if not convergence_curve:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            iterations = list(range(1, len(convergence_curve) + 1))
            
            ax.plot(iterations, convergence_curve, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Objective Value')
            ax.set_title('Convergence Curve')
            ax.grid(True, alpha=0.3)
            
            # 添加最佳值标注
            if self.history.best_result:
                best_iter = self.history.best_result.iteration
                best_value = self.history.best_result.objective_value
                ax.axhline(y=best_value, color='r', linestyle='--', alpha=0.7, 
                          label=f'Best: {best_value:.4f} at iter {best_iter}')
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成收敛曲线失败: {e}")
    
    def _plot_parameter_importance_english(self, save_path: str) -> None:
        """生成英文版参数重要性图"""
        try:
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if not importance_ranking:
                return
            
            # 取前10个最重要的参数
            top_params = importance_ranking[:min(10, len(importance_ranking))]
            param_names = [name for name, _ in top_params]
            importance_scores = [score for _, score in top_params]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(param_names, importance_scores, alpha=0.8, color='skyblue')
            ax.set_xlabel('Importance Score')
            ax.set_title('Parameter Importance Ranking')
            ax.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for bar, score in zip(bars, importance_scores):
                ax.text(score + max(importance_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成参数重要性图失败: {e}")
    
    def _plot_objective_distribution_english(self, save_path: str) -> None:
        """生成英文版目标函数值分布图"""
        try:
            obj_values = [r.objective_value for r in self.history.results if r.objective_value is not None]
            if not obj_values:
                return
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(obj_values, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
            ax.set_xlabel('Objective Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Objective Value Distribution')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(obj_values)
            std_val = np.std(obj_values)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.4f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8, 
                      label=f'Mean + Std: {mean_val + std_val:.4f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8, 
                      label=f'Mean - Std: {mean_val - std_val:.4f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成目标函数值分布图失败: {e}")
    
    def _plot_parameter_correlation_heatmap_english(self, save_path: str) -> None:
        """生成英文版参数相关性热力图"""
        try:
            if not HAS_SEABORN:
                warnings.warn("seaborn未安装，跳过相关性热力图")
                return
            
            correlation_matrix = self.analyzer.analyze_parameter_correlations()
            if correlation_matrix.empty or len(correlation_matrix.columns) < 2:
                warnings.warn("参数数据不足，无法生成相关性热力图")
                return
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 创建遮罩，只显示下三角
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # 绘制热力图
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                       fmt='.2f', annot_kws={'size': 8})
            
            ax.set_title('Parameter Correlation Heatmap')
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Parameters')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成参数相关性热力图失败: {e}")
    
    def _plot_performance_heatmap_english(self, save_path: str) -> None:
        """生成英文版性能热力图"""
        try:
            # 获取最重要的两个参数
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if len(importance_ranking) < 2:
                warnings.warn("参数数量不足，无法生成性能热力图")
                return
            
            param1 = importance_ranking[0][0]
            param2 = importance_ranking[1][0]
            
            # 准备数据
            results_df = self.visualizer.results_df
            param1_col = f'param_{param1}'
            param2_col = f'param_{param2}'
            
            if param1_col not in results_df.columns or param2_col not in results_df.columns:
                warnings.warn(f"参数 {param1} 或 {param2} 数据不存在")
                return
            
            x_values = results_df[param1_col].values
            y_values = results_df[param2_col].values
            z_values = results_df['objective_value'].values
            
            # 判断参数类型
            param1_is_categorical = (results_df[param1_col].dtype == 'object' or 
                                   results_df[param1_col].nunique() <= 10)
            param2_is_categorical = (results_df[param2_col].dtype == 'object' or 
                                   results_df[param2_col].nunique() <= 10)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if param1_is_categorical and param2_is_categorical:
                # 两个都是分类参数：创建分组热力图
                df_pivot = results_df.pivot_table(
                    values='objective_value', 
                    index=param2_col, 
                    columns=param1_col, 
                    aggfunc='mean'
                )
                
                if HAS_SEABORN:
                    sns.heatmap(df_pivot, annot=True, cmap='viridis', ax=ax, 
                               fmt='.3f', cbar_kws={'label': 'Average Objective Value'})
                else:
                    im = ax.imshow(df_pivot.values, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, label='Average Objective Value')
                    ax.set_xticks(range(len(df_pivot.columns)))
                    ax.set_yticks(range(len(df_pivot.index)))
                    ax.set_xticklabels(df_pivot.columns)
                    ax.set_yticklabels(df_pivot.index)
                
            elif not param1_is_categorical and not param2_is_categorical:
                # 两个都是连续参数：插值热力图
                try:
                    from scipy.interpolate import griddata
                    
                    # 创建网格
                    xi = np.linspace(x_values.min(), x_values.max(), 50)
                    yi = np.linspace(y_values.min(), y_values.max(), 50)
                    xi, yi = np.meshgrid(xi, yi)
                    
                    # 插值
                    zi = griddata((x_values, y_values), z_values, (xi, yi), method='cubic')
                    
                    # 绘制热力图
                    im = ax.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.8)
                    
                    # 添加原始数据点
                    scatter = ax.scatter(x_values, y_values, c=z_values, cmap='viridis', 
                                       s=50, edgecolors='white', linewidth=0.5)
                    
                    # 添加颜色条
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Objective Value')
                    
                except ImportError:
                    # 如果没有scipy，使用简单的散点图
                    scatter = ax.scatter(x_values, y_values, c=z_values, cmap='viridis', 
                                       s=50, alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label='Objective Value')
            
            else:
                # 一个分类一个连续：分组散点图
                if param1_is_categorical:
                    cat_param, cont_param = param1, param2
                    cat_values, cont_values = x_values, y_values
                    ax.set_xlabel(param2)
                    ax.set_ylabel('Objective Value')
                else:
                    cat_param, cont_param = param2, param1
                    cat_values, cont_values = y_values, x_values
                    ax.set_xlabel(param1)
                    ax.set_ylabel('Objective Value')
                
                # 为每个类别绘制散点
                unique_cats = np.unique(cat_values)
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_cats)))
                
                for i, cat in enumerate(unique_cats):
                    mask = cat_values == cat
                    ax.scatter(cont_values[mask], z_values[mask], 
                              c=[colors[i]], label=f'{cat_param}={cat}', 
                              alpha=0.7, s=50)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            if not (param1_is_categorical and not param2_is_categorical):
                ax.set_xlabel(param1)
            if not (not param1_is_categorical and param2_is_categorical):
                ax.set_ylabel(param2)
            
            ax.set_title(f'Performance Heatmap: {param1} vs {param2}')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成性能热力图失败: {e}")
    
    def _plot_parameter_distributions_english(self, save_path: str) -> None:
        """生成英文版参数分布图"""
        try:
            # 获取重要参数
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if not importance_ranking:
                return
            
            # 选择前6个最重要的参数
            top_params = [name for name, _ in importance_ranking[:6]]
            results_df = self.visualizer.results_df
            
            # 计算子图布局
            n_params = len(top_params)
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_params == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, param_name in enumerate(top_params):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                param_col = f'param_{param_name}'
                if param_col not in results_df.columns:
                    continue
                
                param_values = results_df[param_col].dropna()
                obj_values = results_df.loc[param_values.index, 'objective_value']
                
                if param_values.dtype == 'object' or param_values.nunique() <= 10:
                    # 分类参数：箱线图
                    unique_vals = param_values.unique()
                    box_data = [obj_values[param_values == val].values for val in unique_vals]
                    ax.boxplot(box_data, labels=unique_vals)
                    ax.set_xticklabels(unique_vals, rotation=45)
                else:
                    # 连续参数：散点图
                    ax.scatter(param_values, obj_values, alpha=0.6, s=30)
                
                ax.set_xlabel(param_name)
                ax.set_ylabel('Objective Value')
                ax.set_title(f'Distribution: {param_name}')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(n_params, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成参数分布图失败: {e}")
    
    def _plot_parameter_evolution_english(self, save_path: str) -> None:
        """生成英文版参数演化图"""
        try:
            # 获取重要参数
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if not importance_ranking:
                return
            
            # 选择前4个最重要的参数
            top_params = [name for name, _ in importance_ranking[:4]]
            results_df = self.visualizer.results_df
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, param_name in enumerate(top_params):
                if i >= 4:
                    break
                
                ax = axes[i]
                param_col = f'param_{param_name}'
                
                if param_col not in results_df.columns:
                    continue
                
                param_values = results_df[param_col].values
                iterations = results_df['iteration'].values
                obj_values = results_df['objective_value'].values
                
                if results_df[param_col].dtype == 'object':
                    # 分类参数：用不同颜色表示不同类别
                    unique_vals = results_df[param_col].unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
                    
                    for j, val in enumerate(unique_vals):
                        mask = param_values == val
                        ax.scatter(iterations[mask], obj_values[mask], 
                                 c=[colors[j]], label=str(val), alpha=0.7, s=30)
                    
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # 连续参数：颜色映射
                    scatter = ax.scatter(iterations, obj_values, c=param_values, 
                                       cmap='viridis', alpha=0.7, s=30)
                    plt.colorbar(scatter, ax=ax, label=param_name)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Objective Value')
                ax.set_title(f'Parameter Evolution: {param_name}')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(len(top_params), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成参数演化图失败: {e}")
    
    def _plot_optimization_landscape_3d_english(self, save_path: str) -> None:
        """生成英文版3D优化景观图"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # 获取最重要的两个参数
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if len(importance_ranking) < 2:
                return
            
            param1 = importance_ranking[0][0]
            param2 = importance_ranking[1][0]
            
            results_df = self.visualizer.results_df
            param1_col = f'param_{param1}'
            param2_col = f'param_{param2}'
            
            if param1_col not in results_df.columns or param2_col not in results_df.columns:
                return
            
            # 只处理连续参数
            if (results_df[param1_col].dtype == 'object' or 
                results_df[param2_col].dtype == 'object'):
                return
            
            x_values = results_df[param1_col].values
            y_values = results_df[param2_col].values
            z_values = results_df['objective_value'].values
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 创建3D散点图
            scatter = ax.scatter(x_values, y_values, z_values, 
                               c=z_values, cmap='viridis', s=50, alpha=0.7)
            
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_zlabel('Objective Value')
            ax.set_title(f'3D Optimization Landscape: {param1} vs {param2}')
            
            # 添加颜色条
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Objective Value')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"生成3D优化景观图失败: {e}")
    
    def _save_enhanced_json_report(self, output_dir: str, base_filename: str, report_data: Dict[str, Any]) -> None:
        """保存增强版JSON报告"""
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        
        # 移除图表数据（JSON中不需要base64数据）
        json_data = report_data.copy()
        if 'charts' in json_data:
            del json_data['charts']
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"增强版JSON报告已保存到: {json_path}")
    
    def _save_enhanced_html_report(self, output_dir: str, base_filename: str, report_data: Dict[str, Any]) -> None:
        """保存增强版HTML报告，引用单独的图表文件"""
        html_path = os.path.join(output_dir, f"{base_filename}.html")
        
        html_content = self._generate_enhanced_html(report_data)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"增强版HTML报告已保存到: {html_path}")
    
    def _generate_enhanced_html(self, report_data: Dict[str, Any]) -> str:
        """生成增强版HTML，使用单独的图表文件"""
        
        # 获取数据
        metadata = report_data.get('metadata', {})
        experiment_config = report_data.get('experiment_configuration', {})
        optimization_summary = report_data.get('optimization_summary', {})
        best_parameters = report_data.get('best_parameters', {})
        convergence_analysis = report_data.get('convergence_analysis', {})
        parameter_analysis = report_data.get('parameter_analysis', {})
        recommendations = report_data.get('recommendations', {})
        chart_files = report_data.get('chart_files', {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata.get('title', '优化报告')}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #2c3e50; border-left: 4px solid #3498db; padding-left: 10px; }}
        .section h3 {{ color: #34495e; margin-top: 20px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 3px solid #3498db; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .parameter-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .parameter-table th, .parameter-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .parameter-table th {{ background-color: #f2f2f2; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .recommendation {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #27ae60; }}
        .warning {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        .error {{ background: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #dc3545; }}
        .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{metadata.get('title', '优化报告')}</h1>
        <p>生成时间: {metadata.get('generation_time', '')[:19]} | 作者: {metadata.get('author', '')}</p>
    </div>

    <!-- 实验配置 -->
    <div class="section">
        <h2>实验配置</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{experiment_config.get('task_type', 'N/A')}</div>
                <div class="metric-label">任务类型</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{experiment_config.get('acquisition_function', 'N/A')}</div>
                <div class="metric-label">采集函数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{experiment_config.get('parameter_space_summary', {}).get('total_parameters', 0)}</div>
                <div class="metric-label">参数总数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{experiment_config.get('total_duration', 0):.2f}s</div>
                <div class="metric-label">总耗时</div>
            </div>
        </div>
    </div>

    <!-- 优化摘要 -->
    <div class="section">
        <h2>优化摘要</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{optimization_summary.get('total_evaluations', 0)}</div>
                <div class="metric-label">总评估次数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{optimization_summary.get('best_objective_value', 0) or 0:.4f}</div>
                <div class="metric-label">最佳目标值</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{optimization_summary.get('best_iteration', 0)}</div>
                <div class="metric-label">最佳迭代</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{optimization_summary.get('success_rate', 0):.2%}</div>
                <div class="metric-label">成功率</div>
            </div>
        </div>
    </div>
"""
        
        # 最佳参数部分
        if best_parameters.get('best_single_result'):
            best_result = best_parameters['best_single_result']
            html_content += f"""
    <!-- 最佳参数 -->
    <div class="section">
        <h2>最佳参数组合</h2>
        <h3>最佳结果详情</h3>
        <p><strong>迭代次数:</strong> {best_result.get('iteration', 0)}</p>
        <p><strong>目标函数值:</strong> {best_result.get('objective_value', 0):.6f}</p>
        <p><strong>评估时间:</strong> {best_result.get('evaluation_time', 0):.2f}秒</p>
        
        <h3>参数值</h3>
        <table class="parameter-table">
            <tr><th>参数名</th><th>值</th></tr>
"""
            for param_name, param_value in best_result.get('parameters', {}).items():
                html_content += f"            <tr><td>{param_name}</td><td>{param_value}</td></tr>\n"
            
            html_content += "        </table>\n"
            
            # 性能指标
            if best_result.get('metrics'):
                html_content += """
        <h3>性能指标</h3>
        <div class="metric-grid">
"""
                for metric_name, metric_value in best_result['metrics'].items():
                    html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{metric_value:.4f}</div>
                <div class="metric-label">{metric_name}</div>
            </div>
"""
                html_content += "        </div>\n"
            
            html_content += "    </div>\n"
        
        # 收敛性分析
        if convergence_analysis:
            html_content += f"""
    <!-- 收敛性分析 -->
    <div class="section">
        <h2>收敛性分析</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{'是' if convergence_analysis.get('is_converged') else '否'}</div>
                <div class="metric-label">是否收敛</div>
            </div>
"""
            if convergence_analysis.get('convergence_iteration'):
                html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{convergence_analysis.get('convergence_iteration')}</div>
                <div class="metric-label">收敛迭代</div>
            </div>
"""
            html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{convergence_analysis.get('improvement_rate', 0):.6f}</div>
                <div class="metric-label">改进速率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{convergence_analysis.get('final_improvement', 0):.4f}</div>
                <div class="metric-label">最终改进</div>
            </div>
        </div>
        <p>{convergence_analysis.get('analysis_summary', '')}</p>
    </div>
"""
        
        # 参数分析
        if parameter_analysis.get('parameter_sensitivity'):
            html_content += f"""
    <!-- 参数分析 -->
    <div class="section">
        <h2>参数敏感性分析</h2>
        <p>{parameter_analysis.get('sensitivity_summary', '')}</p>
        
        <h3>参数重要性排序</h3>
        <table class="parameter-table">
            <tr><th>排名</th><th>参数名</th><th>敏感性得分</th><th>相关系数</th><th>显著性</th></tr>
"""
            for param in parameter_analysis['parameter_sensitivity'][:10]:
                html_content += f"""
            <tr>
                <td>{param.get('importance_rank', 0)}</td>
                <td>{param.get('parameter_name', '')}</td>
                <td>{param.get('sensitivity_score', 0):.4f}</td>
                <td>{param.get('correlation_coefficient', 0):.4f}</td>
                <td>{param.get('significance', '')}</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""
        
        # 图表部分
        if chart_files:
            html_content += """
    <!-- 可视化分析 -->
    <div class="section">
        <h2>可视化分析</h2>
        
        <div class="chart-grid">
"""
            
            # 收敛曲线
            if 'convergence_curve' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>收敛曲线</h3>
                <img src="{chart_files['convergence_curve']}" alt="收敛曲线">
            </div>
"""
            
            # 参数重要性
            if 'parameter_importance' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>参数重要性</h3>
                <img src="{chart_files['parameter_importance']}" alt="参数重要性">
            </div>
"""
            
            # 目标函数值分布
            if 'objective_distribution' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>目标函数值分布</h3>
                <img src="{chart_files['objective_distribution']}" alt="目标函数值分布">
            </div>
"""
            
            # 参数相关性热力图
            if 'parameter_correlation_heatmap' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>参数相关性热力图</h3>
                <img src="{chart_files['parameter_correlation_heatmap']}" alt="参数相关性热力图">
            </div>
"""
            
            # 性能热力图
            if 'performance_heatmap' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>性能热力图</h3>
                <img src="{chart_files['performance_heatmap']}" alt="性能热力图">
            </div>
"""
            
            # 参数分布图
            if 'parameter_distributions' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>参数分布图</h3>
                <img src="{chart_files['parameter_distributions']}" alt="参数分布图">
            </div>
"""
            
            # 参数演化图
            if 'parameter_evolution' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>参数演化图</h3>
                <img src="{chart_files['parameter_evolution']}" alt="参数演化图">
            </div>
"""
            
            # 3D优化景观图
            if 'optimization_landscape_3d' in chart_files:
                html_content += f"""
            <div class="chart-container">
                <h3>3D优化景观图</h3>
                <img src="{chart_files['optimization_landscape_3d']}" alt="3D优化景观图">
            </div>
"""
            
            html_content += """
        </div>
    </div>
"""
        
        # 优化建议
        if recommendations:
            html_content += """
    <!-- 优化建议 -->
    <div class="section">
        <h2>优化建议</h2>
"""
            
            if recommendations.get('parameter_tuning'):
                html_content += """
        <h3>参数调优建议</h3>
"""
                for rec in recommendations['parameter_tuning']:
                    html_content += f'        <div class="recommendation">{rec}</div>\n'
            
            if recommendations.get('optimization_strategy'):
                html_content += """
        <h3>优化策略建议</h3>
"""
                for rec in recommendations['optimization_strategy']:
                    html_content += f'        <div class="recommendation">{rec}</div>\n'
            
            if recommendations.get('next_steps'):
                html_content += """
        <h3>下一步建议</h3>
"""
                for rec in recommendations['next_steps']:
                    html_content += f'        <div class="recommendation">{rec}</div>\n'
            
            html_content += "    </div>\n"
        
        # 页脚
        html_content += f"""
    <div class="footer">
        <p>报告由AutoDL贝叶斯优化系统自动生成 | 版本: {metadata.get('version', '1.0.0')}</p>
    </div>
</body>
</html>
"""
        
        return html_content


def create_enhanced_report_from_checkpoint(checkpoint_path: str, 
                                         output_dir: str,
                                         config: Optional[ReportConfig] = None) -> None:
    """
    从检查点文件创建增强版报告
    
    Args:
        checkpoint_path: 检查点文件路径
        output_dir: 输出目录
        config: 报告配置
    """
    try:
        from state_manager import StateManager
        
        # 加载状态数据
        state_manager = StateManager()
        state_data = state_manager.load_state(checkpoint_path)
        
        if 'optimization_history' not in state_data:
            print("错误：检查点文件中没有优化历史数据")
            return
        
        # 创建优化历史对象
        from autodl_core import OptimizationHistory
        history = OptimizationHistory.from_dict(state_data['optimization_history'])
        
        # 创建参数空间对象（如果存在）
        parameter_space = None
        if 'parameter_space' in state_data:
            from autodl_core import ParameterSpace
            parameter_space = ParameterSpace.from_dict(state_data['parameter_space'])
        
        # 创建增强报告生成器
        generator = EnhancedReportGenerator(history, parameter_space, config=config)
        
        # 生成报告
        generator.generate_enhanced_report(output_dir)
        
        print(f"增强版报告生成完成！输出目录: {output_dir}")
        
    except Exception as e:
        print(f"从检查点创建增强版报告失败: {e}")


if __name__ == "__main__":
    # 测试增强报告生成器
    print("测试增强报告生成器...")
    
    # 查找最新的检查点文件
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoint_files)[-1])
            print(f"使用检查点文件: {latest_checkpoint}")
            
            # 创建增强报告
            output_dir = "enhanced_reports"
            config = ReportConfig(
                title="增强版贝叶斯超参数优化报告",
                author="增强AutoDL系统",
                include_charts=True
            )
            
            create_enhanced_report_from_checkpoint(latest_checkpoint, output_dir, config)
        else:
            print("未找到检查点文件")
    else:
        print("检查点目录不存在")