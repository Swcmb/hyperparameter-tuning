"""
可视化器（Visualizer）

本模块实现了贝叶斯超参数优化结果的可视化功能，包括：
- 收敛曲线可视化
- 参数分布和性能热力图
- 多目标优化的帕累托前沿可视化
- 参数重要性图表
- 优化过程动态可视化
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import os
from pathlib import Path

from autodl_core import OptimizationHistory, OptimizationResult, ParameterSpace
from result_analyzer import ResultAnalyzer, ParameterSensitivityResult


class Visualizer:
    """
    贝叶斯优化结果可视化器
    
    提供多种可视化功能，包括收敛曲线、参数分布、性能热力图、
    帕累托前沿和参数重要性分析等。
    """
    
    def __init__(self, optimization_history: OptimizationHistory, 
                 parameter_space: Optional[ParameterSpace] = None,
                 result_analyzer: Optional[ResultAnalyzer] = None):
        """
        初始化可视化器
        
        Args:
            optimization_history: 优化历史记录
            parameter_space: 参数空间定义（可选）
            result_analyzer: 结果分析器（可选，如果不提供会自动创建）
        """
        self.history = optimization_history
        self.parameter_space = parameter_space
        
        if result_analyzer is None:
            self.analyzer = ResultAnalyzer(optimization_history, parameter_space)
        else:
            self.analyzer = result_analyzer
        
        # 设置中文字体和绘图风格
        self._setup_plotting_style()
        
        # 创建结果DataFrame
        self.results_df = self._create_results_dataframe()
    
    def _setup_plotting_style(self):
        """设置绘图风格和中文字体"""
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置seaborn风格
        sns.set_style("whitegrid")
        sns.set_context("talk")
        
        # 设置高质量绘图参数
        plt.rcParams.update({
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "lines.antialiased": True,
            "patch.antialiased": True,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """创建结果DataFrame用于可视化"""
        if not self.history.results:
            return pd.DataFrame()
        
        data = []
        for result in self.history.results:
            row = {
                'iteration': result.iteration,
                'objective_value': result.objective_value,
                'evaluation_time': result.evaluation_time,
                'timestamp': result.timestamp,
                'has_error': result.error_info is not None
            }
            
            # 添加参数值
            for param_name, param_value in result.parameters.items():
                row[f'param_{param_name}'] = param_value
            
            # 添加其他指标
            if result.metrics:
                for metric_name, metric_value in result.metrics.items():
                    row[f'metric_{metric_name}'] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_convergence_curve(self, save_path: Optional[str] = None, 
                              show_confidence_interval: bool = True,
                              smooth: bool = False, window_size: int = 5) -> None:
        """
        绘制收敛曲线
        
        Args:
            save_path: 保存路径
            show_confidence_interval: 是否显示置信区间
            smooth: 是否平滑曲线
            window_size: 平滑窗口大小
        """
        if len(self.results_df) == 0:
            warnings.warn("没有优化结果数据，无法绘制收敛曲线")
            return
        
        # 获取收敛曲线数据
        convergence_curve = self.history.get_convergence_curve()
        iterations = list(range(1, len(convergence_curve) + 1))
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 上图：收敛曲线
        if smooth and len(convergence_curve) > window_size:
            # 应用滑动平均平滑
            smoothed_curve = pd.Series(convergence_curve).rolling(
                window=window_size, center=True, min_periods=1
            ).mean().values
            ax1.plot(iterations, smoothed_curve, 'b-', linewidth=2.5, 
                    label=f'平滑收敛曲线 (窗口={window_size})', alpha=0.8)
            ax1.plot(iterations, convergence_curve, 'b-', linewidth=1, 
                    alpha=0.3, label='原始收敛曲线')
        else:
            ax1.plot(iterations, convergence_curve, 'b-', linewidth=2.5, 
                    label='收敛曲线', marker='o', markersize=3)
        
        # 标记最佳点
        best_iteration = self.history.best_result.iteration if self.history.best_result else 1
        best_value = self.history.get_best_objective_value() or 0
        ax1.scatter([best_iteration], [best_value], color='red', s=100, 
                   zorder=5, label=f'最佳结果 (迭代 {best_iteration})')
        
        # 添加置信区间（基于历史方差）
        if show_confidence_interval and len(convergence_curve) > 10:
            # 计算滑动标准差
            obj_values = self.results_df['objective_value'].values
            rolling_std = pd.Series(obj_values).rolling(
                window=min(10, len(obj_values)), min_periods=1
            ).std().values
            
            upper_bound = np.array(convergence_curve) + rolling_std * 0.5
            lower_bound = np.array(convergence_curve) - rolling_std * 0.5
            
            ax1.fill_between(iterations, lower_bound, upper_bound, 
                           alpha=0.2, color='blue', label='置信区间')
        
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('目标函数值 (AUROC)')
        ax1.set_title(f'贝叶斯优化收敛曲线 - {self.history.task_type}任务')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：每次迭代的目标函数值（散点图）
        colors = ['red' if error else 'blue' for error in self.results_df['has_error']]
        ax2.scatter(self.results_df['iteration'], self.results_df['objective_value'], 
                   c=colors, alpha=0.6, s=50)
        
        # 添加趋势线
        z = np.polyfit(self.results_df['iteration'], self.results_df['objective_value'], 1)
        p = np.poly1d(z)
        ax2.plot(iterations, p(iterations), "r--", alpha=0.8, linewidth=1.5, 
                label=f'趋势线 (斜率={z[0]:.4f})')
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('目标函数值')
        ax2.set_title('每次迭代的目标函数值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = f"""统计信息:
总迭代次数: {len(convergence_curve)}
最佳值: {best_value:.4f}
最终值: {convergence_curve[-1]:.4f}
改进幅度: {convergence_curve[-1] - convergence_curve[0]:.4f}
成功率: {(1 - self.results_df['has_error'].mean()):.2%}"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_parameter_distributions(self, save_path: Optional[str] = None,
                                   max_params: int = 12) -> None:
        """
        绘制参数分布图
        
        Args:
            save_path: 保存路径
            max_params: 最大显示参数数量
        """
        param_columns = [col for col in self.results_df.columns if col.startswith('param_')]
        
        if len(param_columns) == 0:
            warnings.warn("没有参数数据，无法绘制参数分布")
            return
        
        # 限制参数数量
        if len(param_columns) > max_params:
            # 根据参数重要性排序
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            important_params = [f'param_{name}' for name, _ in importance_ranking[:max_params]]
            param_columns = [col for col in param_columns if col in important_params]
        
        # 计算子图布局
        n_params = len(param_columns)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 获取目标函数值用于颜色映射
        obj_values = self.results_df['objective_value'].values
        
        for i, param_col in enumerate(param_columns):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            param_name = param_col.replace('param_', '')
            param_values = self.results_df[param_col].values
            
            # 判断参数类型
            if self.parameter_space and param_name in self.parameter_space.parameters:
                param_config = self.parameter_space.parameters[param_name]
                is_categorical = param_config.param_type.value == 'categorical'
            else:
                # 自动判断
                is_categorical = (self.results_df[param_col].dtype == 'object' or 
                                self.results_df[param_col].nunique() <= 10)
            
            if is_categorical:
                # 分类参数：箱线图
                df_plot = pd.DataFrame({
                    'param': param_values,
                    'objective': obj_values
                })
                sns.boxplot(data=df_plot, x='param', y='objective', ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            else:
                # 连续参数：散点图
                scatter = ax.scatter(param_values, obj_values, c=obj_values, 
                                   cmap='viridis', alpha=0.6, s=50)
                
                # 添加趋势线
                try:
                    z = np.polyfit(param_values, obj_values, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(param_values.min(), param_values.max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=1.5)
                except:
                    pass
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('目标函数值')
            ax.set_title(f'{param_name} 分布')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(f'参数分布分析 - {self.history.task_type}任务', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_parameter_correlation_heatmap(self, save_path: Optional[str] = None,
                                         method: str = 'pearson') -> None:
        """
        绘制参数相关性热力图
        
        Args:
            save_path: 保存路径
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
        """
        correlation_matrix = self.analyzer.analyze_parameter_correlations()
        
        if correlation_matrix.empty:
            warnings.warn("没有足够的参数数据，无法绘制相关性热力图")
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 创建自定义颜色映射
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffff', 
                 '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # 绘制热力图
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # 只显示下三角
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                   fmt='.3f', annot_kws={'size': 8})
        
        ax.set_title(f'参数相关性热力图 ({method.capitalize()})', fontsize=16)
        ax.set_xlabel('参数')
        ax.set_ylabel('参数')
        
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_parameter_importance(self, save_path: Optional[str] = None,
                                top_k: int = 15) -> None:
        """
        绘制参数重要性图表
        
        Args:
            save_path: 保存路径
            top_k: 显示前k个重要参数
        """
        sensitivity_results = self.analyzer.analyze_parameter_sensitivity()
        
        if not sensitivity_results:
            warnings.warn("没有参数敏感性分析结果，无法绘制重要性图表")
            return
        
        # 取前k个重要参数
        top_results = sensitivity_results[:min(top_k, len(sensitivity_results))]
        
        # 准备数据
        param_names = [result.parameter_name for result in top_results]
        sensitivity_scores = [result.sensitivity_score for result in top_results]
        correlation_coeffs = [abs(result.correlation_coefficient) for result in top_results]
        mutual_info_scores = [result.mutual_information for result in top_results]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：参数重要性条形图
        y_pos = np.arange(len(param_names))
        bars = ax1.barh(y_pos, sensitivity_scores, alpha=0.8, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(param_names))))
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(param_names)
        ax1.set_xlabel('敏感性得分')
        ax1.set_title('参数重要性排序')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, sensitivity_scores)):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=9)
        
        # 右图：多指标对比雷达图
        if len(top_results) <= 8:  # 只有参数不太多时才绘制雷达图
            # 准备雷达图数据
            angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            # 归一化数据
            sensitivity_norm = np.array(sensitivity_scores) / max(sensitivity_scores) if max(sensitivity_scores) > 0 else np.zeros_like(sensitivity_scores)
            correlation_norm = np.array(correlation_coeffs) / max(correlation_coeffs) if max(correlation_coeffs) > 0 else np.zeros_like(correlation_coeffs)
            mutual_info_norm = np.array(mutual_info_scores) / max(mutual_info_scores) if max(mutual_info_scores) > 0 else np.zeros_like(mutual_info_scores)
            
            # 闭合数据
            sensitivity_norm = np.concatenate([sensitivity_norm, [sensitivity_norm[0]]])
            correlation_norm = np.concatenate([correlation_norm, [correlation_norm[0]]])
            mutual_info_norm = np.concatenate([mutual_info_norm, [mutual_info_norm[0]]])
            
            ax2 = plt.subplot(122, projection='polar')
            ax2.plot(angles, sensitivity_norm, 'o-', linewidth=2, label='敏感性得分', alpha=0.8)
            ax2.fill(angles, sensitivity_norm, alpha=0.25)
            ax2.plot(angles, correlation_norm, 's-', linewidth=2, label='相关系数', alpha=0.8)
            ax2.plot(angles, mutual_info_norm, '^-', linewidth=2, label='互信息', alpha=0.8)
            
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(param_names, fontsize=10)
            ax2.set_ylim(0, 1)
            ax2.set_title('多指标参数重要性对比', pad=20)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax2.grid(True)
        else:
            # 参数太多时，绘制多指标条形图
            x = np.arange(len(param_names))
            width = 0.25
            
            ax2.bar(x - width, sensitivity_scores, width, label='敏感性得分', alpha=0.8)
            ax2.bar(x, correlation_coeffs, width, label='相关系数', alpha=0.8)
            ax2.bar(x + width, mutual_info_scores, width, label='互信息', alpha=0.8)
            
            ax2.set_xlabel('参数')
            ax2.set_ylabel('得分')
            ax2.set_title('多指标参数重要性对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(param_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_performance_heatmap(self, save_path: Optional[str] = None,
                               param1: Optional[str] = None, 
                               param2: Optional[str] = None) -> None:
        """
        绘制参数性能热力图
        
        Args:
            save_path: 保存路径
            param1: 第一个参数名（如果不指定，自动选择最重要的参数）
            param2: 第二个参数名（如果不指定，自动选择第二重要的参数）
        """
        # 自动选择参数
        if param1 is None or param2 is None:
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if len(importance_ranking) < 2:
                warnings.warn("参数数量不足，无法绘制二维热力图")
                return
            
            if param1 is None:
                param1 = importance_ranking[0][0]
            if param2 is None:
                param2 = importance_ranking[1][0]
        
        # 检查参数是否存在
        param1_col = f'param_{param1}'
        param2_col = f'param_{param2}'
        
        if param1_col not in self.results_df.columns or param2_col not in self.results_df.columns:
            warnings.warn(f"参数 {param1} 或 {param2} 不存在")
            return
        
        # 获取数据
        x_values = self.results_df[param1_col].values
        y_values = self.results_df[param2_col].values
        z_values = self.results_df['objective_value'].values
        
        # 判断参数类型
        param1_is_categorical = (self.results_df[param1_col].dtype == 'object' or 
                               self.results_df[param1_col].nunique() <= 10)
        param2_is_categorical = (self.results_df[param2_col].dtype == 'object' or 
                               self.results_df[param2_col].nunique() <= 10)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if param1_is_categorical and param2_is_categorical:
            # 两个都是分类参数：创建分组热力图
            df_pivot = self.results_df.pivot_table(
                values='objective_value', 
                index=param2_col, 
                columns=param1_col, 
                aggfunc='mean'
            )
            
            sns.heatmap(df_pivot, annot=True, cmap='viridis', ax=ax, 
                       fmt='.3f', cbar_kws={'label': '平均目标函数值'})
            
        elif not param1_is_categorical and not param2_is_categorical:
            # 两个都是连续参数：插值热力图
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
            cbar.set_label('目标函数值')
            
        else:
            # 一个分类一个连续：分组散点图
            if param1_is_categorical:
                cat_param, cont_param = param1, param2
                cat_values, cont_values = x_values, y_values
            else:
                cat_param, cont_param = param2, param1
                cat_values, cont_values = y_values, x_values
            
            # 为每个类别绘制散点
            unique_cats = np.unique(cat_values)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_cats)))
            
            for i, cat in enumerate(unique_cats):
                mask = cat_values == cat
                ax.scatter(cont_values[mask], z_values[mask], 
                          c=[colors[i]], label=f'{cat_param}={cat}', 
                          alpha=0.7, s=50)
            
            ax.set_xlabel(cont_param)
            ax.set_ylabel('目标函数值')
            ax.legend()
        
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f'参数性能热力图: {param1} vs {param2}')
        
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pareto_frontier(self, save_path: Optional[str] = None,
                           objective1: str = 'AUROC', 
                           objective2: str = 'AUPRC') -> None:
        """
        绘制帕累托前沿（多目标优化）
        
        Args:
            save_path: 保存路径
            objective1: 第一个目标函数名
            objective2: 第二个目标函数名
        """
        # 检查是否有多目标数据
        obj1_col = f'metric_{objective1}'
        obj2_col = f'metric_{objective2}'
        
        if obj1_col not in self.results_df.columns or obj2_col not in self.results_df.columns:
            warnings.warn(f"缺少目标函数数据: {objective1} 或 {objective2}")
            return
        
        # 获取目标函数值
        obj1_values = self.results_df[obj1_col].values
        obj2_values = self.results_df[obj2_col].values
        
        # 计算帕累托前沿
        pareto_indices = self._find_pareto_frontier(obj1_values, obj2_values)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制所有点
        ax.scatter(obj1_values, obj2_values, alpha=0.6, s=50, 
                  color='lightblue', label='所有解')
        
        # 绘制帕累托前沿点
        pareto_obj1 = obj1_values[pareto_indices]
        pareto_obj2 = obj2_values[pareto_indices]
        ax.scatter(pareto_obj1, pareto_obj2, alpha=0.8, s=100, 
                  color='red', label='帕累托前沿', zorder=5)
        
        # 连接帕累托前沿点
        sorted_indices = np.argsort(pareto_obj1)
        sorted_pareto_obj1 = pareto_obj1[sorted_indices]
        sorted_pareto_obj2 = pareto_obj2[sorted_indices]
        ax.plot(sorted_pareto_obj1, sorted_pareto_obj2, 'r--', 
               alpha=0.7, linewidth=2, zorder=4)
        
        # 标记最佳点
        best_idx = self.history.best_result.iteration - 1 if self.history.best_result else 0
        if best_idx < len(obj1_values):
            ax.scatter([obj1_values[best_idx]], [obj2_values[best_idx]], 
                      color='gold', s=150, marker='*', 
                      label='最佳解', zorder=6, edgecolors='black')
        
        ax.set_xlabel(f'{objective1}')
        ax.set_ylabel(f'{objective2}')
        ax.set_title(f'帕累托前沿分析: {objective1} vs {objective2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"""帕累托前沿统计:
前沿解数量: {len(pareto_indices)}
总解数量: {len(obj1_values)}
前沿比例: {len(pareto_indices)/len(obj1_values):.2%}
{objective1}范围: [{pareto_obj1.min():.3f}, {pareto_obj1.max():.3f}]
{objective2}范围: [{pareto_obj2.min():.3f}, {pareto_obj2.max():.3f}]"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _find_pareto_frontier(self, obj1: np.ndarray, obj2: np.ndarray, 
                            maximize: bool = True) -> np.ndarray:
        """
        找到帕累托前沿点的索引
        
        Args:
            obj1: 第一个目标函数值数组
            obj2: 第二个目标函数值数组
            maximize: 是否最大化目标函数
            
        Returns:
            帕累托前沿点的索引数组
        """
        # 组合目标函数值
        objectives = np.column_stack([obj1, obj2])
        
        if not maximize:
            objectives = -objectives
        
        # 找到帕累托前沿
        pareto_indices = []
        n_points = len(objectives)
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # 检查点j是否支配点i
                    if (objectives[j] >= objectives[i]).all() and (objectives[j] > objectives[i]).any():
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return np.array(pareto_indices)
    
    def plot_optimization_landscape_3d(self, save_path: Optional[str] = None,
                                     param1: Optional[str] = None,
                                     param2: Optional[str] = None) -> None:
        """
        绘制3D优化景观图
        
        Args:
            save_path: 保存路径
            param1: 第一个参数名
            param2: 第二个参数名
        """
        # 自动选择参数
        if param1 is None or param2 is None:
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if len(importance_ranking) < 2:
                warnings.warn("参数数量不足，无法绘制3D景观图")
                return
            
            if param1 is None:
                param1 = importance_ranking[0][0]
            if param2 is None:
                param2 = importance_ranking[1][0]
        
        # 检查参数是否存在且为连续型
        param1_col = f'param_{param1}'
        param2_col = f'param_{param2}'
        
        if param1_col not in self.results_df.columns or param2_col not in self.results_df.columns:
            warnings.warn(f"参数 {param1} 或 {param2} 不存在")
            return
        
        # 检查是否为连续型参数
        if (self.results_df[param1_col].dtype == 'object' or 
            self.results_df[param2_col].dtype == 'object'):
            warnings.warn("3D景观图只支持连续型参数")
            return
        
        # 获取数据
        x_values = self.results_df[param1_col].values
        y_values = self.results_df[param2_col].values
        z_values = self.results_df['objective_value'].values
        
        # 创建3D图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格进行插值
        from scipy.interpolate import griddata
        
        xi = np.linspace(x_values.min(), x_values.max(), 30)
        yi = np.linspace(y_values.min(), y_values.max(), 30)
        xi, yi = np.meshgrid(xi, yi)
        
        # 插值
        zi = griddata((x_values, y_values), z_values, (xi, yi), method='cubic')
        
        # 绘制表面
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7, 
                              linewidth=0, antialiased=True)
        
        # 添加原始数据点
        ax.scatter(x_values, y_values, z_values, c=z_values, cmap='viridis', 
                  s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # 标记最佳点
        if self.history.best_result:
            best_params = self.history.best_result.parameters
            if param1 in best_params and param2 in best_params:
                best_x = best_params[param1]
                best_y = best_params[param2]
                best_z = self.history.best_result.objective_value
                ax.scatter([best_x], [best_y], [best_z], color='red', s=200, 
                          marker='*', label='最佳解')
        
        # 设置标签和标题
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel('目标函数值')
        ax.set_title(f'3D优化景观: {param1} vs {param2}')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='目标函数值')
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_parameter_evolution(self, save_path: Optional[str] = None,
                               params: Optional[List[str]] = None,
                               max_params: int = 6) -> None:
        """
        绘制参数演化图（参数值随迭代次数的变化）
        
        Args:
            save_path: 保存路径
            params: 要显示的参数列表
            max_params: 最大显示参数数量
        """
        if params is None:
            # 自动选择重要参数
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            params = [name for name, _ in importance_ranking[:max_params]]
        
        if not params:
            warnings.warn("没有可显示的参数")
            return
        
        # 计算子图布局
        n_params = len(params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        iterations = self.results_df['iteration'].values
        obj_values = self.results_df['objective_value'].values
        
        for i, param_name in enumerate(params):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            param_col = f'param_{param_name}'
            if param_col not in self.results_df.columns:
                continue
            
            param_values = self.results_df[param_col].values
            
            # 判断参数类型
            is_categorical = (self.results_df[param_col].dtype == 'object' or 
                            self.results_df[param_col].nunique() <= 10)
            
            if is_categorical:
                # 分类参数：使用不同颜色表示不同类别
                unique_values = self.results_df[param_col].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
                
                for j, value in enumerate(unique_values):
                    mask = param_values == value
                    ax.scatter(iterations[mask], obj_values[mask], 
                             c=[colors[j]], label=str(value), alpha=0.7, s=30)
                
                ax.legend(title=param_name, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_ylabel('目标函数值')
            else:
                # 连续参数：颜色映射表示参数值
                scatter = ax.scatter(iterations, obj_values, c=param_values, 
                                   cmap='viridis', alpha=0.7, s=30)
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(param_name)
                ax.set_ylabel('目标函数值')
            
            ax.set_xlabel('迭代次数')
            ax.set_title(f'{param_name} 演化')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(f'参数演化分析 - {self.history.task_type}任务', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        创建交互式仪表板（使用Plotly）
        
        Args:
            save_path: 保存路径（HTML文件）
        """
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('收敛曲线', '参数重要性', '参数分布', '性能散点图'),
                specs=[[{"secondary_y": False}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # 1. 收敛曲线
            convergence_curve = self.history.get_convergence_curve()
            iterations = list(range(1, len(convergence_curve) + 1))
            
            fig.add_trace(
                go.Scatter(x=iterations, y=convergence_curve, mode='lines+markers',
                          name='收敛曲线', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            # 2. 参数重要性
            importance_ranking = self.analyzer.get_parameter_importance_ranking()
            if importance_ranking:
                top_params = importance_ranking[:10]  # 前10个
                param_names = [name for name, _ in top_params]
                importance_scores = [score for _, score in top_params]
                
                fig.add_trace(
                    go.Bar(x=param_names, y=importance_scores, name='参数重要性',
                          marker_color='viridis'),
                    row=1, col=2
                )
            
            # 3. 参数分布（选择最重要的参数）
            if importance_ranking:
                top_param = importance_ranking[0][0]
                param_col = f'param_{top_param}'
                if param_col in self.results_df.columns:
                    fig.add_trace(
                        go.Scatter(x=self.results_df[param_col], 
                                  y=self.results_df['objective_value'],
                                  mode='markers', name=f'{top_param} 分布',
                                  marker=dict(color=self.results_df['objective_value'],
                                            colorscale='viridis', showscale=True)),
                        row=2, col=1
                    )
            
            # 4. 性能散点图（迭代 vs 目标函数值）
            fig.add_trace(
                go.Scatter(x=self.results_df['iteration'], 
                          y=self.results_df['objective_value'],
                          mode='markers', name='性能散点',
                          marker=dict(color=self.results_df['objective_value'],
                                    colorscale='viridis', size=8)),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title=f'贝叶斯优化交互式仪表板 - {self.history.task_type}任务',
                height=800,
                showlegend=True
            )
            
            # 更新坐标轴标签
            fig.update_xaxes(title_text="迭代次数", row=1, col=1)
            fig.update_yaxes(title_text="目标函数值", row=1, col=1)
            fig.update_xaxes(title_text="参数", row=1, col=2)
            fig.update_yaxes(title_text="重要性得分", row=1, col=2)
            fig.update_xaxes(title_text="参数值", row=2, col=1)
            fig.update_yaxes(title_text="目标函数值", row=2, col=1)
            fig.update_xaxes(title_text="迭代次数", row=2, col=2)
            fig.update_yaxes(title_text="目标函数值", row=2, col=2)
            
            if save_path:
                self._ensure_dir(save_path)
                fig.write_html(save_path)
                print(f"交互式仪表板已保存到: {save_path}")
            else:
                fig.show()
                
        except ImportError:
            warnings.warn("需要安装plotly库才能创建交互式仪表板: pip install plotly")
    
    def generate_comprehensive_report(self, output_dir: str) -> None:
        """
        生成综合可视化报告
        
        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print("正在生成综合可视化报告...")
        
        # 1. 收敛曲线
        print("  - 生成收敛曲线...")
        self.plot_convergence_curve(
            save_path=os.path.join(output_dir, "convergence_curve.png")
        )
        
        # 2. 参数分布
        print("  - 生成参数分布图...")
        self.plot_parameter_distributions(
            save_path=os.path.join(output_dir, "parameter_distributions.png")
        )
        
        # 3. 参数相关性热力图
        print("  - 生成参数相关性热力图...")
        self.plot_parameter_correlation_heatmap(
            save_path=os.path.join(output_dir, "parameter_correlations.png")
        )
        
        # 4. 参数重要性
        print("  - 生成参数重要性图...")
        self.plot_parameter_importance(
            save_path=os.path.join(output_dir, "parameter_importance.png")
        )
        
        # 5. 性能热力图
        print("  - 生成性能热力图...")
        self.plot_performance_heatmap(
            save_path=os.path.join(output_dir, "performance_heatmap.png")
        )
        
        # 6. 帕累托前沿（如果有多目标数据）
        if 'metric_AUPRC' in self.results_df.columns:
            print("  - 生成帕累托前沿图...")
            self.plot_pareto_frontier(
                save_path=os.path.join(output_dir, "pareto_frontier.png")
            )
        
        # 7. 参数演化
        print("  - 生成参数演化图...")
        self.plot_parameter_evolution(
            save_path=os.path.join(output_dir, "parameter_evolution.png")
        )
        
        # 8. 3D景观图（如果有足够的连续参数）
        importance_ranking = self.analyzer.get_parameter_importance_ranking()
        continuous_params = []
        for param_name, _ in importance_ranking:
            param_col = f'param_{param_name}'
            if (param_col in self.results_df.columns and 
                self.results_df[param_col].dtype != 'object' and
                self.results_df[param_col].nunique() > 10):
                continuous_params.append(param_name)
        
        if len(continuous_params) >= 2:
            print("  - 生成3D优化景观图...")
            self.plot_optimization_landscape_3d(
                save_path=os.path.join(output_dir, "optimization_landscape_3d.png"),
                param1=continuous_params[0],
                param2=continuous_params[1]
            )
        
        # 9. 交互式仪表板
        print("  - 生成交互式仪表板...")
        self.create_interactive_dashboard(
            save_path=os.path.join(output_dir, "interactive_dashboard.html")
        )
        
        print(f"综合可视化报告已生成完成，保存在: {output_dir}")
    
    def _ensure_dir(self, filepath: str) -> None:
        """确保文件路径的目录存在"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def create_visualizer_from_checkpoint(checkpoint_path: str) -> Optional[Visualizer]:
    """
    从检查点文件创建可视化器
    
    Args:
        checkpoint_path: 检查点文件路径
        
    Returns:
        可视化器实例，如果加载失败则返回None
    """
    try:
        from state_manager import StateManager
        from result_analyzer import create_result_analyzer_from_checkpoint
        
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
        
        # 创建结果分析器
        analyzer = create_result_analyzer_from_checkpoint(checkpoint_path)
        
        return Visualizer(history, parameter_space, analyzer)
        
    except Exception as e:
        warnings.warn(f"从检查点创建可视化器失败: {e}")
        return None


if __name__ == "__main__":
    # 测试代码
    print("测试可视化器...")
    
    # 创建模拟数据
    from autodl_core import create_default_parameter_space
    import numpy as np
    
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    
    # 添加模拟优化结果
    np.random.seed(42)
    for i in range(100):
        params = parameter_space.sample_random_parameters(seed=42+i)
        
        # 模拟目标函数值（某些参数更重要）
        obj_value = 0.7 + 0.2 * np.random.random()
        if params.get('lr', 0.001) < 0.001:
            obj_value += 0.05
        if params.get('dimensions', 256) > 300:
            obj_value += 0.03
        if params.get('fusion_strategy') == 'co_attention':
            obj_value += 0.02
        
        # 添加一些噪声和趋势
        obj_value += 0.001 * i  # 轻微的改进趋势
        obj_value += 0.02 * np.sin(i / 10)  # 一些周期性变化
        
        result = OptimizationResult(
            parameters=params,
            objective_value=obj_value,
            metrics={
                'AUROC': obj_value, 
                'AUPRC': obj_value - 0.02 + 0.01 * np.random.random(), 
                'F1': obj_value - 0.05 + 0.02 * np.random.random()
            },
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=120.0 + 30 * np.random.random()
        )
        
        history.add_result(result)
    
    # 创建可视化器
    visualizer = Visualizer(history, parameter_space)
    
    print(f"创建了包含 {len(history.results)} 个结果的可视化器")
    
    # 测试各种可视化功能
    output_dir = "test_visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("测试收敛曲线...")
    visualizer.plot_convergence_curve(
        save_path=os.path.join(output_dir, "test_convergence.png")
    )
    
    print("测试参数分布...")
    visualizer.plot_parameter_distributions(
        save_path=os.path.join(output_dir, "test_distributions.png")
    )
    
    print("测试参数重要性...")
    visualizer.plot_parameter_importance(
        save_path=os.path.join(output_dir, "test_importance.png")
    )
    
    print("测试性能热力图...")
    visualizer.plot_performance_heatmap(
        save_path=os.path.join(output_dir, "test_heatmap.png")
    )
    
    print("测试帕累托前沿...")
    visualizer.plot_pareto_frontier(
        save_path=os.path.join(output_dir, "test_pareto.png")
    )
    
    print("测试参数演化...")
    visualizer.plot_parameter_evolution(
        save_path=os.path.join(output_dir, "test_evolution.png")
    )
    
    print("生成综合报告...")
    visualizer.generate_comprehensive_report(output_dir)
    
    print(f"可视化器测试完成! 结果保存在: {output_dir}")