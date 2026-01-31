"""
结果分析器（ResultAnalyzer）

本模块实现了贝叶斯超参数优化结果的深度分析功能，包括：
- 参数敏感性分析
- 优化历史统计分析  
- 参数重要性排序
- 收敛性分析
- 参数相关性分析
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression
import warnings
from dataclasses import dataclass
from datetime import datetime
import json

from autodl_core import OptimizationHistory, OptimizationResult, ParameterSpace


@dataclass
class ParameterSensitivityResult:
    """参数敏感性分析结果"""
    parameter_name: str
    sensitivity_score: float  # 敏感性得分（0-1）
    correlation_coefficient: float  # 与目标函数的相关系数
    p_value: float  # 统计显著性p值
    mutual_information: float  # 互信息得分
    importance_rank: int  # 重要性排名
    analysis_method: str  # 分析方法


@dataclass
class ConvergenceAnalysisResult:
    """收敛性分析结果"""
    is_converged: bool  # 是否收敛
    convergence_iteration: Optional[int]  # 收敛迭代次数
    convergence_threshold: float  # 收敛阈值
    improvement_rate: float  # 改进速率
    plateau_length: int  # 平台期长度
    final_improvement: float  # 最终改进幅度


@dataclass
class StatisticalSummary:
    """统计摘要"""
    total_evaluations: int
    best_objective_value: float
    worst_objective_value: float
    mean_objective_value: float
    std_objective_value: float
    median_objective_value: float
    q25_objective_value: float
    q75_objective_value: float
    total_time: float
    average_evaluation_time: float
    success_rate: float  # 成功评估率（无错误）


class ResultAnalyzer:
    """
    结果分析器
    
    提供贝叶斯优化结果的全面分析功能，包括参数敏感性、收敛性、
    统计摘要和参数重要性排序等。
    """
    
    def __init__(self, optimization_history: OptimizationHistory, 
                 parameter_space: Optional[ParameterSpace] = None):
        """
        初始化结果分析器
        
        Args:
            optimization_history: 优化历史记录
            parameter_space: 参数空间定义（可选，用于更详细的分析）
        """
        self.history = optimization_history
        self.parameter_space = parameter_space
        self.results_df = self._create_results_dataframe()
        
        # 缓存分析结果
        self._sensitivity_results: Optional[List[ParameterSensitivityResult]] = None
        self._convergence_result: Optional[ConvergenceAnalysisResult] = None
        self._statistical_summary: Optional[StatisticalSummary] = None
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """将优化历史转换为DataFrame格式以便分析"""
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
    
    def analyze_parameter_sensitivity(self, 
                                    methods: List[str] = ['correlation', 'mutual_info', 'random_forest'],
                                    min_samples: int = 10) -> List[ParameterSensitivityResult]:
        """
        分析参数敏感性
        
        Args:
            methods: 分析方法列表，可选 'correlation', 'mutual_info', 'random_forest'
            min_samples: 最小样本数要求
            
        Returns:
            参数敏感性分析结果列表，按重要性排序
        """
        if self._sensitivity_results is not None:
            return self._sensitivity_results
        
        if len(self.results_df) < min_samples:
            warnings.warn(f"样本数量({len(self.results_df)})少于最小要求({min_samples})，分析结果可能不可靠")
        
        # 获取参数列
        param_columns = [col for col in self.results_df.columns if col.startswith('param_')]
        if not param_columns:
            return []
        
        # 准备数据
        X = self.results_df[param_columns].copy()
        y = self.results_df['objective_value'].values
        
        # 处理分类变量
        categorical_params = []
        numerical_params = []
        
        for col in param_columns:
            param_name = col.replace('param_', '')
            if self.parameter_space and param_name in self.parameter_space.parameters:
                param_config = self.parameter_space.parameters[param_name]
                if param_config.param_type.value == 'categorical':
                    categorical_params.append(col)
                else:
                    numerical_params.append(col)
            else:
                # 自动判断数据类型
                if X[col].dtype == 'object' or X[col].nunique() <= 10:
                    categorical_params.append(col)
                else:
                    numerical_params.append(col)
        
        # 编码分类变量
        X_encoded = X.copy()
        label_encoders = {}
        for col in categorical_params:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        sensitivity_results = []
        
        for col in param_columns:
            param_name = col.replace('param_', '')
            param_values = X_encoded[col].values
            
            # 跳过常数参数
            if len(np.unique(param_values)) <= 1:
                continue
            
            result = ParameterSensitivityResult(
                parameter_name=param_name,
                sensitivity_score=0.0,
                correlation_coefficient=0.0,
                p_value=1.0,
                mutual_information=0.0,
                importance_rank=0,
                analysis_method='combined'
            )
            
            # 相关性分析
            if 'correlation' in methods:
                if col in numerical_params:
                    # 皮尔逊相关系数（连续变量）
                    corr, p_val = pearsonr(param_values, y)
                else:
                    # 斯皮尔曼相关系数（分类变量）
                    corr, p_val = spearmanr(param_values, y)
                
                result.correlation_coefficient = corr
                result.p_value = p_val
                result.sensitivity_score = max(result.sensitivity_score, abs(corr))
            
            # 互信息分析
            if 'mutual_info' in methods:
                try:
                    mi_score = mutual_info_regression(
                        param_values.reshape(-1, 1), y, 
                        discrete_features=[col in categorical_params],
                        random_state=42
                    )[0]
                    result.mutual_information = mi_score
                    # 归一化互信息得分
                    normalized_mi = mi_score / max(np.var(y), 1e-8)
                    result.sensitivity_score = max(result.sensitivity_score, normalized_mi)
                except Exception as e:
                    warnings.warn(f"互信息计算失败 for {param_name}: {e}")
            
            sensitivity_results.append(result)
        
        # 随机森林特征重要性分析
        if 'random_forest' in methods and len(X_encoded.columns) > 0:
            try:
                rf = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10,
                    min_samples_split=max(2, len(X_encoded) // 20)
                )
                rf.fit(X_encoded, y)
                
                feature_importance = rf.feature_importances_
                for i, col in enumerate(param_columns):
                    param_name = col.replace('param_', '')
                    # 找到对应的结果
                    for result in sensitivity_results:
                        if result.parameter_name == param_name:
                            rf_importance = feature_importance[i]
                            result.sensitivity_score = max(result.sensitivity_score, rf_importance)
                            break
            except Exception as e:
                warnings.warn(f"随机森林分析失败: {e}")
        
        # 排序并分配排名
        sensitivity_results.sort(key=lambda x: x.sensitivity_score, reverse=True)
        for i, result in enumerate(sensitivity_results):
            result.importance_rank = i + 1
        
        self._sensitivity_results = sensitivity_results
        return sensitivity_results
    
    def analyze_convergence(self, 
                          convergence_threshold: float = 0.001,
                          patience: int = 10) -> ConvergenceAnalysisResult:
        """
        分析优化收敛性
        
        Args:
            convergence_threshold: 收敛阈值（相对改进）
            patience: 耐心参数（连续多少次迭代无改进视为收敛）
            
        Returns:
            收敛性分析结果
        """
        if self._convergence_result is not None:
            return self._convergence_result
        
        if len(self.results_df) < 2:
            return ConvergenceAnalysisResult(
                is_converged=False,
                convergence_iteration=None,
                convergence_threshold=convergence_threshold,
                improvement_rate=0.0,
                plateau_length=0,
                final_improvement=0.0
            )
        
        # 获取收敛曲线（历史最佳值）
        convergence_curve = []
        current_best = float('-inf')
        
        for _, row in self.results_df.iterrows():
            obj_val = row['objective_value']
            if obj_val > current_best:
                current_best = obj_val
            convergence_curve.append(current_best)
        
        convergence_curve = np.array(convergence_curve)
        
        # 检测收敛点
        is_converged = False
        convergence_iteration = None
        plateau_length = 0
        
        for i in range(patience, len(convergence_curve)):
            # 检查过去patience次迭代的相对改进
            recent_best = convergence_curve[i]
            past_best = convergence_curve[i - patience]
            
            if past_best > 0:
                relative_improvement = (recent_best - past_best) / abs(past_best)
            else:
                relative_improvement = abs(recent_best - past_best)
            
            if relative_improvement < convergence_threshold:
                is_converged = True
                convergence_iteration = i - patience + 1
                plateau_length = len(convergence_curve) - convergence_iteration
                break
        
        # 计算改进速率
        if len(convergence_curve) > 1:
            initial_best = convergence_curve[0]
            final_best = convergence_curve[-1]
            
            if initial_best != 0:
                final_improvement = (final_best - initial_best) / abs(initial_best)
            else:
                final_improvement = abs(final_best - initial_best)
            
            # 计算平均改进速率（每次迭代）
            improvement_rate = final_improvement / len(convergence_curve)
        else:
            final_improvement = 0.0
            improvement_rate = 0.0
        
        result = ConvergenceAnalysisResult(
            is_converged=is_converged,
            convergence_iteration=convergence_iteration,
            convergence_threshold=convergence_threshold,
            improvement_rate=improvement_rate,
            plateau_length=plateau_length,
            final_improvement=final_improvement
        )
        
        self._convergence_result = result
        return result
    
    def get_statistical_summary(self) -> StatisticalSummary:
        """获取优化过程的统计摘要"""
        if self._statistical_summary is not None:
            return self._statistical_summary
        
        if len(self.results_df) == 0:
            return StatisticalSummary(
                total_evaluations=0,
                best_objective_value=0.0,
                worst_objective_value=0.0,
                mean_objective_value=0.0,
                std_objective_value=0.0,
                median_objective_value=0.0,
                q25_objective_value=0.0,
                q75_objective_value=0.0,
                total_time=0.0,
                average_evaluation_time=0.0,
                success_rate=0.0
            )
        
        obj_values = self.results_df['objective_value'].values
        eval_times = self.results_df['evaluation_time'].values
        error_count = self.results_df['has_error'].sum()
        
        summary = StatisticalSummary(
            total_evaluations=len(self.results_df),
            best_objective_value=float(np.max(obj_values)),
            worst_objective_value=float(np.min(obj_values)),
            mean_objective_value=float(np.mean(obj_values)),
            std_objective_value=float(np.std(obj_values)),
            median_objective_value=float(np.median(obj_values)),
            q25_objective_value=float(np.percentile(obj_values, 25)),
            q75_objective_value=float(np.percentile(obj_values, 75)),
            total_time=float(np.sum(eval_times)),
            average_evaluation_time=float(np.mean(eval_times)),
            success_rate=float((len(self.results_df) - error_count) / len(self.results_df))
        )
        
        self._statistical_summary = summary
        return summary
    
    def analyze_parameter_correlations(self) -> pd.DataFrame:
        """
        分析参数之间的相关性
        
        Returns:
            参数相关性矩阵
        """
        param_columns = [col for col in self.results_df.columns if col.startswith('param_')]
        if len(param_columns) < 2:
            return pd.DataFrame()
        
        # 准备数据
        X = self.results_df[param_columns].copy()
        
        # 处理分类变量
        for col in param_columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # 计算相关性矩阵
        correlation_matrix = X.corr()
        
        # 重命名列和索引（去掉param_前缀）
        new_names = [col.replace('param_', '') for col in correlation_matrix.columns]
        correlation_matrix.columns = new_names
        correlation_matrix.index = new_names
        
        return correlation_matrix
    
    def get_parameter_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        获取参数重要性排序
        
        Returns:
            参数重要性排序列表，格式为 [(参数名, 重要性得分), ...]
        """
        sensitivity_results = self.analyze_parameter_sensitivity()
        
        ranking = [(result.parameter_name, result.sensitivity_score) 
                  for result in sensitivity_results]
        
        return ranking
    
    def get_best_parameters_analysis(self, top_k: int = 10) -> Dict[str, Any]:
        """
        分析表现最好的参数组合
        
        Args:
            top_k: 分析前k个最佳结果
            
        Returns:
            最佳参数分析结果
        """
        if len(self.results_df) == 0:
            return {}
        
        # 获取前k个最佳结果
        top_results = self.results_df.nlargest(min(top_k, len(self.results_df)), 'objective_value')
        
        param_columns = [col for col in self.results_df.columns if col.startswith('param_')]
        
        analysis = {
            'top_k': len(top_results),
            'best_objective_value': float(top_results['objective_value'].iloc[0]),
            'mean_top_k_objective': float(top_results['objective_value'].mean()),
            'parameter_statistics': {}
        }
        
        # 分析每个参数在最佳结果中的分布
        for col in param_columns:
            param_name = col.replace('param_', '')
            param_values = top_results[col]
            
            if param_values.dtype == 'object' or param_values.nunique() <= 10:
                # 分类参数：统计频次
                value_counts = param_values.value_counts()
                analysis['parameter_statistics'][param_name] = {
                    'type': 'categorical',
                    'most_frequent': str(value_counts.index[0]),
                    'frequency': int(value_counts.iloc[0]),
                    'distribution': value_counts.to_dict()
                }
            else:
                # 数值参数：统计分布
                analysis['parameter_statistics'][param_name] = {
                    'type': 'numerical',
                    'mean': float(param_values.mean()),
                    'std': float(param_values.std()),
                    'min': float(param_values.min()),
                    'max': float(param_values.max()),
                    'median': float(param_values.median())
                }
        
        return analysis
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """
        生成完整的分析报告
        
        Returns:
            包含所有分析结果的综合报告
        """
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'optimization_summary': {
                'task_type': self.history.task_type,
                'acquisition_function': self.history.acquisition_function,
                'total_iterations': self.history.total_iterations,
                'total_time': self.history.total_time
            }
        }
        
        # 统计摘要
        statistical_summary = self.get_statistical_summary()
        report['statistical_summary'] = {
            'total_evaluations': statistical_summary.total_evaluations,
            'best_objective_value': statistical_summary.best_objective_value,
            'worst_objective_value': statistical_summary.worst_objective_value,
            'mean_objective_value': statistical_summary.mean_objective_value,
            'std_objective_value': statistical_summary.std_objective_value,
            'median_objective_value': statistical_summary.median_objective_value,
            'q25_objective_value': statistical_summary.q25_objective_value,
            'q75_objective_value': statistical_summary.q75_objective_value,
            'total_time': statistical_summary.total_time,
            'average_evaluation_time': statistical_summary.average_evaluation_time,
            'success_rate': statistical_summary.success_rate
        }
        
        # 参数敏感性分析
        sensitivity_results = self.analyze_parameter_sensitivity()
        report['parameter_sensitivity'] = []
        for result in sensitivity_results:
            report['parameter_sensitivity'].append({
                'parameter_name': result.parameter_name,
                'sensitivity_score': result.sensitivity_score,
                'correlation_coefficient': result.correlation_coefficient,
                'p_value': result.p_value,
                'mutual_information': result.mutual_information,
                'importance_rank': result.importance_rank,
                'analysis_method': result.analysis_method
            })
        
        # 收敛性分析
        convergence_result = self.analyze_convergence()
        report['convergence_analysis'] = {
            'is_converged': convergence_result.is_converged,
            'convergence_iteration': convergence_result.convergence_iteration,
            'convergence_threshold': convergence_result.convergence_threshold,
            'improvement_rate': convergence_result.improvement_rate,
            'plateau_length': convergence_result.plateau_length,
            'final_improvement': convergence_result.final_improvement
        }
        
        # 参数重要性排序
        importance_ranking = self.get_parameter_importance_ranking()
        report['parameter_importance_ranking'] = importance_ranking
        
        # 最佳参数分析
        best_params_analysis = self.get_best_parameters_analysis()
        report['best_parameters_analysis'] = best_params_analysis
        
        # 参数相关性分析
        correlation_matrix = self.analyze_parameter_correlations()
        if not correlation_matrix.empty:
            report['parameter_correlations'] = correlation_matrix.to_dict()
        
        return report
    
    def save_analysis_report(self, filepath: str) -> None:
        """
        保存分析报告到文件
        
        Args:
            filepath: 保存路径
        """
        report = self.generate_analysis_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    def get_convergence_curve(self) -> List[float]:
        """获取收敛曲线（历史最佳值序列）"""
        return self.history.get_convergence_curve()
    
    def get_parameter_history(self, parameter_name: str) -> List[Any]:
        """获取特定参数的历史值"""
        return self.history.get_parameter_history(parameter_name)
    
    def identify_parameter_patterns(self) -> Dict[str, Any]:
        """
        识别参数模式和趋势
        
        Returns:
            参数模式分析结果
        """
        if len(self.results_df) < 10:
            return {'warning': '样本数量不足，无法进行模式分析'}
        
        patterns = {}
        param_columns = [col for col in self.results_df.columns if col.startswith('param_')]
        
        # 按时间排序
        df_sorted = self.results_df.sort_values('iteration')
        
        for col in param_columns:
            param_name = col.replace('param_', '')
            param_values = df_sorted[col].values
            obj_values = df_sorted['objective_value'].values
            
            # 检测趋势
            if len(np.unique(param_values)) > 1:
                # 计算参数值与目标函数的滑动相关性
                window_size = min(10, len(param_values) // 3)
                if window_size >= 3:
                    correlations = []
                    for i in range(window_size, len(param_values)):
                        window_params = param_values[i-window_size:i]
                        window_objs = obj_values[i-window_size:i]
                        
                        if len(np.unique(window_params)) > 1:
                            corr, _ = spearmanr(window_params, window_objs)
                            if not np.isnan(corr):
                                correlations.append(corr)
                    
                    if correlations:
                        patterns[param_name] = {
                            'trend_correlation': np.mean(correlations),
                            'correlation_stability': np.std(correlations),
                            'trend_direction': 'increasing' if np.mean(correlations) > 0.1 else 
                                            'decreasing' if np.mean(correlations) < -0.1 else 'stable'
                        }
        
        return patterns


def create_result_analyzer_from_checkpoint(checkpoint_path: str) -> Optional[ResultAnalyzer]:
    """
    从检查点文件创建结果分析器
    
    Args:
        checkpoint_path: 检查点文件路径
        
    Returns:
        结果分析器实例，如果加载失败则返回None
    """
    try:
        from state_manager import StateManager
        
        state_manager = StateManager()
        state_data = state_manager.load_state(checkpoint_path)
        
        if 'optimization_history' in state_data:
            history = OptimizationHistory.from_dict(state_data['optimization_history'])
            parameter_space = None
            
            if 'parameter_space' in state_data:
                from autodl_core import ParameterSpace
                parameter_space = ParameterSpace.from_dict(state_data['parameter_space'])
            
            return ResultAnalyzer(history, parameter_space)
        
    except Exception as e:
        warnings.warn(f"从检查点加载结果分析器失败: {e}")
    
    return None


if __name__ == "__main__":
    # 测试代码
    print("测试结果分析器...")
    
    # 创建模拟数据
    from autodl_core import create_default_parameter_space
    
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    
    # 添加一些模拟结果
    np.random.seed(42)
    for i in range(50):
        params = parameter_space.sample_random_parameters(seed=42+i)
        
        # 模拟目标函数值（某些参数更重要）
        obj_value = 0.7 + 0.2 * np.random.random()
        if params.get('lr', 0.001) < 0.001:
            obj_value += 0.05
        if params.get('dimensions', 256) > 300:
            obj_value += 0.03
        if params.get('fusion_strategy') == 'co_attention':
            obj_value += 0.02
        
        result = OptimizationResult(
            parameters=params,
            objective_value=obj_value,
            metrics={'AUROC': obj_value, 'AUPRC': obj_value - 0.02, 'F1': obj_value - 0.05},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=120.0 + 30 * np.random.random()
        )
        
        history.add_result(result)
    
    # 创建分析器
    analyzer = ResultAnalyzer(history, parameter_space)
    
    # 测试各种分析功能
    print(f"创建了包含 {len(history.results)} 个结果的分析器")
    
    # 统计摘要
    summary = analyzer.get_statistical_summary()
    print(f"统计摘要: 最佳值={summary.best_objective_value:.4f}, 平均值={summary.mean_objective_value:.4f}")
    
    # 参数敏感性分析
    sensitivity_results = analyzer.analyze_parameter_sensitivity()
    print(f"参数敏感性分析: 找到 {len(sensitivity_results)} 个参数")
    for result in sensitivity_results[:5]:  # 显示前5个
        print(f"  {result.parameter_name}: 敏感性={result.sensitivity_score:.4f}, 排名={result.importance_rank}")
    
    # 收敛性分析
    convergence = analyzer.analyze_convergence()
    print(f"收敛性分析: 是否收敛={convergence.is_converged}, 改进率={convergence.improvement_rate:.4f}")
    
    # 参数重要性排序
    importance_ranking = analyzer.get_parameter_importance_ranking()
    print(f"参数重要性排序 (前5个):")
    for param_name, score in importance_ranking[:5]:
        print(f"  {param_name}: {score:.4f}")
    
    # 最佳参数分析
    best_analysis = analyzer.get_best_parameters_analysis(top_k=10)
    print(f"最佳参数分析: 前10个结果的平均目标值={best_analysis.get('mean_top_k_objective', 0):.4f}")
    
    # 生成完整报告
    report = analyzer.generate_analysis_report()
    print(f"生成完整分析报告，包含 {len(report)} 个主要部分")
    
    print("结果分析器测试完成!")