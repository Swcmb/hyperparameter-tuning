"""
结果分析器使用示例

本示例展示如何使用ResultAnalyzer进行贝叶斯优化结果的深度分析
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

from result_analyzer import ResultAnalyzer, create_result_analyzer_from_checkpoint
from autodl_core import OptimizationHistory, OptimizationResult, create_default_parameter_space


def create_sample_optimization_history(n_iterations: int = 100) -> OptimizationHistory:
    """创建示例优化历史数据"""
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    history.start_time = datetime.now() - timedelta(hours=2)
    
    np.random.seed(42)
    
    for i in range(n_iterations):
        # 生成参数组合
        params = parameter_space.sample_random_parameters(seed=42+i)
        
        # 模拟目标函数值（带有一些参数依赖性）
        base_score = 0.75 + 0.15 * np.random.random()
        
        # 学习率影响
        lr = params.get('lr', 0.001)
        if 0.0001 <= lr <= 0.001:
            base_score += 0.05
        elif lr > 0.01:
            base_score -= 0.03
        
        # 维度影响
        dimensions = params.get('dimensions', 256)
        if 200 <= dimensions <= 400:
            base_score += 0.03
        
        # 融合策略影响
        fusion_strategy = params.get('fusion_strategy', 'self_attention')
        if fusion_strategy == 'co_attention':
            base_score += 0.04
        elif fusion_strategy == 'hybrid':
            base_score += 0.02
        
        # 批大小影响
        batch_size = params.get('batch', 32)
        if batch_size >= 32:
            base_score += 0.01
        
        # 添加一些噪声和趋势
        trend_bonus = min(0.05, i * 0.001)  # 随时间轻微改进
        noise = 0.02 * np.random.randn()
        
        objective_value = base_score + trend_bonus + noise
        objective_value = np.clip(objective_value, 0.5, 1.0)
        
        # 创建结果
        result = OptimizationResult(
            parameters=params,
            objective_value=objective_value,
            metrics={
                'AUROC': objective_value,
                'AUPRC': objective_value - 0.02 + 0.01 * np.random.randn(),
                'F1': objective_value - 0.05 + 0.02 * np.random.randn()
            },
            iteration=i + 1,
            timestamp=history.start_time + timedelta(minutes=i*2),
            evaluation_time=90 + 60 * np.random.random(),
            error_info=None if np.random.random() > 0.05 else "模拟评估错误"
        )
        
        history.add_result(result)
    
    history.end_time = datetime.now()
    history.total_time = (history.end_time - history.start_time).total_seconds()
    
    return history


def demonstrate_basic_analysis():
    """演示基本分析功能"""
    print("=== 基本分析功能演示 ===")
    
    # 创建示例数据
    history = create_sample_optimization_history(100)
    parameter_space = create_default_parameter_space()
    
    # 创建分析器
    analyzer = ResultAnalyzer(history, parameter_space)
    
    # 1. 统计摘要
    print("\n1. 统计摘要:")
    summary = analyzer.get_statistical_summary()
    print(f"   总评估次数: {summary.total_evaluations}")
    print(f"   最佳目标值: {summary.best_objective_value:.4f}")
    print(f"   平均目标值: {summary.mean_objective_value:.4f} ± {summary.std_objective_value:.4f}")
    print(f"   中位数: {summary.median_objective_value:.4f}")
    print(f"   成功率: {summary.success_rate:.2%}")
    print(f"   平均评估时间: {summary.average_evaluation_time:.1f}秒")
    
    # 2. 参数敏感性分析
    print("\n2. 参数敏感性分析:")
    sensitivity_results = analyzer.analyze_parameter_sensitivity()
    print(f"   分析了 {len(sensitivity_results)} 个参数")
    print("   前10个最重要的参数:")
    for i, result in enumerate(sensitivity_results[:10]):
        print(f"   {i+1:2d}. {result.parameter_name:15s}: "
              f"敏感性={result.sensitivity_score:6.3f}, "
              f"相关性={result.correlation_coefficient:6.3f}, "
              f"p值={result.p_value:.3f}")
    
    # 3. 收敛性分析
    print("\n3. 收敛性分析:")
    convergence = analyzer.analyze_convergence()
    print(f"   是否收敛: {convergence.is_converged}")
    if convergence.is_converged:
        print(f"   收敛迭代: {convergence.convergence_iteration}")
        print(f"   平台期长度: {convergence.plateau_length}")
    print(f"   总体改进率: {convergence.improvement_rate:.4f}")
    print(f"   最终改进幅度: {convergence.final_improvement:.4f}")
    
    # 4. 最佳参数分析
    print("\n4. 最佳参数分析 (前10个结果):")
    best_analysis = analyzer.get_best_parameters_analysis(top_k=10)
    print(f"   最佳目标值: {best_analysis['best_objective_value']:.4f}")
    print(f"   前10个平均值: {best_analysis['mean_top_k_objective']:.4f}")
    print("   最佳参数统计:")
    for param_name, stats in best_analysis['parameter_statistics'].items():
        if stats['type'] == 'categorical':
            print(f"   {param_name:15s}: {stats['most_frequent']} (出现{stats['frequency']}次)")
        else:
            print(f"   {param_name:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return analyzer


def demonstrate_advanced_analysis(analyzer: ResultAnalyzer):
    """演示高级分析功能"""
    print("\n=== 高级分析功能演示 ===")
    
    # 1. 参数相关性分析
    print("\n1. 参数相关性分析:")
    correlation_matrix = analyzer.analyze_parameter_correlations()
    if not correlation_matrix.empty:
        print("   强相关性参数对 (|相关系数| > 0.3):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.3:
                    param1 = correlation_matrix.columns[i]
                    param2 = correlation_matrix.columns[j]
                    print(f"   {param1} - {param2}: {corr:.3f}")
    
    # 2. 参数模式识别
    print("\n2. 参数模式识别:")
    patterns = analyzer.identify_parameter_patterns()
    if 'warning' not in patterns:
        for param_name, pattern in patterns.items():
            print(f"   {param_name:15s}: {pattern['trend_direction']} "
                  f"(相关性={pattern['trend_correlation']:.3f})")
    else:
        print(f"   {patterns['warning']}")
    
    # 3. 收敛曲线分析
    print("\n3. 收敛曲线分析:")
    convergence_curve = analyzer.get_convergence_curve()
    if len(convergence_curve) > 10:
        print(f"   初始10次迭代改进: {convergence_curve[9] - convergence_curve[0]:.4f}")
        print(f"   最后10次迭代改进: {convergence_curve[-1] - convergence_curve[-10]:.4f}")
        print(f"   总体改进: {convergence_curve[-1] - convergence_curve[0]:.4f}")


def demonstrate_report_generation(analyzer: ResultAnalyzer):
    """演示报告生成功能"""
    print("\n=== 报告生成演示 ===")
    
    # 生成完整报告
    report = analyzer.generate_analysis_report()
    
    print(f"生成的报告包含以下部分:")
    for key in report.keys():
        print(f"  - {key}")
    
    # 保存报告
    report_path = "optimization_analysis_report.json"
    analyzer.save_analysis_report(report_path)
    print(f"\n报告已保存到: {report_path}")
    
    # 显示报告摘要
    print(f"\n报告摘要:")
    print(f"  分析时间: {report['analysis_timestamp']}")
    print(f"  任务类型: {report['optimization_summary']['task_type']}")
    print(f"  采集函数: {report['optimization_summary']['acquisition_function']}")
    print(f"  总迭代数: {report['optimization_summary']['total_iterations']}")
    print(f"  最重要参数: {report['parameter_importance_ranking'][0][0]} "
          f"(重要性: {report['parameter_importance_ranking'][0][1]:.3f})")


def create_visualization_examples(analyzer: ResultAnalyzer):
    """创建可视化示例"""
    print("\n=== 可视化示例 ===")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 收敛曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    convergence_curve = analyzer.get_convergence_curve()
    plt.plot(convergence_curve, 'b-', linewidth=2)
    plt.title('优化收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳目标值')
    plt.grid(True, alpha=0.3)
    
    # 2. 参数重要性
    plt.subplot(2, 2, 2)
    importance_ranking = analyzer.get_parameter_importance_ranking()
    top_params = importance_ranking[:8]  # 前8个参数
    param_names = [item[0] for item in top_params]
    importance_scores = [item[1] for item in top_params]
    
    plt.barh(param_names, importance_scores)
    plt.title('参数重要性排序')
    plt.xlabel('重要性得分')
    
    # 3. 目标值分布
    plt.subplot(2, 2, 3)
    obj_values = [result.objective_value for result in analyzer.history.results]
    plt.hist(obj_values, bins=20, alpha=0.7, color='green')
    plt.title('目标值分布')
    plt.xlabel('目标值')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    
    # 4. 评估时间趋势
    plt.subplot(2, 2, 4)
    eval_times = [result.evaluation_time for result in analyzer.history.results]
    iterations = list(range(1, len(eval_times) + 1))
    plt.scatter(iterations, eval_times, alpha=0.6, color='red')
    plt.title('评估时间趋势')
    plt.xlabel('迭代次数')
    plt.ylabel('评估时间 (秒)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存到: optimization_analysis_visualization.png")
    plt.show()
    
    # 5. 参数相关性热力图
    correlation_matrix = analyzer.analyze_parameter_correlations()
    if not correlation_matrix.empty and len(correlation_matrix.columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('参数相关性矩阵')
        plt.tight_layout()
        plt.savefig('parameter_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("参数相关性热力图已保存到: parameter_correlation_heatmap.png")
        plt.show()


def main():
    """主函数"""
    print("结果分析器完整示例")
    print("=" * 50)
    
    # 基本分析演示
    analyzer = demonstrate_basic_analysis()
    
    # 高级分析演示
    demonstrate_advanced_analysis(analyzer)
    
    # 报告生成演示
    demonstrate_report_generation(analyzer)
    
    # 可视化示例
    create_visualization_examples(analyzer)
    
    print("\n" + "=" * 50)
    print("示例演示完成！")
    print("\n生成的文件:")
    print("  - optimization_analysis_report.json")
    print("  - optimization_analysis_visualization.png")
    print("  - parameter_correlation_heatmap.png")


if __name__ == "__main__":
    main()