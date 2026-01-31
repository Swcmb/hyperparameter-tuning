"""
可视化器使用示例

本文件展示了如何使用Visualizer类进行贝叶斯优化结果的可视化分析。
"""

import os
import numpy as np
from datetime import datetime
from visualizer import Visualizer, create_visualizer_from_checkpoint
from autodl_core import OptimizationHistory, OptimizationResult, create_default_parameter_space
from result_analyzer import ResultAnalyzer


def create_sample_optimization_data():
    """创建示例优化数据"""
    print("创建示例优化数据...")
    
    # 创建参数空间
    parameter_space = create_default_parameter_space()
    
    # 创建优化历史
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    history.start_time = datetime.now()
    
    # 生成模拟优化结果
    np.random.seed(42)
    
    for i in range(80):
        # 随机采样参数
        params = parameter_space.sample_random_parameters(seed=42+i)
        
        # 模拟目标函数值（基于参数的复杂关系）
        base_score = 0.75
        
        # 学习率的影响
        lr = params.get('lr', 0.001)
        if 0.0001 <= lr <= 0.001:
            base_score += 0.05
        elif lr > 0.01:
            base_score -= 0.03
        
        # 网络结构的影响
        dimensions = params.get('dimensions', 256)
        hidden1 = params.get('hidden1', 128)
        hidden2 = params.get('hidden2', 64)
        
        if 200 <= dimensions <= 400 and 100 <= hidden1 <= 200 and 50 <= hidden2 <= 100:
            base_score += 0.04
        
        # 融合策略的影响
        fusion_strategy = params.get('fusion_strategy', 'self_attention')
        if fusion_strategy == 'co_attention':
            base_score += 0.03
        elif fusion_strategy == 'hybrid':
            base_score += 0.02
        
        # 批大小的影响
        batch_size = params.get('batch', 32)
        if batch_size == 32:
            base_score += 0.02
        elif batch_size == 64:
            base_score += 0.01
        
        # 添加随机噪声
        noise = np.random.normal(0, 0.02)
        base_score += noise
        
        # 添加轻微的改进趋势（模拟优化过程）
        improvement_trend = 0.001 * i * np.exp(-i/50)
        base_score += improvement_trend
        
        # 确保在合理范围内
        objective_value = np.clip(base_score, 0.6, 0.95)
        
        # 生成其他指标
        auprc = objective_value - 0.02 + np.random.normal(0, 0.01)
        f1 = objective_value - 0.05 + np.random.normal(0, 0.015)
        
        # 模拟评估时间
        eval_time = 120 + np.random.exponential(30)
        
        # 模拟偶尔的评估失败
        has_error = np.random.random() < 0.05
        error_info = "模拟评估错误" if has_error else None
        
        if has_error:
            objective_value = 0.5  # 失败时给予较低分数
        
        result = OptimizationResult(
            parameters=params,
            objective_value=objective_value,
            metrics={
                'AUROC': objective_value,
                'AUPRC': max(0.5, auprc),
                'F1': max(0.4, f1)
            },
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=eval_time,
            error_info=error_info
        )
        
        history.add_result(result)
    
    history.end_time = datetime.now()
    history.total_time = sum(r.evaluation_time for r in history.results)
    
    print(f"生成了 {len(history.results)} 个优化结果")
    print(f"最佳目标函数值: {history.get_best_objective_value():.4f}")
    
    return history, parameter_space


def demonstrate_basic_visualizations():
    """演示基本可视化功能"""
    print("\n=== 演示基本可视化功能 ===")
    
    # 创建示例数据
    history, parameter_space = create_sample_optimization_data()
    
    # 创建可视化器
    visualizer = Visualizer(history, parameter_space)
    
    # 创建输出目录
    output_dir = "visualization_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n可视化结果将保存到: {output_dir}")
    
    # 1. 收敛曲线
    print("\n1. 绘制收敛曲线...")
    visualizer.plot_convergence_curve(
        save_path=os.path.join(output_dir, "convergence_curve.png"),
        show_confidence_interval=True,
        smooth=True,
        window_size=5
    )
    
    # 2. 参数分布
    print("2. 绘制参数分布...")
    visualizer.plot_parameter_distributions(
        save_path=os.path.join(output_dir, "parameter_distributions.png"),
        max_params=12
    )
    
    # 3. 参数相关性热力图
    print("3. 绘制参数相关性热力图...")
    visualizer.plot_parameter_correlation_heatmap(
        save_path=os.path.join(output_dir, "parameter_correlations.png"),
        method='spearman'
    )
    
    # 4. 参数重要性
    print("4. 绘制参数重要性...")
    visualizer.plot_parameter_importance(
        save_path=os.path.join(output_dir, "parameter_importance.png"),
        top_k=15
    )
    
    # 5. 性能热力图
    print("5. 绘制性能热力图...")
    visualizer.plot_performance_heatmap(
        save_path=os.path.join(output_dir, "performance_heatmap.png")
    )
    
    # 6. 帕累托前沿
    print("6. 绘制帕累托前沿...")
    visualizer.plot_pareto_frontier(
        save_path=os.path.join(output_dir, "pareto_frontier.png"),
        objective1='AUROC',
        objective2='AUPRC'
    )
    
    # 7. 参数演化
    print("7. 绘制参数演化...")
    visualizer.plot_parameter_evolution(
        save_path=os.path.join(output_dir, "parameter_evolution.png"),
        max_params=6
    )
    
    # 8. 3D优化景观
    print("8. 绘制3D优化景观...")
    visualizer.plot_optimization_landscape_3d(
        save_path=os.path.join(output_dir, "optimization_landscape_3d.png")
    )
    
    # 9. 交互式仪表板
    print("9. 创建交互式仪表板...")
    try:
        visualizer.create_interactive_dashboard(
            save_path=os.path.join(output_dir, "interactive_dashboard.html")
        )
    except ImportError:
        print("   跳过交互式仪表板（需要安装plotly）")
    
    print(f"\n基本可视化演示完成！结果保存在: {output_dir}")


def demonstrate_comprehensive_report():
    """演示综合报告生成"""
    print("\n=== 演示综合报告生成 ===")
    
    # 创建示例数据
    history, parameter_space = create_sample_optimization_data()
    
    # 创建可视化器
    visualizer = Visualizer(history, parameter_space)
    
    # 生成综合报告
    report_dir = "comprehensive_report_output"
    visualizer.generate_comprehensive_report(report_dir)
    
    print(f"综合报告生成完成！保存在: {report_dir}")


def demonstrate_custom_analysis():
    """演示自定义分析"""
    print("\n=== 演示自定义分析 ===")
    
    # 创建示例数据
    history, parameter_space = create_sample_optimization_data()
    
    # 创建可视化器和分析器
    visualizer = Visualizer(history, parameter_space)
    analyzer = visualizer.analyzer
    
    # 输出目录
    output_dir = "custom_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n自定义分析结果将保存到: {output_dir}")
    
    # 1. 分析特定参数组合
    print("\n1. 分析学习率和批大小的组合效果...")
    visualizer.plot_performance_heatmap(
        save_path=os.path.join(output_dir, "lr_batch_heatmap.png"),
        param1='lr',
        param2='batch'
    )
    
    # 2. 分析网络结构参数
    print("2. 分析网络结构参数演化...")
    visualizer.plot_parameter_evolution(
        save_path=os.path.join(output_dir, "network_structure_evolution.png"),
        params=['dimensions', 'hidden1', 'hidden2', 'decoder1']
    )
    
    # 3. 分析损失权重参数
    print("3. 分析损失权重参数...")
    visualizer.plot_parameter_evolution(
        save_path=os.path.join(output_dir, "loss_weights_evolution.png"),
        params=['alpha', 'beta', 'gamma']
    )
    
    # 4. 获取统计摘要
    print("4. 生成统计摘要...")
    summary = analyzer.get_statistical_summary()
    
    print(f"   总评估次数: {summary.total_evaluations}")
    print(f"   最佳目标值: {summary.best_objective_value:.4f}")
    print(f"   平均目标值: {summary.mean_objective_value:.4f}")
    print(f"   标准差: {summary.std_objective_value:.4f}")
    print(f"   成功率: {summary.success_rate:.2%}")
    print(f"   总时间: {summary.total_time:.1f}秒")
    print(f"   平均评估时间: {summary.average_evaluation_time:.1f}秒")
    
    # 5. 分析收敛性
    print("5. 分析收敛性...")
    convergence = analyzer.analyze_convergence()
    
    print(f"   是否收敛: {convergence.is_converged}")
    if convergence.convergence_iteration:
        print(f"   收敛迭代: {convergence.convergence_iteration}")
    print(f"   改进率: {convergence.improvement_rate:.6f}")
    print(f"   最终改进: {convergence.final_improvement:.4f}")
    
    # 6. 分析最佳参数
    print("6. 分析最佳参数...")
    best_analysis = analyzer.get_best_parameters_analysis(top_k=10)
    
    print(f"   前10个结果的平均目标值: {best_analysis.get('mean_top_k_objective', 0):.4f}")
    print("   最佳参数统计:")
    
    for param_name, stats in best_analysis.get('parameter_statistics', {}).items():
        if stats['type'] == 'categorical':
            print(f"     {param_name}: 最常见值 = {stats['most_frequent']} (频次: {stats['frequency']})")
        else:
            print(f"     {param_name}: 均值 = {stats['mean']:.4f}, 标准差 = {stats['std']:.4f}")
    
    print(f"\n自定义分析完成！结果保存在: {output_dir}")


def demonstrate_checkpoint_loading():
    """演示从检查点加载可视化器"""
    print("\n=== 演示从检查点加载可视化器 ===")
    
    # 首先创建一个模拟的检查点文件
    print("1. 创建模拟检查点文件...")
    
    history, parameter_space = create_sample_optimization_data()
    
    # 模拟保存检查点
    checkpoint_data = {
        'optimization_history': history.to_dict(),
        'parameter_space': parameter_space.to_dict(),
        'iteration': len(history.results),
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = "demo_checkpoint.json"
    import json
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"   检查点文件已保存: {checkpoint_path}")
    
    # 2. 从检查点加载可视化器
    print("2. 从检查点加载可视化器...")
    
    try:
        visualizer = create_visualizer_from_checkpoint(checkpoint_path)
        
        if visualizer:
            print("   成功从检查点加载可视化器")
            
            # 生成一个简单的可视化
            output_dir = "checkpoint_visualization_output"
            os.makedirs(output_dir, exist_ok=True)
            
            visualizer.plot_convergence_curve(
                save_path=os.path.join(output_dir, "checkpoint_convergence.png")
            )
            
            print(f"   可视化结果保存在: {output_dir}")
        else:
            print("   从检查点加载失败")
    
    except Exception as e:
        print(f"   加载检查点时出错: {e}")
    
    # 清理临时文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def main():
    """主函数"""
    print("贝叶斯优化可视化器演示")
    print("=" * 50)
    
    try:
        # 1. 基本可视化功能演示
        demonstrate_basic_visualizations()
        
        # 2. 综合报告生成演示
        demonstrate_comprehensive_report()
        
        # 3. 自定义分析演示
        demonstrate_custom_analysis()
        
        # 4. 检查点加载演示
        demonstrate_checkpoint_loading()
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        print("\n生成的文件夹:")
        print("- visualization_demo_output/: 基本可视化演示")
        print("- comprehensive_report_output/: 综合报告")
        print("- custom_analysis_output/: 自定义分析")
        print("- checkpoint_visualization_output/: 检查点加载演示")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()