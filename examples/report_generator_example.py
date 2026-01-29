"""
报告生成器使用示例

演示如何使用ReportGenerator生成详细的优化报告
"""

import os
import numpy as np
from datetime import datetime, timedelta

from autodl_core import OptimizationHistory, OptimizationResult, create_default_parameter_space
from report_generator import ReportGenerator, ReportConfig


def create_sample_optimization_history():
    """创建示例优化历史数据"""
    print("创建示例优化历史数据...")
    
    # 创建参数空间
    parameter_space = create_default_parameter_space()
    
    # 创建优化历史
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "Expected Improvement"
    history.start_time = datetime.now() - timedelta(hours=2)
    
    # 模拟优化过程
    np.random.seed(42)
    base_time = history.start_time
    
    for i in range(100):
        # 生成参数
        params = parameter_space.sample_random_parameters(seed=42+i)
        
        # 模拟目标函数值（带有一些趋势和噪声）
        obj_value = 0.65 + 0.25 * np.random.random()
        
        # 某些参数组合更好
        if params.get('lr', 0.001) < 0.0005:
            obj_value += 0.08
        if params.get('dimensions', 256) > 350:
            obj_value += 0.05
        if params.get('fusion_strategy') == 'co_attention':
            obj_value += 0.04
        if params.get('dropout', 0.3) < 0.2:
            obj_value += 0.03
        
        # 添加改进趋势
        obj_value += 0.002 * i  # 轻微的整体改进
        obj_value += 0.03 * np.sin(i / 15)  # 一些周期性变化
        
        # 限制在合理范围内
        obj_value = min(max(obj_value, 0.5), 0.98)
        
        # 生成其他指标
        metrics = {
            'AUROC': obj_value,
            'AUPRC': obj_value - 0.02 + 0.01 * np.random.random(),
            'F1': obj_value - 0.05 + 0.02 * np.random.random(),
            'Precision': obj_value - 0.03 + 0.015 * np.random.random(),
            'Recall': obj_value - 0.04 + 0.02 * np.random.random()
        }
        
        # 确保指标在合理范围内
        for key in metrics:
            metrics[key] = min(max(metrics[key], 0.3), 1.0)
        
        # 模拟评估时间（某些参数组合更慢）
        eval_time = 90 + 60 * np.random.random()
        if params.get('batch', 32) < 20:
            eval_time *= 1.5  # 小批量更慢
        if params.get('dimensions', 256) > 400:
            eval_time *= 1.3  # 大维度更慢
        
        # 模拟一些失败的评估
        error_info = None
        if np.random.random() < 0.08:  # 8%的失败率
            error_types = [
                "CUDA out of memory",
                "Training timeout after 300s",
                "NaN loss detected",
                "Invalid parameter combination"
            ]
            error_info = np.random.choice(error_types)
            obj_value = 0.5  # 失败时的惩罚值
            eval_time *= 0.3  # 失败时通常更快结束
        
        # 创建结果
        result = OptimizationResult(
            parameters=params,
            objective_value=obj_value,
            metrics=metrics,
            iteration=i + 1,
            timestamp=base_time + timedelta(seconds=sum(eval_time for eval_time in [90] * i)),
            evaluation_time=eval_time,
            error_info=error_info
        )
        
        history.add_result(result)
    
    history.end_time = datetime.now()
    history.total_time = sum(r.evaluation_time for r in history.results)
    
    print(f"创建了包含 {len(history.results)} 个结果的优化历史")
    print(f"最佳目标值: {history.get_best_objective_value():.4f}")
    print(f"成功率: {len([r for r in history.results if r.error_info is None]) / len(history.results):.2%}")
    
    return history, parameter_space


def demonstrate_basic_report_generation():
    """演示基本报告生成功能"""
    print("\n=== 演示基本报告生成 ===")
    
    # 创建示例数据
    history, parameter_space = create_sample_optimization_history()
    
    # 创建报告生成器
    config = ReportConfig(
        title="LDA任务贝叶斯超参数优化报告",
        author="AutoDL优化系统",
        include_charts=True,
        include_parameter_details=True,
        include_convergence_analysis=True,
        include_sensitivity_analysis=True
    )
    
    generator = ReportGenerator(history, parameter_space, config=config)
    
    # 创建输出目录
    output_dir = "example_reports/basic"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成JSON报告
    print("生成JSON报告...")
    generator.save_json_report(os.path.join(output_dir, "optimization_report.json"))
    
    # 生成HTML报告
    print("生成HTML报告...")
    generator.save_html_report(os.path.join(output_dir, "optimization_report.html"))
    
    # 尝试生成PDF报告
    try:
        print("生成PDF报告...")
        generator.save_pdf_report(os.path.join(output_dir, "optimization_report.pdf"))
    except ImportError:
        print("跳过PDF生成（需要安装weasyprint）")
    except Exception as e:
        print(f"PDF生成失败: {e}")
    
    print(f"基本报告已生成到: {output_dir}")


def demonstrate_comprehensive_report():
    """演示综合报告生成"""
    print("\n=== 演示综合报告生成 ===")
    
    # 创建示例数据
    history, parameter_space = create_sample_optimization_history()
    
    # 创建详细配置
    config = ReportConfig(
        title="综合优化分析报告 - LDA任务",
        author="研究团队",
        include_charts=True,
        include_parameter_details=True,
        include_convergence_analysis=True,
        include_sensitivity_analysis=True,
        include_best_parameters_analysis=True,
        chart_dpi=300,
        max_parameters_in_charts=12
    )
    
    generator = ReportGenerator(history, parameter_space, config=config)
    
    # 创建输出目录
    output_dir = "example_reports/comprehensive"
    
    # 生成所有格式的报告
    print("生成所有格式的综合报告...")
    generator.generate_all_formats(output_dir, "comprehensive_optimization_report")
    
    print(f"综合报告已生成到: {output_dir}")


def demonstrate_custom_analysis():
    """演示自定义分析报告"""
    print("\n=== 演示自定义分析报告 ===")
    
    # 创建示例数据
    history, parameter_space = create_sample_optimization_history()
    
    # 创建自定义配置（只包含特定分析）
    config = ReportConfig(
        title="参数敏感性专项分析报告",
        author="参数调优专家",
        include_charts=True,
        include_parameter_details=False,  # 不包含详细参数信息
        include_convergence_analysis=False,  # 不包含收敛分析
        include_sensitivity_analysis=True,  # 重点关注敏感性分析
        include_best_parameters_analysis=True,
        max_parameters_in_charts=8
    )
    
    generator = ReportGenerator(history, parameter_space, config=config)
    
    # 生成报告数据并查看内容
    report_data = generator.generate_report_data()
    
    print("报告包含的主要部分:")
    for section_name in report_data.keys():
        print(f"  - {section_name}")
    
    # 显示参数敏感性分析结果
    if 'parameter_analysis' in report_data and 'parameter_sensitivity' in report_data['parameter_analysis']:
        print("\n参数敏感性排序（前5个）:")
        for i, param in enumerate(report_data['parameter_analysis']['parameter_sensitivity'][:5]):
            print(f"  {i+1}. {param['parameter_name']}: {param['sensitivity_score']:.4f} ({param['significance']})")
    
    # 保存自定义报告
    output_dir = "example_reports/custom"
    os.makedirs(output_dir, exist_ok=True)
    
    generator.save_html_report(os.path.join(output_dir, "sensitivity_analysis_report.html"))
    generator.save_json_report(os.path.join(output_dir, "sensitivity_analysis_report.json"))
    
    print(f"自定义分析报告已生成到: {output_dir}")


def demonstrate_report_data_access():
    """演示如何访问和使用报告数据"""
    print("\n=== 演示报告数据访问 ===")
    
    # 创建示例数据
    history, parameter_space = create_sample_optimization_history()
    generator = ReportGenerator(history, parameter_space)
    
    # 获取报告数据
    report_data = generator.generate_report_data()
    
    # 访问关键信息
    print("关键优化信息:")
    print(f"  任务类型: {report_data['metadata']['system_info']['task_type']}")
    print(f"  总迭代次数: {report_data['optimization_summary']['total_evaluations']}")
    print(f"  最佳目标值: {report_data['optimization_summary']['best_objective_value']:.4f}")
    print(f"  成功率: {report_data['optimization_summary']['success_rate']:.2%}")
    
    # 访问最佳参数
    if report_data['best_parameters']['best_single_result']:
        best_params = report_data['best_parameters']['best_single_result']['parameters']
        print(f"\n最佳参数组合:")
        for param_name, param_value in list(best_params.items())[:5]:  # 只显示前5个
            print(f"  {param_name}: {param_value}")
    
    # 访问参数重要性
    if report_data['parameter_analysis']['importance_ranking']:
        print(f"\n参数重要性排序（前3个）:")
        for i, (param_name, score) in enumerate(report_data['parameter_analysis']['importance_ranking'][:3]):
            print(f"  {i+1}. {param_name}: {score:.4f}")
    
    # 访问收敛信息
    if report_data['convergence_analysis']:
        conv_info = report_data['convergence_analysis']
        print(f"\n收敛性分析:")
        print(f"  是否收敛: {'是' if conv_info['is_converged'] else '否'}")
        if conv_info['convergence_iteration']:
            print(f"  收敛迭代: {conv_info['convergence_iteration']}")
        print(f"  改进速率: {conv_info['improvement_rate']:.6f}")
    
    # 访问建议
    if report_data['recommendations']:
        print(f"\n优化建议:")
        for category, suggestions in report_data['recommendations'].items():
            if suggestions:
                print(f"  {category}:")
                for suggestion in suggestions[:2]:  # 只显示前2个建议
                    print(f"    - {suggestion}")


def main():
    """主函数"""
    print("报告生成器使用示例")
    print("=" * 50)
    
    try:
        # 演示各种功能
        demonstrate_basic_report_generation()
        demonstrate_comprehensive_report()
        demonstrate_custom_analysis()
        demonstrate_report_data_access()
        
        print("\n" + "=" * 50)
        print("所有示例演示完成!")
        print("生成的报告文件位于 example_reports/ 目录中")
        print("建议查看HTML报告以获得最佳阅读体验")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()