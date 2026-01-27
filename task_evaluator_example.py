"""
TaskEvaluator使用示例

展示如何在贝叶斯超参数优化中使用TaskEvaluator
"""

import numpy as np
import time
from datetime import datetime
from task_evaluator import create_task_evaluator
from autodl_core import create_default_parameter_space, OptimizationHistory


def example_bayesian_optimization():
    """
    贝叶斯优化示例
    
    展示如何使用TaskEvaluator进行参数优化
    """
    print("=== TaskEvaluator贝叶斯优化示例 ===")
    
    # 创建参数空间
    parameter_space = create_default_parameter_space("LDA")
    print(f"参数空间包含 {parameter_space.get_parameter_count()} 个参数")
    
    # 创建任务评估器
    evaluator = create_task_evaluator(
        task_type="LDA",
        use_real_training=False  # 使用模拟模式进行演示
    )
    
    # 创建优化历史记录
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    history.start_time = datetime.now()
    
    print("\n开始优化过程...")
    
    # 模拟贝叶斯优化的几次迭代
    n_iterations = 5
    
    for iteration in range(n_iterations):
        print(f"\n--- 迭代 {iteration + 1}/{n_iterations} ---")
        
        # 采样参数（在真实的贝叶斯优化中，这里会使用采集函数）
        parameters = parameter_space.sample_random_parameters(seed=42 + iteration)
        print(f"采样参数: lr={parameters['lr']:.6f}, batch={parameters['batch']}, "
              f"hidden1={parameters['hidden1']}, hidden2={parameters['hidden2']}")
        
        # 验证参数
        is_valid, errors = evaluator.validate_parameters(parameters)
        if not is_valid:
            print(f"参数验证失败: {errors}")
            continue
        
        # 评估参数
        start_time = time.time()
        metrics = evaluator.evaluate_parameters(parameters, n_folds=3)  # 使用3折以加快演示
        evaluation_time = time.time() - start_time
        
        # 创建优化结果
        result = evaluator.create_optimization_result(
            parameters=parameters,
            metrics=metrics,
            iteration=iteration + 1,
            evaluation_time=evaluation_time
        )
        
        # 添加到历史记录
        history.add_result(result)
        
        print(f"评估结果: AUROC={metrics['AUROC']:.4f}, "
              f"AUPRC={metrics['AUPRC']:.4f}, F1={metrics['F1']:.4f}")
        print(f"评估耗时: {evaluation_time:.2f}秒")
        
        # 显示当前最佳结果
        if history.best_result:
            print(f"当前最佳: AUROC={history.best_result.objective_value:.4f} "
                  f"(迭代 {history.best_result.iteration})")
    
    # 优化完成
    history.end_time = datetime.now()
    history.total_time = (history.end_time - history.start_time).total_seconds()
    
    print(f"\n=== 优化完成 ===")
    print(f"总迭代次数: {history.total_iterations}")
    print(f"总耗时: {history.total_time:.2f}秒")
    print(f"最佳AUROC: {history.get_best_objective_value():.4f}")
    
    # 显示最佳参数
    best_params = history.get_best_parameters()
    if best_params:
        print("\n最佳参数组合:")
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    # 显示收敛曲线
    convergence = history.get_convergence_curve()
    print(f"\n收敛曲线: {[f'{x:.4f}' for x in convergence]}")
    
    # 清理资源
    evaluator.cleanup()
    
    return history


def example_parameter_validation():
    """
    参数验证示例
    """
    print("\n=== 参数验证示例 ===")
    
    evaluator = create_task_evaluator("LDA", use_real_training=False)
    
    # 测试有效参数
    valid_params = {
        'dimensions': 256,
        'hidden1': 128,
        'hidden2': 64,
        'lr': 0.001,
        'batch': 32,
        'gat_heads': 4,
        'gt_heads': 4,
        'fusion_heads': 4
    }
    
    is_valid, errors = evaluator.validate_parameters(valid_params)
    print(f"有效参数验证: {is_valid}")
    if errors:
        print(f"错误: {errors}")
    
    # 测试无效参数
    invalid_params = {
        'dimensions': 256,
        'hidden1': 300,  # 违反递减约束
        'hidden2': 64,
        'lr': -0.001,    # 负学习率
        'batch': 0,      # 无效批大小
        'gat_heads': 3,  # 不能整除hidden1
        'gt_heads': 4,
        'fusion_heads': 4
    }
    
    is_valid, errors = evaluator.validate_parameters(invalid_params)
    print(f"\n无效参数验证: {is_valid}")
    if errors:
        print("错误列表:")
        for error in errors:
            print(f"  - {error}")
    
    evaluator.cleanup()


def example_multi_task_evaluation():
    """
    多任务评估示例
    """
    print("\n=== 多任务评估示例 ===")
    
    tasks = ["LDA", "MDA", "LMI"]
    
    # 相同的参数配置
    test_params = {
        'dimensions': 256,
        'hidden1': 128,
        'hidden2': 64,
        'lr': 0.001,
        'batch': 32,
        'epochs': 1,
        'gat_heads': 4,
        'gt_heads': 4,
        'fusion_heads': 4
    }
    
    results = {}
    
    for task in tasks:
        print(f"\n评估任务: {task}")
        
        evaluator = create_task_evaluator(task, use_real_training=False)
        
        # 评估参数
        metrics = evaluator.evaluate_parameters(test_params, n_folds=2)
        results[task] = metrics
        
        print(f"{task} - AUROC: {metrics['AUROC']:.4f}, "
              f"AUPRC: {metrics['AUPRC']:.4f}, F1: {metrics['F1']:.4f}")
        
        evaluator.cleanup()
    
    # 比较结果
    print(f"\n=== 任务性能比较 ===")
    for task in tasks:
        auroc = results[task]['AUROC']
        print(f"{task}: AUROC={auroc:.4f}")
    
    best_task = max(tasks, key=lambda t: results[t]['AUROC'])
    print(f"\n最佳任务: {best_task} (AUROC={results[best_task]['AUROC']:.4f})")


if __name__ == "__main__":
    # 运行所有示例
    try:
        # 贝叶斯优化示例
        history = example_bayesian_optimization()
        
        # 参数验证示例
        example_parameter_validation()
        
        # 多任务评估示例
        example_multi_task_evaluation()
        
        print("\n所有示例运行完成！")
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()