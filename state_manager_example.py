"""
状态管理器使用示例

演示如何在贝叶斯优化过程中使用StateManager进行状态保存和恢复
"""

import numpy as np
from datetime import datetime
import os
import shutil

from state_manager import create_default_state_manager
from autodl_core import (
    create_default_parameter_space, OptimizationHistory, OptimizationResult
)
from gaussian_process import create_default_gaussian_process


def simulate_optimization_with_checkpoints():
    """模拟带检查点的优化过程"""
    print("=== 贝叶斯优化状态管理示例 ===\n")
    
    # 1. 创建状态管理器
    print("1. 创建状态管理器...")
    state_manager = create_default_state_manager("example_checkpoints")
    print(f"   检查点目录: {state_manager.checkpoint_dir}")
    print(f"   最大检查点数: {state_manager.max_checkpoints}")
    print(f"   压缩存储: {state_manager.compression}\n")
    
    # 2. 初始化优化组件
    print("2. 初始化优化组件...")
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    gp = create_default_gaussian_process(random_state=42)
    
    print(f"   参数空间: {len(parameter_space.parameters)} 个参数")
    print(f"   历史记录: 初始化完成")
    print(f"   高斯过程: 初始化完成\n")
    
    # 3. 模拟优化过程
    print("3. 开始优化过程...")
    max_iterations = 10
    checkpoint_freq = 3  # 每3次迭代保存一次检查点
    
    for iteration in range(1, max_iterations + 1):
        print(f"   迭代 {iteration}/{max_iterations}")
        
        # 生成参数
        params = parameter_space.sample_random_parameters(seed=42 + iteration)
        
        # 模拟评估（简化版）
        # 在实际应用中，这里会调用TaskEvaluator
        objective_value = 0.6 + 0.3 * np.random.random() + 0.1 * np.sin(iteration)
        
        result = OptimizationResult(
            parameters=params,
            objective_value=objective_value,
            metrics={
                'AUROC': objective_value,
                'AUPRC': objective_value - 0.1,
                'F1': objective_value - 0.05
            },
            iteration=iteration,
            timestamp=datetime.now(),
            evaluation_time=100.0 + 20 * iteration
        )
        
        # 更新历史记录
        history.add_result(result)
        
        # 更新高斯过程
        if iteration == 1:
            # 第一次拟合
            X = np.array([list(params.values())[:10]])  # 取前10个数值参数
            y = np.array([objective_value])
            gp.fit(X, y)
        else:
            # 增量更新
            X_new = np.array([list(params.values())[:10]])
            y_new = np.array([objective_value])
            gp.update(X_new, y_new)
        
        print(f"     目标值: {objective_value:.4f}")
        print(f"     当前最佳: {history.get_best_objective_value():.4f}")
        print(f"     GP观测数: {gp.n_observations}")
        
        # 创建优化器状态
        optimizer_state = {
            'history': history,
            'parameter_space': parameter_space,
            'gaussian_process': gp,
            'acquisition_function': {'type': 'EI', 'xi': 0.01},
            'config': {
                'max_iterations': max_iterations,
                'task_type': 'LDA',
                'current_iteration': iteration,
                'checkpoint_freq': checkpoint_freq
            }
        }
        
        # 根据频率创建检查点
        checkpoint_path = state_manager.create_checkpoint(
            optimizer_state, iteration, checkpoint_freq
        )
        
        if checkpoint_path:
            print(f"     ✓ 检查点已保存: {os.path.basename(checkpoint_path)}")
        
        print()
    
    print("4. 优化完成，查看检查点状态...")
    
    # 4. 列出所有检查点
    checkpoints = state_manager.list_checkpoints()
    print(f"   总检查点数: {len(checkpoints)}")
    
    for i, cp in enumerate(checkpoints):
        status = "✓" if cp['is_valid'] else "✗"
        size_kb = cp['file_size'] / 1024
        print(f"   {i+1}. {status} {cp['filename']} (迭代{cp['iteration']}, {size_kb:.1f}KB)")
    
    print()
    
    # 5. 获取最新检查点的详细信息
    print("5. 最新检查点详细信息...")
    latest_checkpoint = state_manager.get_latest_checkpoint()
    if latest_checkpoint:
        info = state_manager.get_checkpoint_info(latest_checkpoint)
        print(f"   文件: {os.path.basename(info['file_path'])}")
        print(f"   迭代: {info['iteration']}")
        print(f"   时间: {info['timestamp']}")
        print(f"   组件: {', '.join(info['components'])}")
        
        if 'history_info' in info:
            hist_info = info['history_info']
            print(f"   历史记录: {hist_info['total_iterations']} 次迭代")
            print(f"   最佳目标值: {hist_info['best_objective_value']:.4f}")
            print(f"   任务类型: {hist_info['task_type']}")
        
        if 'gaussian_process_info' in info:
            gp_info = info['gaussian_process_info']
            print(f"   高斯过程: {gp_info['n_observations']} 个观测点")
    
    print()
    
    # 6. 模拟从检查点恢复
    print("6. 模拟从检查点恢复优化...")
    if latest_checkpoint:
        print(f"   从检查点恢复: {os.path.basename(latest_checkpoint)}")
        
        # 加载状态
        loaded_state = state_manager.load_state(latest_checkpoint)
        
        # 验证恢复的状态
        loaded_history = loaded_state['history']
        loaded_gp = loaded_state['gaussian_process']
        loaded_config = loaded_state['config']
        
        print(f"   恢复的历史记录: {loaded_history.total_iterations} 次迭代")
        print(f"   恢复的最佳值: {loaded_history.get_best_objective_value():.4f}")
        print(f"   恢复的GP观测数: {loaded_gp.n_observations}")
        print(f"   恢复的配置: 迭代 {loaded_config['current_iteration']}")
        
        # 验证高斯过程可以继续使用
        X_test = np.random.uniform(-1, 1, (2, 10))
        mean, std = loaded_gp.predict(X_test)
        print(f"   GP预测测试: 均值范围 [{mean.min():.3f}, {mean.max():.3f}]")
        
        print("   ✓ 状态恢复成功，可以继续优化")
    
    print()
    
    # 7. 演示检查点管理功能
    print("7. 检查点管理功能演示...")
    
    # 导出检查点
    if latest_checkpoint:
        export_path = "exported_checkpoint.pkl"
        state_manager.export_checkpoint(latest_checkpoint, export_path)
        print(f"   ✓ 检查点已导出到: {export_path}")
        
        # 验证导出的文件
        if state_manager.validate_checkpoint(export_path):
            print("   ✓ 导出的检查点验证通过")
        
        # 清理导出文件
        if os.path.exists(export_path):
            os.remove(export_path)
            print("   ✓ 导出文件已清理")
    
    # 清理损坏的检查点（如果有的话）
    cleaned_count = state_manager.cleanup_corrupted_checkpoints()
    if cleaned_count > 0:
        print(f"   ✓ 清理了 {cleaned_count} 个损坏的检查点")
    else:
        print("   ✓ 没有发现损坏的检查点")
    
    print()
    print("=== 示例完成 ===")
    
    return state_manager


def demonstrate_error_recovery():
    """演示错误恢复功能"""
    print("\n=== 错误恢复演示 ===\n")
    
    state_manager = create_default_state_manager("error_demo_checkpoints")
    
    # 创建一些测试数据
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    
    # 添加一些结果
    for i in range(3):
        params = parameter_space.sample_random_parameters(seed=100 + i)
        result = OptimizationResult(
            parameters=params,
            objective_value=0.7 + 0.1 * i,
            metrics={'AUROC': 0.7 + 0.1 * i},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=120.0
        )
        history.add_result(result)
    
    optimizer_state = {
        'history': history,
        'parameter_space': parameter_space,
        'config': {'task_type': 'MDA'}
    }
    
    # 保存正常检查点
    checkpoint_path = state_manager.save_state(optimizer_state, iteration=3)
    print(f"1. 保存正常检查点: {os.path.basename(checkpoint_path)}")
    
    # 创建损坏的检查点文件
    corrupted_path = os.path.join(state_manager.checkpoint_dir, "corrupted_checkpoint.pkl")
    with open(corrupted_path, 'w') as f:
        f.write("这是损坏的数据")
    
    print(f"2. 创建损坏的检查点: {os.path.basename(corrupted_path)}")
    
    # 验证检查点
    print("3. 验证检查点:")
    print(f"   正常检查点: {'✓' if state_manager.validate_checkpoint(checkpoint_path) else '✗'}")
    print(f"   损坏检查点: {'✓' if state_manager.validate_checkpoint(corrupted_path) else '✗'}")
    
    # 列出检查点（包括损坏的）
    print("4. 检查点列表:")
    checkpoints = state_manager.list_checkpoints()
    for cp in checkpoints:
        status = "✓" if cp['is_valid'] else "✗ (损坏)"
        print(f"   {status} {cp['filename']}")
    
    # 清理损坏的检查点
    print("5. 清理损坏的检查点:")
    cleaned_count = state_manager.cleanup_corrupted_checkpoints()
    print(f"   清理了 {cleaned_count} 个损坏的检查点")
    
    # 验证清理结果
    remaining_checkpoints = state_manager.list_checkpoints()
    valid_count = sum(1 for cp in remaining_checkpoints if cp['is_valid'])
    print(f"   剩余有效检查点: {valid_count} 个")
    
    print("\n=== 错误恢复演示完成 ===")
    
    return state_manager


if __name__ == "__main__":
    # 运行主要示例
    main_state_manager = simulate_optimization_with_checkpoints()
    
    # 运行错误恢复示例
    error_state_manager = demonstrate_error_recovery()
    
    # 清理示例文件
    print("\n清理示例文件...")
    for dir_name in ["example_checkpoints", "error_demo_checkpoints"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✓ 已清理: {dir_name}")
    
    print("\n所有示例运行完成！")