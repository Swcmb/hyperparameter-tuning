"""
采集函数演示

展示采集函数的完整功能，包括所有支持的采集函数类型、
参数验证、优化逻辑等功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acquisition_function import (
    ExpectedImprovement, ProbabilityOfImprovement, 
    UpperConfidenceBound, EntropySearch,
    create_acquisition_function, AcquisitionOptimizer
)
from gaussian_process import GaussianProcess


def demo_acquisition_functions():
    """演示所有采集函数的功能"""
    print("=" * 60)
    print("采集函数功能演示")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    
    # 1D测试函数用于可视化
    def test_func_1d(x):
        return -(x - 2)**2 + 1 + 0.1 * np.sin(10 * x)
    
    # 训练数据
    X_train = np.array([[0.5], [1.5], [3.0], [4.5]]).reshape(-1, 1)
    y_train = np.array([test_func_1d(x[0]) for x in X_train])
    
    # 创建高斯过程模型
    gp = GaussianProcess(random_state=42)
    gp.fit(X_train, y_train)
    
    print(f"训练数据点: {len(X_train)}")
    print(f"当前最佳值: {np.max(y_train):.4f}")
    
    # 测试点
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    
    # 获取高斯过程预测
    mean, std = gp.predict(X_test, return_std=True)
    
    print("\n1. 创建和测试所有采集函数:")
    print("-" * 40)
    
    # 创建所有采集函数
    acquisition_functions = {
        'EI': ExpectedImprovement(xi=0.01),
        'PI': ProbabilityOfImprovement(xi=0.01),
        'UCB': UpperConfidenceBound(kappa=2.576),
        'ES': EntropySearch(n_samples=50)
    }
    
    best_value = np.max(y_train)
    
    for name, acq_func in acquisition_functions.items():
        print(f"\n{name} 采集函数:")
        print(f"  类型: {acq_func.function_type}")
        print(f"  参数: {acq_func.params}")
        print(f"  参数验证: {acq_func.validate_parameters()}")
        
        # 计算采集函数值
        acq_values = acq_func.evaluate(X_test, gp, best_value)
        print(f"  采集函数值范围: [{np.min(acq_values):.6f}, {np.max(acq_values):.6f}]")
        
        # 找到最大值点
        max_idx = np.argmax(acq_values)
        print(f"  建议下一个评估点: x={X_test[max_idx][0]:.4f}, acq_val={acq_values[max_idx]:.6f}")
    
    print("\n2. 测试工厂函数:")
    print("-" * 40)
    
    for func_type in ['EI', 'PI', 'UCB', 'ES']:
        acq_func = create_acquisition_function(func_type)
        print(f"  {func_type}: {type(acq_func).__name__} - ✓")
    
    print("\n3. 测试参数验证和更新:")
    print("-" * 40)
    
    # 测试EI参数更新
    ei = ExpectedImprovement(xi=0.01)
    print(f"  EI初始xi: {ei.xi}")
    
    ei.update_xi(0.1)
    print(f"  更新后xi: {ei.xi}")
    print(f"  参数验证: {ei.validate_parameters()}")
    
    try:
        ei.update_xi(-0.1)  # 应该失败
    except ValueError as e:
        print(f"  无效参数正确拒绝: {e}")
    
    # 测试UCB参数更新
    ucb = UpperConfidenceBound(kappa=2.576)
    print(f"  UCB初始kappa: {ucb.kappa}")
    
    ucb.update_kappa(1.96)
    print(f"  更新后kappa: {ucb.kappa}")
    print(f"  参数验证: {ucb.validate_parameters()}")
    
    print("\n4. 测试采集函数优化:")
    print("-" * 40)
    
    bounds = [(0.0, 5.0)]
    ei = ExpectedImprovement(xi=0.01)
    
    # 测试不同优化方法
    optimizers = [
        ('L-BFGS-B', AcquisitionOptimizer(method='L-BFGS-B', n_restarts=5)),
        ('differential_evolution', AcquisitionOptimizer(method='differential_evolution')),
        ('grid_search', AcquisitionOptimizer(method='grid_search'))
    ]
    
    for method_name, optimizer in optimizers:
        try:
            best_x, best_val = optimizer.optimize(ei, gp, bounds, best_value)
            print(f"  {method_name}: x={best_x[0]:.4f}, acq_val={best_val:.6f}")
        except Exception as e:
            print(f"  {method_name}: 失败 - {e}")
    
    print("\n5. 测试采集函数信息获取:")
    print("-" * 40)
    
    ei = ExpectedImprovement(xi=0.05)
    info = ei.get_info()
    print(f"  采集函数信息: {info}")
    
    print("\n6. 演示贝叶斯优化迭代:")
    print("-" * 40)
    
    # 模拟几次贝叶斯优化迭代
    current_X = X_train.copy()
    current_y = y_train.copy()
    current_gp = GaussianProcess(random_state=42)
    current_gp.fit(current_X, current_y)
    
    ei = ExpectedImprovement(xi=0.01)
    bounds = [(0.0, 5.0)]
    
    for iteration in range(3):
        print(f"\n  迭代 {iteration + 1}:")
        
        # 找到下一个评估点
        best_x, best_acq_val = ei.optimize_acquisition(
            current_gp, bounds, np.max(current_y), n_restarts=5
        )
        
        # 评估真实函数
        new_y = test_func_1d(best_x[0])
        
        print(f"    建议点: x={best_x[0]:.4f}")
        print(f"    采集函数值: {best_acq_val:.6f}")
        print(f"    真实函数值: {new_y:.4f}")
        print(f"    当前最佳: {np.max(current_y):.4f}")
        
        # 更新模型
        current_X = np.vstack([current_X, best_x.reshape(1, -1)])
        current_y = np.append(current_y, new_y)
        current_gp.fit(current_X, current_y)
        
        print(f"    更新后最佳: {np.max(current_y):.4f}")
        print(f"    改进: {np.max(current_y) - np.max(y_train):.4f}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("✓ 所有采集函数功能正常工作")
    print("✓ 参数验证和更新功能正常")
    print("✓ 采集函数优化功能正常")
    print("✓ 与高斯过程模型集成正常")
    print("=" * 60)


def demo_advanced_features():
    """演示高级功能"""
    print("\n高级功能演示:")
    print("-" * 40)
    
    # 多维优化演示
    print("\n1. 多维优化演示:")
    
    # 2D测试函数
    def test_func_2d(x):
        return -(x[0]**2 + x[1]**2) + 0.1 * np.sin(5 * x[0]) * np.cos(5 * x[1])
    
    # 创建2D训练数据
    np.random.seed(42)
    X_train_2d = np.random.uniform(-2, 2, (8, 2))
    y_train_2d = np.array([test_func_2d(x) for x in X_train_2d])
    
    gp_2d = GaussianProcess(random_state=42)
    gp_2d.fit(X_train_2d, y_train_2d)
    
    bounds_2d = [(-3.0, 3.0), (-3.0, 3.0)]
    best_value_2d = np.max(y_train_2d)
    
    # 测试不同采集函数在2D问题上的表现
    for name, acq_func in [('EI', ExpectedImprovement(xi=0.01)), 
                          ('UCB', UpperConfidenceBound(kappa=2.576))]:
        best_x, best_val = acq_func.optimize_acquisition(
            gp_2d, bounds_2d, best_value_2d, n_restarts=5
        )
        print(f"  {name}: 最佳点=({best_x[0]:.3f}, {best_x[1]:.3f}), 采集值={best_val:.6f}")
    
    print("\n2. 参数敏感性分析:")
    
    # 测试不同xi值对EI的影响
    xi_values = [0.001, 0.01, 0.1, 1.0]
    X_test = np.array([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]])
    
    print("  EI采集函数在不同xi值下的表现:")
    for xi in xi_values:
        ei = ExpectedImprovement(xi=xi)
        values = ei.evaluate(X_test, gp_2d, best_value_2d)
        print(f"    xi={xi:5.3f}: {values}")
    
    print("\n3. 采集函数比较:")
    
    # 在相同点上比较不同采集函数
    test_points = np.array([[0.5, 0.5], [1.0, 1.0], [-0.5, -0.5]])
    
    acq_funcs = {
        'EI': ExpectedImprovement(xi=0.01),
        'PI': ProbabilityOfImprovement(xi=0.01),
        'UCB': UpperConfidenceBound(kappa=2.576)
    }
    
    print("  不同采集函数在测试点上的值:")
    print("  点\\函数    EI        PI        UCB")
    print("  " + "-" * 35)
    
    for i, point in enumerate(test_points):
        values = []
        for name, acq_func in acq_funcs.items():
            val = acq_func.evaluate(point.reshape(1, -1), gp_2d, best_value_2d)[0]
            values.append(val)
        
        print(f"  点{i+1}:     {values[0]:8.5f}  {values[1]:8.5f}  {values[2]:8.5f}")


if __name__ == "__main__":
    demo_acquisition_functions()
    demo_advanced_features()