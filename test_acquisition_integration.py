"""
采集函数集成测试

测试新的采集函数实现与现有高斯过程模型的集成。
验证采集函数能够正确与高斯过程模型协作进行贝叶斯优化。
"""

import numpy as np
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


def test_acquisition_gp_integration():
    """测试采集函数与高斯过程的集成"""
    print("测试采集函数与高斯过程模型的集成...")
    
    # 创建测试数据
    np.random.seed(42)
    
    # 定义一个简单的测试函数 (Branin function的简化版本)
    def test_function(x):
        x1, x2 = x[0], x[1]
        return -(x2 - (5.1/(4*np.pi**2))*x1**2 + (5/np.pi)*x1 - 6)**2 - 10*(1 - 1/(8*np.pi))*np.cos(x1) - 10
    
    # 初始训练数据
    X_train = np.random.uniform(-5, 10, (5, 2))
    y_train = np.array([test_function(x) for x in X_train])
    
    # 创建高斯过程模型
    gp = GaussianProcess(random_state=42)
    gp.fit(X_train, y_train)
    
    print(f"初始训练数据: {len(X_train)} 个点")
    print(f"当前最佳值: {np.max(y_train):.4f}")
    
    # 测试所有采集函数
    acquisition_functions = [
        ('EI', ExpectedImprovement(xi=0.01)),
        ('PI', ProbabilityOfImprovement(xi=0.01)),
        ('UCB', UpperConfidenceBound(kappa=2.576)),
        ('ES', EntropySearch(n_samples=50))
    ]
    
    bounds = [(-5.0, 10.0), (-5.0, 15.0)]
    
    for name, acq_func in acquisition_functions:
        print(f"\n测试 {name} 采集函数:")
        
        try:
            # 1. 测试采集函数评估
            X_candidates = np.random.uniform(-5, 10, (10, 2))
            acquisition_values = acq_func.evaluate(X_candidates, gp, np.max(y_train))
            
            print(f"  候选点采集函数值范围: [{np.min(acquisition_values):.6f}, {np.max(acquisition_values):.6f}]")
            
            # 2. 测试采集函数优化
            best_x, best_acq_val = acq_func.optimize_acquisition(
                gp, bounds, np.max(y_train), n_restarts=5
            )
            
            print(f"  优化找到的最佳点: [{best_x[0]:.4f}, {best_x[1]:.4f}]")
            print(f"  对应的采集函数值: {best_acq_val:.6f}")
            
            # 3. 验证优化结果
            # 评估真实函数值
            true_value = test_function(best_x)
            print(f"  该点的真实函数值: {true_value:.4f}")
            
            # 验证点在边界内
            assert bounds[0][0] <= best_x[0] <= bounds[0][1], f"x1 超出边界"
            assert bounds[1][0] <= best_x[1] <= bounds[1][1], f"x2 超出边界"
            
            # 4. 测试模型更新
            gp_updated = GaussianProcess(random_state=42)
            X_new = np.vstack([X_train, best_x.reshape(1, -1)])
            y_new = np.append(y_train, true_value)
            gp_updated.fit(X_new, y_new)
            
            print(f"  更新后模型观测数量: {gp_updated.n_observations}")
            
            # 5. 测试更新后的采集函数
            new_acquisition_values = acq_func.evaluate(X_candidates, gp_updated, np.max(y_new))
            print(f"  更新后采集函数值范围: [{np.min(new_acquisition_values):.6f}, {np.max(new_acquisition_values):.6f}]")
            
            print(f"  ✓ {name} 采集函数集成测试通过")
            
        except Exception as e:
            print(f"  ✗ {name} 采集函数集成测试失败: {e}")
            return False
    
    return True


def test_acquisition_optimizer_integration():
    """测试采集函数优化器的集成"""
    print("\n测试采集函数优化器集成...")
    
    # 创建测试数据
    np.random.seed(42)
    X_train = np.random.uniform(-2, 2, (8, 2))
    y_train = np.sum(X_train**2, axis=1) + 0.1 * np.random.randn(8)
    
    gp = GaussianProcess(random_state=42)
    gp.fit(X_train, y_train)
    
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    best_value = np.max(y_train)
    
    # 测试不同的优化方法
    optimizers = [
        ('L-BFGS-B', AcquisitionOptimizer(method='L-BFGS-B', n_restarts=5)),
        ('differential_evolution', AcquisitionOptimizer(method='differential_evolution', n_restarts=3)),
        ('grid_search', AcquisitionOptimizer(method='grid_search'))
    ]
    
    ei_func = ExpectedImprovement(xi=0.01)
    
    for method_name, optimizer in optimizers:
        print(f"\n测试 {method_name} 优化方法:")
        
        try:
            best_x, best_val = optimizer.optimize(ei_func, gp, bounds, best_value)
            
            print(f"  找到的最佳点: [{best_x[0]:.4f}, {best_x[1]:.4f}]")
            print(f"  采集函数值: {best_val:.6f}")
            
            # 验证结果
            assert len(best_x) == 2, "结果维度不正确"
            assert bounds[0][0] <= best_x[0] <= bounds[0][1], "x1 超出边界"
            assert bounds[1][0] <= best_x[1] <= bounds[1][1], "x2 超出边界"
            assert np.isfinite(best_val), "采集函数值不是有限数"
            
            print(f"  ✓ {method_name} 优化器测试通过")
            
        except Exception as e:
            print(f"  ✗ {method_name} 优化器测试失败: {e}")
            return False
    
    return True


def test_parameter_validation_integration():
    """测试参数验证的集成"""
    print("\n测试参数验证集成...")
    
    # 创建测试数据
    np.random.seed(42)
    X_train = np.random.uniform(-1, 1, (5, 2))
    y_train = np.sum(X_train**2, axis=1)
    
    gp = GaussianProcess(random_state=42)
    gp.fit(X_train, y_train)
    
    X_test = np.array([[0.5, 0.5], [-0.5, -0.5]])
    best_value = np.max(y_train)
    
    # 测试参数验证和更新
    test_cases = [
        ('EI', ExpectedImprovement(xi=0.01), 'update_xi', [0.001, 0.1, 1.0], [-0.1]),
        ('UCB', UpperConfidenceBound(kappa=2.576), 'update_kappa', [0.1, 1.0, 5.0], [0.0, -1.0]),
        ('ES', EntropySearch(n_samples=50), 'update_n_samples', [10, 100, 500], [0, -1])
    ]
    
    for name, acq_func, update_method, valid_params, invalid_params in test_cases:
        print(f"\n测试 {name} 参数验证:")
        
        try:
            # 测试有效参数
            for param in valid_params:
                getattr(acq_func, update_method)(param)
                assert acq_func.validate_parameters(), f"有效参数 {param} 验证失败"
                
                # 测试更新后仍能正常工作
                values = acq_func.evaluate(X_test, gp, best_value)
                assert len(values) == 2, "评估结果长度不正确"
                assert np.all(np.isfinite(values)), "评估结果包含无效值"
            
            print(f"  ✓ 有效参数测试通过")
            
            # 测试无效参数
            for param in invalid_params:
                try:
                    getattr(acq_func, update_method)(param)
                    assert False, f"无效参数 {param} 应该被拒绝"
                except ValueError:
                    pass  # 期望的行为
            
            print(f"  ✓ 无效参数拒绝测试通过")
            print(f"  ✓ {name} 参数验证集成测试通过")
            
        except Exception as e:
            print(f"  ✗ {name} 参数验证集成测试失败: {e}")
            return False
    
    return True


def main():
    """运行所有集成测试"""
    print("=" * 60)
    print("采集函数集成测试")
    print("=" * 60)
    
    tests = [
        test_acquisition_gp_integration,
        test_acquisition_optimizer_integration,
        test_parameter_validation_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"\n✗ 测试 {test.__name__} 失败")
        except Exception as e:
            print(f"\n✗ 测试 {test.__name__} 出现异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"集成测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有集成测试通过！采集函数实现正确。")
        return True
    else:
        print("❌ 部分集成测试失败，需要检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)