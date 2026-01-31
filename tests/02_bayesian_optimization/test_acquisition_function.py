"""
采集函数测试

测试采集函数与高斯过程模型的集成，验证参数验证和优化功能。
"""

import unittest
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


class TestAcquisitionFunction(unittest.TestCase):
    """采集函数测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        np.random.seed(42)
        self.X_train = np.random.uniform(-2, 2, (10, 2))
        self.y_train = np.sum(self.X_train**2, axis=1) + 0.1 * np.random.randn(10)
        
        # 创建并训练高斯过程模型
        self.gp = GaussianProcess(random_state=42)
        self.gp.fit(self.X_train, self.y_train)
        
        # 测试点
        self.X_test = np.array([[0.5, 0.5], [1.0, 1.0], [-0.5, -0.5]])
        self.best_value = np.max(self.y_train)
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    
    def test_expected_improvement(self):
        """测试Expected Improvement采集函数"""
        # 创建EI采集函数
        ei = ExpectedImprovement(xi=0.01)
        
        # 测试基本信息
        self.assertEqual(ei.function_type, 'EI')
        self.assertEqual(ei.xi, 0.01)
        self.assertTrue(ei.validate_parameters())
        
        # 测试评估
        ei_values = ei.evaluate(self.X_test, self.gp, self.best_value)
        self.assertEqual(len(ei_values), len(self.X_test))
        self.assertTrue(np.all(ei_values >= 0))  # EI值应该非负
        
        # 测试参数更新
        ei.update_xi(0.1)
        self.assertEqual(ei.xi, 0.1)
        
        # 测试无效参数
        with self.assertRaises(ValueError):
            ei.update_xi(-0.1)
    
    def test_probability_of_improvement(self):
        """测试Probability of Improvement采集函数"""
        # 创建PI采集函数
        pi = ProbabilityOfImprovement(xi=0.01)
        
        # 测试基本信息
        self.assertEqual(pi.function_type, 'PI')
        self.assertEqual(pi.xi, 0.01)
        self.assertTrue(pi.validate_parameters())
        
        # 测试评估
        pi_values = pi.evaluate(self.X_test, self.gp, self.best_value)
        self.assertEqual(len(pi_values), len(self.X_test))
        self.assertTrue(np.all(pi_values >= 0))  # PI值应该非负
        self.assertTrue(np.all(pi_values <= 1))  # PI值应该不超过1
    
    def test_upper_confidence_bound(self):
        """测试Upper Confidence Bound采集函数"""
        # 创建UCB采集函数
        ucb = UpperConfidenceBound(kappa=2.576)
        
        # 测试基本信息
        self.assertEqual(ucb.function_type, 'UCB')
        self.assertEqual(ucb.kappa, 2.576)
        self.assertTrue(ucb.validate_parameters())
        
        # 测试评估
        ucb_values = ucb.evaluate(self.X_test, self.gp, self.best_value)
        self.assertEqual(len(ucb_values), len(self.X_test))
        
        # 测试参数更新
        ucb.update_kappa(1.96)
        self.assertEqual(ucb.kappa, 1.96)
        
        # 测试无效参数
        with self.assertRaises(ValueError):
            ucb.update_kappa(0)
    
    def test_entropy_search(self):
        """测试Entropy Search采集函数"""
        # 创建ES采集函数
        es = EntropySearch(n_samples=50, temperature=1.0)
        
        # 测试基本信息
        self.assertEqual(es.function_type, 'ES')
        self.assertEqual(es.n_samples, 50)
        self.assertEqual(es.temperature, 1.0)
        self.assertTrue(es.validate_parameters())
        
        # 测试评估
        es_values = es.evaluate(self.X_test, self.gp, self.best_value)
        self.assertEqual(len(es_values), len(self.X_test))
        
        # 测试参数更新
        es.update_n_samples(100)
        self.assertEqual(es.n_samples, 100)
        
        es.update_temperature(0.5)
        self.assertEqual(es.temperature, 0.5)
        
        # 测试无效参数
        with self.assertRaises(ValueError):
            es.update_n_samples(0)
        
        with self.assertRaises(ValueError):
            es.update_temperature(0)
    
    def test_factory_function(self):
        """测试采集函数工厂函数"""
        # 测试创建不同类型的采集函数
        ei = create_acquisition_function('EI', xi=0.05)
        self.assertIsInstance(ei, ExpectedImprovement)
        self.assertEqual(ei.xi, 0.05)
        
        pi = create_acquisition_function('PI', xi=0.02)
        self.assertIsInstance(pi, ProbabilityOfImprovement)
        self.assertEqual(pi.xi, 0.02)
        
        ucb = create_acquisition_function('UCB', kappa=1.96)
        self.assertIsInstance(ucb, UpperConfidenceBound)
        self.assertEqual(ucb.kappa, 1.96)
        
        es = create_acquisition_function('ES', n_samples=200)
        self.assertIsInstance(es, EntropySearch)
        self.assertEqual(es.n_samples, 200)
        
        # 测试不支持的类型
        with self.assertRaises(ValueError):
            create_acquisition_function('UNKNOWN')
    
    def test_acquisition_optimization(self):
        """测试采集函数优化"""
        # 创建采集函数
        ei = ExpectedImprovement(xi=0.01)
        
        # 测试优化
        best_x, best_value = ei.optimize_acquisition(
            self.gp, self.bounds, self.best_value, n_restarts=3
        )
        
        # 验证结果
        self.assertEqual(len(best_x), 2)  # 2维参数
        self.assertTrue(self.bounds[0][0] <= best_x[0] <= self.bounds[0][1])
        self.assertTrue(self.bounds[1][0] <= best_x[1] <= self.bounds[1][1])
        self.assertGreaterEqual(best_value, 0)  # EI值应该非负
        
        # 测试无效边界
        with self.assertRaises(ValueError):
            ei.optimize_acquisition(self.gp, [], self.best_value)
        
        with self.assertRaises(ValueError):
            ei.optimize_acquisition(self.gp, [(2, 1)], self.best_value)  # low > high
    
    def test_acquisition_optimizer(self):
        """测试采集函数优化器"""
        # 创建优化器
        optimizer = AcquisitionOptimizer(method='L-BFGS-B', n_restarts=3)
        
        # 创建采集函数
        ei = ExpectedImprovement(xi=0.01)
        
        # 测试优化
        best_x, best_value = optimizer.optimize(ei, self.gp, self.bounds, self.best_value)
        
        # 验证结果
        self.assertEqual(len(best_x), 2)
        self.assertTrue(self.bounds[0][0] <= best_x[0] <= self.bounds[0][1])
        self.assertTrue(self.bounds[1][0] <= best_x[1] <= self.bounds[1][1])
        
        # 测试网格搜索
        grid_optimizer = AcquisitionOptimizer(method='grid_search')
        best_x_grid, best_value_grid = grid_optimizer.optimize(
            ei, self.gp, self.bounds, self.best_value
        )
        
        self.assertEqual(len(best_x_grid), 2)
        self.assertTrue(self.bounds[0][0] <= best_x_grid[0] <= self.bounds[0][1])
        self.assertTrue(self.bounds[1][0] <= best_x_grid[1] <= self.bounds[1][1])
        
        # 测试不支持的方法
        with self.assertRaises(ValueError):
            AcquisitionOptimizer(method='UNKNOWN')
    
    def test_parameter_validation(self):
        """测试参数验证功能"""
        # 测试EI参数验证
        ei_valid = ExpectedImprovement(xi=0.01)
        self.assertTrue(ei_valid.validate_parameters())
        
        ei_invalid = ExpectedImprovement(xi=-0.01)
        self.assertFalse(ei_invalid.validate_parameters())
        
        # 测试UCB参数验证
        ucb_valid = UpperConfidenceBound(kappa=2.576)
        self.assertTrue(ucb_valid.validate_parameters())
        
        ucb_invalid = UpperConfidenceBound(kappa=-1.0)
        self.assertFalse(ucb_invalid.validate_parameters())
        
        # 测试ES参数验证
        es_valid = EntropySearch(n_samples=100, temperature=1.0)
        self.assertTrue(es_valid.validate_parameters())
        
        es_invalid1 = EntropySearch(n_samples=0, temperature=1.0)
        self.assertFalse(es_invalid1.validate_parameters())
        
        es_invalid2 = EntropySearch(n_samples=100, temperature=0.0)
        self.assertFalse(es_invalid2.validate_parameters())
    
    def test_acquisition_function_comparison(self):
        """测试不同采集函数的比较"""
        # 创建不同的采集函数
        ei = ExpectedImprovement(xi=0.01)
        pi = ProbabilityOfImprovement(xi=0.01)
        ucb = UpperConfidenceBound(kappa=2.576)
        
        # 在相同点上评估
        ei_values = ei.evaluate(self.X_test, self.gp, self.best_value)
        pi_values = pi.evaluate(self.X_test, self.gp, self.best_value)
        ucb_values = ucb.evaluate(self.X_test, self.gp, self.best_value)
        
        # 验证值的范围
        self.assertTrue(np.all(ei_values >= 0))
        self.assertTrue(np.all(pi_values >= 0))
        self.assertTrue(np.all(pi_values <= 1))
        # UCB可以是任意值
        
        # 验证形状一致
        self.assertEqual(ei_values.shape, pi_values.shape)
        self.assertEqual(pi_values.shape, ucb_values.shape)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试单点评估
        single_point = np.array([[0.0, 0.0]])
        ei = ExpectedImprovement(xi=0.01)
        
        ei_value = ei.evaluate(single_point, self.gp, self.best_value)
        self.assertEqual(len(ei_value), 1)
        self.assertGreaterEqual(ei_value[0], 0)
        
        # 测试极小的xi值
        ei_small = ExpectedImprovement(xi=1e-10)
        ei_values_small = ei_small.evaluate(self.X_test, self.gp, self.best_value)
        self.assertTrue(np.all(ei_values_small >= 0))
        
        # 测试极大的kappa值
        ucb_large = UpperConfidenceBound(kappa=100.0)
        ucb_values_large = ucb_large.evaluate(self.X_test, self.gp, self.best_value)
        self.assertEqual(len(ucb_values_large), len(self.X_test))


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)