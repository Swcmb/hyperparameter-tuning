"""
采集函数属性测试

使用属性测试（Property-Based Testing）验证采集函数的正确性属性。
测试采集函数支持和参数验证功能。

Feature: bayesian-hyperparameter-optimization
"""

import unittest
import numpy as np
import sys
import os
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acquisition_function import (
    ExpectedImprovement, ProbabilityOfImprovement, 
    UpperConfidenceBound, EntropySearch,
    create_acquisition_function, AcquisitionOptimizer
)
from gaussian_process import GaussianProcess


class TestAcquisitionProperties(unittest.TestCase):
    """采集函数属性测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个简单的高斯过程模型用于测试
        np.random.seed(42)
        X_train = np.random.uniform(-2, 2, (5, 2))
        y_train = np.sum(X_train**2, axis=1) + 0.1 * np.random.randn(5)
        
        self.gp = GaussianProcess(random_state=42)
        self.gp.fit(X_train, y_train)
        self.best_value = np.max(y_train)
    
    @given(
        function_type=st.sampled_from(['EI', 'PI', 'UCB', 'ES']),
        n_points=st.integers(min_value=1, max_value=10),
        n_dims=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=3, deadline=None)
    def test_property_14_acquisition_function_support(self, function_type, n_points, n_dims):
        """
        属性 14: 采集函数支持
        
        对于任何支持的采集函数类型（EI、PI、UCB、ES），系统应该正确实现该采集函数并用于参数选择
        
        **Feature: bayesian-hyperparameter-optimization, Property 14: 采集函数支持**
        **验证: 需求 6.1**
        """
        # 生成测试数据
        X_test = np.random.uniform(-2, 2, (n_points, n_dims))
        
        # 为不同维度创建对应的高斯过程模型
        if n_dims != 2:
            # 重新训练适合当前维度的模型
            X_train_dim = np.random.uniform(-2, 2, (5, n_dims))
            y_train_dim = np.sum(X_train_dim**2, axis=1) + 0.1 * np.random.randn(5)
            gp_dim = GaussianProcess(random_state=42)
            gp_dim.fit(X_train_dim, y_train_dim)
            best_value_dim = np.max(y_train_dim)
        else:
            gp_dim = self.gp
            best_value_dim = self.best_value
        
        try:
            # 1. 系统应该能够创建指定类型的采集函数
            if function_type == 'EI':
                acq_func = create_acquisition_function('EI', xi=0.01)
                self.assertIsInstance(acq_func, ExpectedImprovement)
            elif function_type == 'PI':
                acq_func = create_acquisition_function('PI', xi=0.01)
                self.assertIsInstance(acq_func, ProbabilityOfImprovement)
            elif function_type == 'UCB':
                acq_func = create_acquisition_function('UCB', kappa=2.576)
                self.assertIsInstance(acq_func, UpperConfidenceBound)
            elif function_type == 'ES':
                acq_func = create_acquisition_function('ES', n_samples=50)
                self.assertIsInstance(acq_func, EntropySearch)
            
            # 2. 采集函数应该能够正确评估候选点
            acquisition_values = acq_func.evaluate(X_test, gp_dim, best_value_dim)
            
            # 验证输出格式正确
            self.assertEqual(len(acquisition_values), n_points)
            self.assertTrue(np.all(np.isfinite(acquisition_values)))  # 所有值都应该是有限的
            
            # 3. 不同采集函数应该有合理的值域
            if function_type == 'EI':
                # EI值应该非负
                self.assertTrue(np.all(acquisition_values >= 0))
            elif function_type == 'PI':
                # PI值应该在[0,1]范围内
                self.assertTrue(np.all(acquisition_values >= 0))
                self.assertTrue(np.all(acquisition_values <= 1))
            elif function_type == 'UCB':
                # UCB值可以是任意实数，但应该是有限的
                self.assertTrue(np.all(np.isfinite(acquisition_values)))
            elif function_type == 'ES':
                # ES值可以是任意实数，但应该是有限的
                self.assertTrue(np.all(np.isfinite(acquisition_values)))
            
            # 4. 采集函数应该能够用于参数选择（优化）
            bounds = [(-2.0, 2.0)] * n_dims
            try:
                best_x, best_val = acq_func.optimize_acquisition(
                    gp_dim, bounds, best_value_dim, n_restarts=2
                )
                
                # 验证优化结果
                self.assertEqual(len(best_x), n_dims)
                for i, (low, high) in enumerate(bounds):
                    self.assertTrue(low <= best_x[i] <= high)
                self.assertTrue(np.isfinite(best_val))
                
            except Exception as e:
                # 如果优化失败，至少应该能够评估采集函数
                self.fail(f"采集函数优化失败: {e}")
                
        except Exception as e:
            self.fail(f"采集函数 {function_type} 实现不正确: {e}")
    
    @given(
        function_type=st.sampled_from(['EI', 'PI', 'UCB', 'ES']),
        param_changes=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=3, deadline=None)
    def test_property_15_acquisition_parameter_validation(self, function_type, param_changes):
        """
        属性 15: 采集函数参数验证
        
        对于任何采集函数参数的调整，系统应该验证参数有效性并正确应用新设置
        
        **Feature: bayesian-hyperparameter-optimization, Property 15: 采集函数参数验证**
        **验证: 需求 6.3**
        """
        # 创建采集函数
        if function_type == 'EI':
            acq_func = ExpectedImprovement(xi=0.01)
            valid_params = [0.001, 0.01, 0.1, 1.0, 10.0]
            invalid_params = [-0.1, -1.0, -10.0]
            param_name = 'xi'
            update_method = acq_func.update_xi
        elif function_type == 'PI':
            acq_func = ProbabilityOfImprovement(xi=0.01)
            valid_params = [0.001, 0.01, 0.1, 1.0, 10.0]
            invalid_params = [-0.1, -1.0, -10.0]
            param_name = 'xi'
            update_method = acq_func.update_xi
        elif function_type == 'UCB':
            acq_func = UpperConfidenceBound(kappa=2.576)
            valid_params = [0.1, 1.0, 2.576, 5.0, 10.0]
            invalid_params = [0.0, -1.0, -10.0]
            param_name = 'kappa'
            update_method = acq_func.update_kappa
        elif function_type == 'ES':
            acq_func = EntropySearch(n_samples=100, temperature=1.0)
            # ES有两个参数，随机选择一个测试
            if np.random.random() < 0.5:
                valid_params = [10, 50, 100, 200, 500]
                invalid_params = [0, -1, -10]
                param_name = 'n_samples'
                update_method = acq_func.update_n_samples
            else:
                valid_params = [0.1, 0.5, 1.0, 2.0, 5.0]
                invalid_params = [0.0, -0.1, -1.0]
                param_name = 'temperature'
                update_method = acq_func.update_temperature
        
        # 测试多次参数变更
        for _ in range(min(param_changes, len(valid_params))):
            # 1. 测试有效参数的验证和应用
            valid_param = np.random.choice(valid_params)
            
            try:
                # 更新参数
                update_method(valid_param)
                
                # 验证参数已正确应用
                if param_name == 'xi':
                    self.assertEqual(acq_func.xi, valid_param)
                elif param_name == 'kappa':
                    self.assertEqual(acq_func.kappa, valid_param)
                elif param_name == 'n_samples':
                    self.assertEqual(acq_func.n_samples, valid_param)
                elif param_name == 'temperature':
                    self.assertEqual(acq_func.temperature, valid_param)
                
                # 验证参数验证函数返回True
                self.assertTrue(acq_func.validate_parameters())
                
                # 验证更新后的采集函数仍能正常工作
                X_test = np.random.uniform(-2, 2, (3, 2))
                acquisition_values = acq_func.evaluate(X_test, self.gp, self.best_value)
                self.assertEqual(len(acquisition_values), 3)
                self.assertTrue(np.all(np.isfinite(acquisition_values)))
                
            except Exception as e:
                self.fail(f"有效参数 {valid_param} 应该被接受，但出现错误: {e}")
        
        # 2. 测试无效参数的拒绝
        for invalid_param in invalid_params[:min(2, len(invalid_params))]:  # 限制测试数量
            with self.assertRaises(ValueError, msg=f"无效参数 {invalid_param} 应该被拒绝"):
                update_method(invalid_param)
    
    @given(
        xi_values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=5),
            elements=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=3, deadline=None)
    def test_ei_parameter_consistency(self, xi_values):
        """测试EI采集函数参数一致性"""
        ei = ExpectedImprovement(xi=0.01)
        X_test = np.random.uniform(-2, 2, (3, 2))
        
        for xi in xi_values:
            # 更新参数
            ei.update_xi(float(xi))
            
            # 验证参数一致性
            self.assertEqual(ei.xi, float(xi))
            self.assertEqual(ei.params['xi'], float(xi))
            
            # 验证采集函数仍能正常工作
            values = ei.evaluate(X_test, self.gp, self.best_value)
            self.assertTrue(np.all(values >= 0))  # EI值应该非负
    
    @given(
        kappa_values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=5),
            elements=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=3, deadline=None)
    def test_ucb_parameter_consistency(self, kappa_values):
        """测试UCB采集函数参数一致性"""
        ucb = UpperConfidenceBound(kappa=2.576)
        X_test = np.random.uniform(-2, 2, (3, 2))
        
        for kappa in kappa_values:
            # 更新参数
            ucb.update_kappa(float(kappa))
            
            # 验证参数一致性
            self.assertEqual(ucb.kappa, float(kappa))
            self.assertEqual(ucb.params['kappa'], float(kappa))
            
            # 验证采集函数仍能正常工作
            values = ucb.evaluate(X_test, self.gp, self.best_value)
            self.assertTrue(np.all(np.isfinite(values)))
    
    def test_acquisition_function_factory_robustness(self):
        """测试采集函数工厂函数的鲁棒性"""
        # 测试所有支持的类型
        supported_types = ['EI', 'PI', 'UCB', 'ES']
        
        for func_type in supported_types:
            # 测试大小写不敏感
            for case_variant in [func_type.lower(), func_type.upper(), func_type]:
                try:
                    acq_func = create_acquisition_function(case_variant)
                    self.assertEqual(acq_func.function_type, func_type)
                except Exception as e:
                    self.fail(f"工厂函数应该支持 {case_variant}: {e}")
        
        # 测试不支持的类型
        unsupported_types = ['UNKNOWN', 'INVALID', 'TEST', '']
        for invalid_type in unsupported_types:
            with self.assertRaises(ValueError):
                create_acquisition_function(invalid_type)
    
    def test_acquisition_optimization_robustness(self):
        """测试采集函数优化的鲁棒性"""
        ei = ExpectedImprovement(xi=0.01)
        
        # 测试不同的边界条件
        test_bounds = [
            [(-1.0, 1.0), (-1.0, 1.0)],  # 标准边界
            [(-10.0, 10.0), (-10.0, 10.0)],  # 大范围边界
            [(-0.1, 0.1), (-0.1, 0.1)],  # 小范围边界
            [(0.0, 1.0), (0.0, 1.0)],  # 非对称边界
        ]
        
        for bounds in test_bounds:
            try:
                best_x, best_val = ei.optimize_acquisition(
                    self.gp, bounds, self.best_value, n_restarts=2
                )
                
                # 验证结果在边界内
                for i, (low, high) in enumerate(bounds):
                    self.assertTrue(low <= best_x[i] <= high, 
                                  f"优化结果 {best_x[i]} 超出边界 [{low}, {high}]")
                
                # 验证采集函数值是有限的
                self.assertTrue(np.isfinite(best_val))
                
            except Exception as e:
                self.fail(f"边界 {bounds} 的优化失败: {e}")


if __name__ == '__main__':
    # 运行属性测试
    unittest.main(verbosity=2)