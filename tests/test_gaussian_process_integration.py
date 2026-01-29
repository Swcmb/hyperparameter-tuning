"""
高斯过程模型集成测试

测试高斯过程模型与现有autodl_core系统的集成
"""

import unittest
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autodl_core import create_default_parameter_space, OptimizationResult, OptimizationHistory
from gaussian_process import GaussianProcess, create_default_gaussian_process


class TestGaussianProcessIntegration(unittest.TestCase):
    """测试高斯过程与autodl_core的集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.parameter_space = create_default_parameter_space()
        self.gp = create_default_gaussian_process(random_state=42)
        
    def test_parameter_space_to_gp_input(self):
        """测试参数空间到高斯过程输入的转换"""
        # 生成一些参数样本
        params_list = []
        for i in range(5):
            params = self.parameter_space.sample_random_parameters(seed=42+i)
            params_list.append(params)
        
        # 转换为高斯过程输入格式
        X = self._convert_params_to_array(params_list)
        
        # 验证转换结果
        self.assertEqual(X.shape[0], 5)  # 5个样本
        self.assertEqual(X.shape[1], len(self.parameter_space.get_continuous_parameter_names()))  # 连续参数数量
        
        # 验证数据类型
        self.assertTrue(np.isfinite(X).all())
        
    def test_gp_with_optimization_results(self):
        """测试高斯过程与优化结果的集成"""
        # 创建一些模拟的优化结果
        results = []
        X_data = []
        y_data = []
        
        for i in range(10):
            params = self.parameter_space.sample_random_parameters(seed=42+i)
            
            # 模拟目标函数值（基于参数的简单函数）
            objective_value = self._mock_objective_function(params)
            
            result = OptimizationResult(
                parameters=params,
                objective_value=objective_value,
                metrics={'AUROC': objective_value, 'AUPRC': objective_value-0.1, 'F1': objective_value-0.05},
                iteration=i+1,
                timestamp=None,
                evaluation_time=10.0
            )
            results.append(result)
            
            # 准备高斯过程训练数据
            X_data.append(self._convert_params_to_array([params])[0])
            y_data.append(objective_value)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # 训练高斯过程
        self.gp.fit(X_data, y_data)
        
        # 验证训练成功
        self.assertTrue(self.gp.is_fitted)
        self.assertEqual(self.gp.n_observations, 10)
        
        # 测试预测
        test_params = self.parameter_space.sample_random_parameters(seed=100)
        X_test = self._convert_params_to_array([test_params])
        
        mean, std = self.gp.predict(X_test)
        
        # 验证预测结果
        self.assertEqual(len(mean), 1)
        self.assertEqual(len(std), 1)
        self.assertTrue(np.isfinite(mean[0]))
        self.assertTrue(np.isfinite(std[0]))
        self.assertGreater(std[0], 0)  # 标准差应该大于0
        
    def test_acquisition_function_integration(self):
        """测试采集函数与参数空间的集成"""
        # 准备训练数据
        X_data = []
        y_data = []
        
        for i in range(8):
            params = self.parameter_space.sample_random_parameters(seed=42+i)
            X_data.append(self._convert_params_to_array([params])[0])
            y_data.append(self._mock_objective_function(params))
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # 训练高斯过程
        self.gp.fit(X_data, y_data)
        
        # 生成候选点
        candidate_params = []
        for i in range(5):
            params = self.parameter_space.sample_random_parameters(seed=200+i)
            candidate_params.append(params)
        
        X_candidates = np.array([self._convert_params_to_array([p])[0] for p in candidate_params])
        
        # 测试不同采集函数
        ei_values = self.gp.compute_acquisition_values(X_candidates, 'EI')
        pi_values = self.gp.compute_acquisition_values(X_candidates, 'PI')
        ucb_values = self.gp.compute_acquisition_values(X_candidates, 'UCB')
        
        # 验证采集函数值
        self.assertEqual(len(ei_values), 5)
        self.assertEqual(len(pi_values), 5)
        self.assertEqual(len(ucb_values), 5)
        
        # 所有值都应该是有限的非负数
        self.assertTrue(np.isfinite(ei_values).all())
        self.assertTrue(np.isfinite(pi_values).all())
        self.assertTrue(np.isfinite(ucb_values).all())
        self.assertTrue((ei_values >= 0).all())
        self.assertTrue((pi_values >= 0).all())
        
    def test_optimization_history_integration(self):
        """测试与优化历史的集成"""
        history = OptimizationHistory()
        history.task_type = "LDA"
        history.acquisition_function = "EI"
        
        # 模拟优化过程
        X_data = []
        y_data = []
        
        for i in range(15):
            params = self.parameter_space.sample_random_parameters(seed=42+i)
            objective_value = self._mock_objective_function(params)
            
            result = OptimizationResult(
                parameters=params,
                objective_value=objective_value,
                metrics={'AUROC': objective_value, 'AUPRC': objective_value-0.1},
                iteration=i+1,
                timestamp=None,
                evaluation_time=5.0
            )
            
            history.add_result(result)
            X_data.append(self._convert_params_to_array([params])[0])
            y_data.append(objective_value)
        
        # 使用历史数据训练高斯过程
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        self.gp.fit(X_data, y_data)
        
        # 验证集成
        self.assertEqual(history.total_iterations, 15)
        self.assertEqual(self.gp.n_observations, 15)
        self.assertIsNotNone(history.best_result)
        
        # 验证收敛曲线
        convergence = history.get_convergence_curve()
        self.assertEqual(len(convergence), 15)
        
        # 最后的收敛值应该等于最佳结果
        self.assertEqual(convergence[-1], history.get_best_objective_value())
        
    def test_model_persistence_integration(self):
        """测试模型持久化与系统集成"""
        # 训练模型
        X_data = []
        y_data = []
        
        for i in range(6):
            params = self.parameter_space.sample_random_parameters(seed=42+i)
            X_data.append(self._convert_params_to_array([params])[0])
            y_data.append(self._mock_objective_function(params))
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        self.gp.fit(X_data, y_data)
        original_hyperparams = self.gp.get_hyperparameters()
        
        # 保存和加载模型
        model_path = 'test_integration_model.pkl'
        try:
            self.gp.save_model(model_path)
            loaded_gp = GaussianProcess.load_model(model_path)
            
            # 验证加载的模型
            self.assertTrue(loaded_gp.is_fitted)
            self.assertEqual(loaded_gp.n_observations, 6)
            
            # 验证预测一致性
            test_params = self.parameter_space.sample_random_parameters(seed=999)
            X_test = self._convert_params_to_array([test_params])
            
            mean1, std1 = self.gp.predict(X_test)
            mean2, std2 = loaded_gp.predict(X_test)
            
            np.testing.assert_array_almost_equal(mean1, mean2, decimal=10)
            np.testing.assert_array_almost_equal(std1, std2, decimal=10)
            
        finally:
            # 清理测试文件
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def _convert_params_to_array(self, params_list):
        """
        将参数字典列表转换为numpy数组（仅包含连续参数）
        
        这是一个简化的转换函数，实际系统中需要更复杂的处理
        """
        continuous_names = self.parameter_space.get_continuous_parameter_names()
        
        arrays = []
        for params in params_list:
            row = []
            for name in continuous_names:
                value = params[name]
                # 处理对数尺度参数
                if name in ['lr', 'weight_decay'] and value > 0:
                    row.append(np.log(value))
                else:
                    row.append(float(value))
            arrays.append(row)
        
        return np.array(arrays)
    
    def _mock_objective_function(self, params):
        """
        模拟目标函数
        
        基于参数生成一个模拟的AUROC值
        """
        # 简单的模拟函数，基于一些关键参数
        lr = float(params['lr'])
        dropout = float(params['dropout'])
        alpha = float(params['alpha'])
        
        # 模拟一个有噪声的目标函数
        base_score = 0.7
        lr_effect = -0.1 * abs(np.log10(lr) + 3)  # lr在1e-3附近最优
        dropout_effect = -0.05 * (dropout - 0.3)**2  # dropout在0.3附近最优
        alpha_effect = 0.02 * alpha
        noise = np.random.RandomState(hash(str(params)) % 2**32).normal(0, 0.02)
        
        score = base_score + lr_effect + dropout_effect + alpha_effect + noise
        return max(0.5, min(0.95, score))  # 限制在合理范围内


if __name__ == '__main__':
    unittest.main()