"""
核心数据结构和配置管理的综合测试

测试OptimizationResult、OptimizationHistory、ParameterConfig和相关功能
"""

import unittest
import numpy as np
from datetime import datetime
from autodl_core import (
    ParameterConfig, ParameterType, OptimizationResult, 
    OptimizationHistory, ParameterSpace, create_default_parameter_space
)
from parameter_validator import ParameterValidator, ConfigurationConverter


class TestParameterConfig(unittest.TestCase):
    """测试ParameterConfig类"""
    
    def test_continuous_parameter(self):
        """测试连续型参数"""
        config = ParameterConfig(
            name="lr",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-2),
            log_scale=True
        )
        
        # 测试有效值
        self.assertTrue(config.validate_value(1e-4))
        self.assertTrue(config.validate_value(1e-3))
        
        # 测试无效值
        self.assertFalse(config.validate_value(1e-6))
        self.assertFalse(config.validate_value(1e-1))
        
        # 测试随机采样
        rng = np.random.default_rng(42)
        value = config.sample_random_value(rng)
        self.assertTrue(config.validate_value(value))
    
    def test_discrete_parameter(self):
        """测试离散型参数"""
        config = ParameterConfig(
            name="batch_size",
            param_type=ParameterType.DISCRETE,
            values=[16, 32, 64, 128]
        )
        
        # 测试有效值
        self.assertTrue(config.validate_value(32))
        self.assertTrue(config.validate_value(64))
        
        # 测试无效值
        self.assertFalse(config.validate_value(48))
        self.assertFalse(config.validate_value(256))
        
        # 测试随机采样
        rng = np.random.default_rng(42)
        value = config.sample_random_value(rng)
        self.assertIn(value, config.values)
    
    def test_categorical_parameter(self):
        """测试分类型参数"""
        config = ParameterConfig(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            values=["adam", "sgd", "rmsprop"]
        )
        
        # 测试有效值
        self.assertTrue(config.validate_value("adam"))
        self.assertTrue(config.validate_value("sgd"))
        
        # 测试无效值
        self.assertFalse(config.validate_value("adagrad"))
        self.assertFalse(config.validate_value("momentum"))
        
        # 测试随机采样
        rng = np.random.default_rng(42)
        value = config.sample_random_value(rng)
        self.assertIn(value, config.values)
    
    def test_serialization(self):
        """测试序列化和反序列化"""
        config = ParameterConfig(
            name="lr",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-2),
            log_scale=True
        )
        
        # 转换为字典
        config_dict = config.to_dict()
        self.assertEqual(config_dict['name'], "lr")
        self.assertEqual(config_dict['param_type'], "continuous")
        
        # 从字典恢复
        restored_config = ParameterConfig.from_dict(config_dict)
        self.assertEqual(restored_config.name, config.name)
        self.assertEqual(restored_config.param_type, config.param_type)
        self.assertEqual(restored_config.bounds, config.bounds)
        self.assertEqual(restored_config.log_scale, config.log_scale)


class TestOptimizationResult(unittest.TestCase):
    """测试OptimizationResult类"""
    
    def test_result_creation(self):
        """测试结果创建"""
        params = {"lr": 0.001, "batch_size": 32}
        metrics = {"AUROC": 0.85, "AUPRC": 0.78, "F1": 0.72}
        
        result = OptimizationResult(
            parameters=params,
            objective_value=0.85,
            metrics=metrics,
            iteration=1,
            timestamp=datetime.now(),
            evaluation_time=120.5
        )
        
        self.assertEqual(result.parameters, params)
        self.assertEqual(result.objective_value, 0.85)
        self.assertEqual(result.metrics, metrics)
        self.assertEqual(result.iteration, 1)
        self.assertEqual(result.evaluation_time, 120.5)
    
    def test_result_comparison(self):
        """测试结果比较"""
        result1 = OptimizationResult(
            parameters={"lr": 0.001},
            objective_value=0.85,
            metrics={"AUROC": 0.85},
            iteration=1,
            timestamp=datetime.now(),
            evaluation_time=100.0
        )
        
        result2 = OptimizationResult(
            parameters={"lr": 0.002},
            objective_value=0.90,
            metrics={"AUROC": 0.90},
            iteration=2,
            timestamp=datetime.now(),
            evaluation_time=110.0
        )
        
        # 测试最大化比较
        self.assertTrue(result2.is_better_than(result1, maximize=True))
        self.assertFalse(result1.is_better_than(result2, maximize=True))
        
        # 测试最小化比较
        self.assertFalse(result2.is_better_than(result1, maximize=False))
        self.assertTrue(result1.is_better_than(result2, maximize=False))
    
    def test_result_serialization(self):
        """测试结果序列化"""
        result = OptimizationResult(
            parameters={"lr": 0.001},
            objective_value=0.85,
            metrics={"AUROC": 0.85},
            iteration=1,
            timestamp=datetime.now(),
            evaluation_time=100.0
        )
        
        # 转换为字典
        result_dict = result.to_dict()
        self.assertEqual(result_dict['objective_value'], 0.85)
        self.assertEqual(result_dict['iteration'], 1)
        
        # 从字典恢复
        restored_result = OptimizationResult.from_dict(result_dict)
        self.assertEqual(restored_result.objective_value, result.objective_value)
        self.assertEqual(restored_result.parameters, result.parameters)


class TestOptimizationHistory(unittest.TestCase):
    """测试OptimizationHistory类"""
    
    def test_history_management(self):
        """测试历史管理"""
        history = OptimizationHistory()
        
        # 添加结果
        result1 = OptimizationResult(
            parameters={"lr": 0.001},
            objective_value=0.85,
            metrics={"AUROC": 0.85},
            iteration=1,
            timestamp=datetime.now(),
            evaluation_time=100.0
        )
        
        result2 = OptimizationResult(
            parameters={"lr": 0.002},
            objective_value=0.90,
            metrics={"AUROC": 0.90},
            iteration=2,
            timestamp=datetime.now(),
            evaluation_time=110.0
        )
        
        history.add_result(result1)
        self.assertEqual(history.total_iterations, 1)
        self.assertEqual(history.get_best_objective_value(), 0.85)
        
        history.add_result(result2)
        self.assertEqual(history.total_iterations, 2)
        self.assertEqual(history.get_best_objective_value(), 0.90)
    
    def test_convergence_curve(self):
        """测试收敛曲线"""
        history = OptimizationHistory()
        
        # 添加递增的结果
        for i, obj_val in enumerate([0.7, 0.8, 0.75, 0.9, 0.85], 1):
            result = OptimizationResult(
                parameters={"lr": 0.001 * i},
                objective_value=obj_val,
                metrics={"AUROC": obj_val},
                iteration=i,
                timestamp=datetime.now(),
                evaluation_time=100.0
            )
            history.add_result(result)
        
        convergence = history.get_convergence_curve()
        expected = [0.7, 0.8, 0.8, 0.9, 0.9]  # 历史最佳值序列
        self.assertEqual(convergence, expected)
    
    def test_parameter_history(self):
        """测试参数历史"""
        history = OptimizationHistory()
        
        lr_values = [0.001, 0.002, 0.003]
        for i, lr in enumerate(lr_values, 1):
            result = OptimizationResult(
                parameters={"lr": lr, "batch_size": 32},
                objective_value=0.8 + i * 0.01,
                metrics={"AUROC": 0.8 + i * 0.01},
                iteration=i,
                timestamp=datetime.now(),
                evaluation_time=100.0
            )
            history.add_result(result)
        
        lr_history = history.get_parameter_history("lr")
        self.assertEqual(lr_history, lr_values)


class TestParameterSpace(unittest.TestCase):
    """测试ParameterSpace类"""
    
    def test_parameter_space_creation(self):
        """测试参数空间创建"""
        space = ParameterSpace()
        
        # 添加不同类型的参数
        space.add_continuous_parameter("lr", 1e-5, 1e-2, log_scale=True)
        space.add_discrete_parameter("batch_size", [16, 32, 64])
        space.add_categorical_parameter("optimizer", ["adam", "sgd"])
        
        self.assertEqual(len(space.parameters), 3)
        self.assertIn("lr", space.parameters)
        self.assertIn("batch_size", space.parameters)
        self.assertIn("optimizer", space.parameters)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        space = create_default_parameter_space()
        
        # 测试有效参数
        valid_params = space.sample_random_parameters(seed=42)
        self.assertTrue(space.validate_parameters(valid_params))
        
        # 测试无效参数
        invalid_params = valid_params.copy()
        invalid_params["lr"] = 1.0  # 超出范围
        self.assertFalse(space.validate_parameters(invalid_params))
    
    def test_parameter_sampling(self):
        """测试参数采样"""
        space = create_default_parameter_space()
        
        # 测试随机采样
        params1 = space.sample_random_parameters(seed=42)
        params2 = space.sample_random_parameters(seed=42)
        params3 = space.sample_random_parameters(seed=43)
        
        # 相同种子应该产生相同结果
        self.assertEqual(params1, params2)
        
        # 不同种子应该产生不同结果
        self.assertNotEqual(params1, params3)
        
        # 所有采样结果都应该有效
        self.assertTrue(space.validate_parameters(params1))
        self.assertTrue(space.validate_parameters(params3))


class TestParameterValidator(unittest.TestCase):
    """测试ParameterValidator类"""
    
    def setUp(self):
        """设置测试环境"""
        self.space = create_default_parameter_space()
        self.validator = ParameterValidator(self.space)
    
    def test_constraint_validation(self):
        """测试约束验证"""
        # 创建符合约束的参数
        valid_params = {
            'dimensions': 256,
            'hidden1': 128,
            'hidden2': 64,
            'decoder1': 512,
            'lr': 0.001,
            'dropout': 0.1,
            'weight_decay': 0.0005,
            'alpha': 1.0,
            'beta': 0.5,
            'gamma': 0.5,
            'gat_heads': 4,
            'gt_heads': 4,
            'fusion_heads': 4,
            'batch': 32,
            'moco_K': 4096,
            'fusion_strategy': 'self_attention',
            'feature_type': 'normal',
            'moco_type': 'basic'
        }
        
        is_valid, errors = self.validator.validate_parameters(valid_params)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_constraint_violation(self):
        """测试约束违反"""
        # 创建违反约束的参数
        invalid_params = {
            'dimensions': 128,
            'hidden1': 256,  # 违反递减约束
            'hidden2': 64,
            'decoder1': 32,  # 违反解码器约束
            'lr': 0.001,
            'dropout': 0.1,
            'weight_decay': 0.01,  # 违反学习率约束
            'alpha': 0.0,  # 违反损失权重约束
            'beta': 0.0,
            'gamma': 0.0,
            'gat_heads': 3,  # 违反整除约束
            'gt_heads': 4,
            'fusion_heads': 4,
            'batch': 32,
            'moco_K': 64,  # 违反批大小约束
            'fusion_strategy': 'self_attention',
            'feature_type': 'normal',
            'moco_type': 'basic'
        }
        
        is_valid, errors = self.validator.validate_parameters(invalid_params)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_parameter_fix(self):
        """测试参数修复"""
        # 创建需要修复的参数
        broken_params = {
            'dimensions': 128,
            'hidden1': 256,  # 需要修复
            'hidden2': 300,  # 需要修复
            'decoder1': 512,
            'lr': 0.001,
            'dropout': 0.1,
            'weight_decay': 0.0005,
            'alpha': 1.0,
            'beta': 0.5,
            'gamma': 0.5,
            'gat_heads': 3,  # 需要修复
            'gt_heads': 4,
            'fusion_heads': 4,
            'batch': 32,
            'moco_K': 4096,
            'fusion_strategy': 'self_attention',
            'feature_type': 'normal',
            'moco_type': 'basic'
        }
        
        fixed_params = self.validator.suggest_parameter_fix(broken_params)
        
        # 检查修复结果
        self.assertLessEqual(fixed_params['hidden1'], fixed_params['dimensions'])
        self.assertLessEqual(fixed_params['hidden2'], fixed_params['hidden1'])
        self.assertIn(fixed_params['gat_heads'], [2, 4, 8, 16])


class TestConfigurationConverter(unittest.TestCase):
    """测试ConfigurationConverter类"""
    
    def test_config_conversion(self):
        """测试配置转换"""
        converter = ConfigurationConverter("LDA")
        
        params = {
            'dimensions': 256,
            'hidden1': 128,
            'hidden2': 64,
            'decoder1': 512,
            'lr': 0.001,
            'dropout': 0.1,
            'weight_decay': 0.0005,
            'alpha': 1.0,
            'beta': 0.5,
            'gamma': 0.5,
            'gat_heads': 4,
            'gt_heads': 4,
            'fusion_heads': 4,
            'batch': 32,
            'moco_K': 4096,
            'fusion_strategy': 'self_attention',
            'feature_type': 'normal',
            'moco_type': 'basic'
        }
        
        config = converter.convert_to_experiment_config(params)
        
        # 检查关键参数是否正确转换
        self.assertEqual(config['dimensions'], 256)
        self.assertEqual(config['lr'], 0.001)
        self.assertEqual(config['batch'], 32)
        self.assertEqual(config['task_type'], "LDA")
        self.assertEqual(config['proj_dim'], 64)  # 应该跟随hidden2
        
        # 检查任务特定配置
        self.assertIn("LDA.edgelist", config['in_file'])
    
    def test_config_validation(self):
        """测试配置验证"""
        converter = ConfigurationConverter("LDA")
        
        # 完整配置
        complete_config = {
            'dimensions': 256,
            'hidden1': 128,
            'hidden2': 64,
            'decoder1': 512,
            'lr': 0.001,
            'dropout': 0.1,
            'weight_decay': 0.0005,
            'batch': 32,
            'gat_heads': 4,
            'gt_heads': 4,
            'fusion_heads': 4,
            'alpha': 1.0,
            'beta': 0.5,
            'gamma': 0.5,
            'task_type': 'LDA'
        }
        
        is_valid, errors = converter.validate_experiment_config(complete_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # 不完整配置
        incomplete_config = {'dimensions': 256}
        is_valid, errors = converter.validate_experiment_config(incomplete_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)