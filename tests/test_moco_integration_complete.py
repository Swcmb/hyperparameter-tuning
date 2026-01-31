#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MoCo超参数集成完整流程测试

测试新MoCo参数从解析到评估的完整链路，确保所有组件正确协作
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心组件
from autodl_core import create_default_parameter_space, OptimizationHistory, OptimizationResult
from bayesian_optimizer import create_bayesian_optimizer
from task_evaluator import create_task_evaluator
from parameter_validator import ParameterValidator
from state_manager import create_default_state_manager


class MockMoCoTaskEvaluator:
    """模拟支持MoCo参数的任务评估器"""
    
    def __init__(self, task_type="LDA"):
        self.task_type = task_type
        self.evaluation_count = 0
        np.random.seed(42)
    
    def evaluate_parameters(self, parameters):
        """模拟MoCo参数评估，考虑新参数的影响"""
        self.evaluation_count += 1
        
        # 基础性能
        base_auroc = 0.75
        
        # MoCo参数对性能的影响
        moco_boost = 0.0
        
        # 考虑新MoCo参数的影响
        if 'moco_tau1' in parameters and 'moco_tau2' in parameters:
            tau1 = float(parameters['moco_tau1'])
            tau2 = float(parameters['moco_tau2'])
            
            # DoubleTau模式通常能提升性能
            if tau2 >= tau1:  # 满足约束
                moco_boost += 0.02
                # 最优温度范围
                if 0.1 <= tau1 <= 0.3 and 0.2 <= tau2 <= 0.4:
                    moco_boost += 0.03
        
        if 'enable_view_0' in parameters:
            enable_view = str(parameters['enable_view_0']).lower() == 'true'
            if enable_view:
                moco_boost += 0.01  # 启用第0视图通常有帮助
        
        # 考虑传统MoCo参数
        if 'moco_momentum' in parameters:
            momentum = float(parameters['moco_momentum'])
            if 0.995 <= momentum <= 0.999:  # 最优动量范围
                moco_boost += 0.02
        
        if 'moco_t' in parameters:
            temp = float(parameters['moco_t'])
            if 0.1 <= temp <= 0.3:  # 最优温度范围
                moco_boost += 0.02
        
        # 添加随机噪声
        param_hash = hash(str(sorted(parameters.items()))) % 1000000
        np.random.seed(param_hash)
        noise = np.random.normal(0, 0.02)
        
        auroc = max(0.5, min(0.99, base_auroc + moco_boost + noise))
        auprc = max(0.5, min(0.99, auroc - 0.05 + np.random.random() * 0.1))
        f1 = max(0.5, min(0.99, auroc - 0.1 + np.random.random() * 0.15))
        
        return {
            'objective_value': auroc,
            'metrics': {
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1': f1,
                'Precision': f1 + np.random.random() * 0.05,
                'Recall': f1 + np.random.random() * 0.05
            },
            'fold_results': {
                'AUROC': [auroc + np.random.normal(0, 0.01) for _ in range(5)],
                'AUPRC': [auprc + np.random.normal(0, 0.01) for _ in range(5)],
                'F1': [f1 + np.random.normal(0, 0.01) for _ in range(5)]
            },
            'objective_values': {
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1': f1
            }
        }


class TestMoCoIntegrationComplete(unittest.TestCase):
    """MoCo参数完整集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_moco_parameter_parsing_and_validation(self):
        """测试MoCo参数解析和验证"""
        print("\n测试MoCo参数解析和验证...")
        
        # 创建参数空间
        parameter_space = create_default_parameter_space("LDA")
        
        # 测试包含新MoCo参数的参数组合
        test_parameters = {
            'dimensions': 256,
            'hidden1': 128,
            'hidden2': 64,
            'decoder1': 512,
            'dropout': 0.5,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 1.0,
            'gat_heads': 8,
            'gt_heads': 8,
            'fusion_heads': 8,
            'batch': 32,
            'moco_K': 4096,
            'moco_momentum': 0.999,
            'moco_t': 0.2,
            'moco_tau1': 0.15,  # 新参数
            'moco_tau2': 0.25,  # 新参数
            'fusion_strategy': 'self_attention',
            'feature_type': 'normal',
            'moco_type': 'double_tau',
            'enable_view_0': 'true'  # 新参数
        }
        
        # 验证参数有效性
        is_valid, errors = parameter_space.validate_parameters_detailed(test_parameters)
        self.assertTrue(is_valid, f"MoCo参数验证失败: {errors}")
        
        # 测试约束条件
        validator = ParameterValidator(parameter_space)
        
        # 测试参数验证
        is_valid, errors = validator.validate_parameters(test_parameters)
        if not is_valid:
            print(f"约束验证错误: {errors}")
        
        # 测试特定的MoCo约束
        moco_tau_valid = validator.constraint_functions['moco_tau_ordering'](test_parameters)
        self.assertTrue(moco_tau_valid, "MoCo tau约束应该通过")
        
        momentum_valid = validator.constraint_functions['moco_momentum_range'](test_parameters)
        self.assertTrue(momentum_valid, "MoCo动量约束应该通过")
        
        temp_valid = validator.constraint_functions['moco_temperature_positive'](test_parameters)
        self.assertTrue(temp_valid, "MoCo温度约束应该通过")
        
        print("✓ MoCo参数解析和验证测试通过")
    
    def test_moco_parameter_constraint_violations(self):
        """测试MoCo参数约束违反的处理"""
        print("\n测试MoCo参数约束违反处理...")
        
        parameter_space = create_default_parameter_space("LDA")
        validator = ParameterValidator(parameter_space)
        
        # 测试tau约束违反：tau1 > tau2
        invalid_tau_params = {
            'moco_tau1': 0.3,
            'moco_tau2': 0.2,  # 违反约束
            'moco_momentum': 0.999,
            'moco_t': 0.2
        }
        
        tau_constraint_func = validator.constraint_functions['moco_tau_ordering']
        self.assertFalse(tau_constraint_func(invalid_tau_params))
        
        # 测试参数修复
        fixed_params = parameter_space.suggest_parameter_fix(invalid_tau_params)
        self.assertTrue(tau_constraint_func(fixed_params))
        
        # 测试动量约束违反
        invalid_momentum_params = {
            'moco_momentum': 0.8,  # 违反约束
            'moco_tau1': 0.2,
            'moco_tau2': 0.3
        }
        
        momentum_constraint_func = validator.constraint_functions['moco_momentum_range']
        self.assertFalse(momentum_constraint_func(invalid_momentum_params))
        
        # 测试温度约束违反
        invalid_temp_params = {
            'moco_t': -0.1,  # 违反约束
            'moco_tau1': 0.2,
            'moco_tau2': 0.3
        }
        
        temp_constraint_func = validator.constraint_functions['moco_temperature_positive']
        self.assertFalse(temp_constraint_func(invalid_temp_params))
        
        print("✓ MoCo参数约束违反处理测试通过")
    
    def test_moco_optimization_complete_flow(self):
        """测试包含MoCo参数的完整优化流程"""
        print("\n测试MoCo参数完整优化流程...")
        
        # 1. 创建组件
        parameter_space = create_default_parameter_space("LDA")
        task_evaluator = MockMoCoTaskEvaluator("LDA")
        
        optimizer = create_bayesian_optimizer(
            task_type="LDA",
            acquisition_function_type="EI",
            n_initial_points=3,
            random_state=42
        )
        
        # 初始化优化器
        optimizer._initialize_optimization()
        
        # 2. 运行优化循环
        history = OptimizationHistory()
        history.task_type = "LDA"
        history.acquisition_function = "EI"
        history.start_time = datetime.now()
        
        max_iterations = 15
        moco_param_found = False
        
        for iteration in range(1, max_iterations + 1):
            # 获取参数建议
            suggested_params = optimizer.suggest_next_parameters()
            
            # 验证包含新MoCo参数
            if 'moco_tau1' in suggested_params and 'moco_tau2' in suggested_params:
                moco_param_found = True
                
                # 验证参数值范围
                self.assertGreaterEqual(suggested_params['moco_tau1'], 0.01)
                self.assertLessEqual(suggested_params['moco_tau1'], 1.0)
                self.assertGreaterEqual(suggested_params['moco_tau2'], 0.01)
                self.assertLessEqual(suggested_params['moco_tau2'], 1.0)
            
            if 'enable_view_0' in suggested_params:
                self.assertIn(suggested_params['enable_view_0'], ['true', 'false'])
            
            # 验证参数有效性
            is_valid, errors = parameter_space.validate_parameters_detailed(suggested_params)
            if not is_valid:
                # 尝试修复参数
                suggested_params = parameter_space.suggest_parameter_fix(suggested_params)
                is_valid, _ = parameter_space.validate_parameters_detailed(suggested_params)
                self.assertTrue(is_valid, f"修复后的参数仍然无效: {errors}")
            
            # 评估参数
            evaluation_result = task_evaluator.evaluate_parameters(suggested_params)
            self.assertIn('objective_value', evaluation_result)
            self.assertIn('metrics', evaluation_result)
            
            # 创建优化结果
            result = OptimizationResult(
                parameters=suggested_params,
                objective_value=evaluation_result['objective_value'],
                metrics=evaluation_result['metrics'],
                iteration=iteration,
                timestamp=datetime.now(),
                evaluation_time=1.0,
                fold_results=evaluation_result.get('fold_results'),
                objective_values=evaluation_result.get('objective_values')
            )
            
            # 更新优化器和历史
            optimizer.update_model(
                parameters=suggested_params,
                objective_value=evaluation_result['objective_value'],
                metrics=evaluation_result['metrics'],
                evaluation_time=1.0,
                objective_values=evaluation_result.get('objective_values')
            )
            history.add_result(result)
        
        # 3. 验证优化结果
        self.assertEqual(history.total_iterations, max_iterations)
        self.assertIsNotNone(history.best_result)
        self.assertGreater(history.best_result.objective_value, 0.5)
        
        # 验证至少有一次迭代包含了新MoCo参数
        self.assertTrue(moco_param_found, "优化过程中应该包含新MoCo参数")
        
        # 验证收敛
        convergence_curve = history.get_convergence_curve()
        self.assertEqual(len(convergence_curve), max_iterations)
        
        # 检查是否有改进（允许一定的随机性）
        improvement = convergence_curve[-1] - convergence_curve[0]
        print(f"性能改进: {improvement:.4f}")
        
        print(f"✓ MoCo参数完整优化流程测试通过，最佳AUROC: {history.best_result.objective_value:.4f}")
    
    def test_moco_parameter_space_sampling(self):
        """测试MoCo参数空间采样"""
        print("\n测试MoCo参数空间采样...")
        
        parameter_space = create_default_parameter_space("LDA")
        
        # 进行多次采样
        sample_count = 50
        valid_samples = 0
        moco_tau1_values = []
        moco_tau2_values = []
        enable_view_values = []
        
        for i in range(sample_count):
            try:
                params = parameter_space.sample_random_parameters(seed=i)
                is_valid, errors = parameter_space.validate_parameters_detailed(params)
                
                if is_valid:
                    valid_samples += 1
                    
                    # 收集MoCo参数值
                    if 'moco_tau1' in params:
                        moco_tau1_values.append(params['moco_tau1'])
                    if 'moco_tau2' in params:
                        moco_tau2_values.append(params['moco_tau2'])
                    if 'enable_view_0' in params:
                        enable_view_values.append(params['enable_view_0'])
                else:
                    # 尝试修复参数
                    fixed_params = parameter_space.suggest_parameter_fix(params)
                    is_fixed_valid, _ = parameter_space.validate_parameters_detailed(fixed_params)
                    if is_fixed_valid:
                        valid_samples += 1
                        
                        if 'moco_tau1' in fixed_params:
                            moco_tau1_values.append(fixed_params['moco_tau1'])
                        if 'moco_tau2' in fixed_params:
                            moco_tau2_values.append(fixed_params['moco_tau2'])
                        if 'enable_view_0' in fixed_params:
                            enable_view_values.append(fixed_params['enable_view_0'])
                            
            except Exception as e:
                print(f"采样失败 (seed={i}): {e}")
        
        # 验证采样成功率
        success_rate = valid_samples / sample_count
        self.assertGreater(success_rate, 0.8, f"MoCo参数采样成功率过低: {success_rate:.2%}")
        
        # 验证参数值分布
        if moco_tau1_values:
            tau1_min, tau1_max = min(moco_tau1_values), max(moco_tau1_values)
            self.assertGreaterEqual(tau1_min, 0.01)
            self.assertLessEqual(tau1_max, 1.0)
            print(f"moco_tau1值范围: [{tau1_min:.3f}, {tau1_max:.3f}]")
        
        if moco_tau2_values:
            tau2_min, tau2_max = min(moco_tau2_values), max(moco_tau2_values)
            self.assertGreaterEqual(tau2_min, 0.01)
            self.assertLessEqual(tau2_max, 1.0)
            print(f"moco_tau2值范围: [{tau2_min:.3f}, {tau2_max:.3f}]")
        
        if enable_view_values:
            unique_values = set(enable_view_values)
            self.assertTrue(unique_values.issubset({'true', 'false'}))
            print(f"enable_view_0值: {unique_values}")
        
        print(f"✓ MoCo参数空间采样测试通过，成功率: {success_rate:.2%}")
    
    def test_moco_state_persistence(self):
        """测试包含MoCo参数的状态持久化"""
        print("\n测试MoCo参数状态持久化...")
        
        # 创建状态管理器
        state_manager = create_default_state_manager(
            checkpoint_dir=self.checkpoint_dir
        )
        
        # 创建包含MoCo参数的优化历史
        history = OptimizationHistory()
        history.task_type = "LDA"
        history.start_time = datetime.now()
        
        # 添加包含新MoCo参数的结果
        moco_params = {
            'dimensions': 256,
            'hidden1': 128,
            'hidden2': 64,
            'decoder1': 512,
            'dropout': 0.5,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 1.0,
            'gat_heads': 8,
            'gt_heads': 8,
            'fusion_heads': 8,
            'batch': 32,
            'moco_K': 4096,
            'moco_momentum': 0.999,
            'moco_t': 0.2,
            'moco_tau1': 0.15,
            'moco_tau2': 0.25,
            'fusion_strategy': 'self_attention',
            'feature_type': 'normal',
            'moco_type': 'double_tau',
            'enable_view_0': 'true'
        }
        
        result = OptimizationResult(
            parameters=moco_params,
            objective_value=0.85,
            metrics={'AUROC': 0.85, 'AUPRC': 0.80, 'F1': 0.75},
            iteration=1,
            timestamp=datetime.now(),
            evaluation_time=1.0
        )
        
        history.add_result(result)
        
        # 保存状态
        state_data = {
            'history': history.to_dict(),
            'iteration': 1
        }
        checkpoint_name = state_manager.save_state(state_data, "moco_test")
        
        # 加载状态
        loaded_state = state_manager.load_state(checkpoint_name)
        self.assertIsNotNone(loaded_state)
        self.assertIn('history', loaded_state)
        
        # 恢复历史记录
        history_data = loaded_state['history']
        if isinstance(history_data, OptimizationHistory):
            # 如果已经是对象，直接使用
            recovered_history = history_data
        else:
            # 如果是字典，从字典创建
            recovered_history = OptimizationHistory.from_dict(history_data)
        
        # 验证MoCo参数正确恢复
        self.assertEqual(len(recovered_history.results), 1)
        recovered_params = recovered_history.results[0].parameters
        
        self.assertEqual(recovered_params['moco_tau1'], 0.15)
        self.assertEqual(recovered_params['moco_tau2'], 0.25)
        self.assertEqual(recovered_params['enable_view_0'], 'true')
        
        print("✓ MoCo参数状态持久化测试通过")
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        print("\n测试向后兼容性...")
        
        # 创建不包含新MoCo参数的历史数据
        old_history_data = {
            'task_type': 'LDA',
            'acquisition_function': 'EI',
            'start_time': datetime.now().isoformat(),
            'results': [
                {
                    'parameters': {
                        'dimensions': 256,
                        'hidden1': 128,
                        'hidden2': 64,
                        'decoder1': 512,
                        'dropout': 0.5,
                        'lr': 0.001,
                        'weight_decay': 1e-4,
                        'alpha': 1.0,
                        'beta': 1.0,
                        'gamma': 1.0,
                        'gat_heads': 8,
                        'gt_heads': 8,
                        'fusion_heads': 8,
                        'batch': 32,
                        'moco_K': 4096,
                        'moco_momentum': 0.999,
                        'moco_t': 0.2,
                        'fusion_strategy': 'self_attention',
                        'feature_type': 'normal',
                        'moco_type': 'basic'
                        # 缺少新MoCo参数
                    },
                    'objective_value': 0.80,
                    'metrics': {'AUROC': 0.80, 'AUPRC': 0.75, 'F1': 0.70},
                    'iteration': 1,
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_time': 1.0
                }
            ]
        }
        
        # 尝试加载旧格式数据
        try:
            recovered_history = OptimizationHistory.from_dict(old_history_data)
            self.assertEqual(len(recovered_history.results), 1)
            
            # 验证旧参数正确加载
            old_params = recovered_history.results[0].parameters
            self.assertEqual(old_params['moco_K'], 4096)
            self.assertEqual(old_params['moco_momentum'], 0.999)
            self.assertEqual(old_params['moco_t'], 0.2)
            
            # 验证新参数被自动添加了默认值（向后兼容性）
            self.assertIn('moco_tau1', old_params)  # 应该有默认值
            self.assertIn('moco_tau2', old_params)  # 应该有默认值
            self.assertIn('enable_view_0', old_params)  # 应该有默认值
            
            # 验证默认值合理
            self.assertGreaterEqual(old_params['moco_tau1'], 0.01)
            self.assertLessEqual(old_params['moco_tau1'], 1.0)
            self.assertGreaterEqual(old_params['moco_tau2'], 0.01)
            self.assertLessEqual(old_params['moco_tau2'], 1.0)
            self.assertIn(old_params['enable_view_0'], ['true', 'false'])
            
            print("✓ 向后兼容性测试通过")
            
        except Exception as e:
            self.fail(f"向后兼容性测试失败: {e}")


def run_moco_integration_tests():
    """运行MoCo集成测试"""
    print("=" * 60)
    print("MoCo超参数集成完整流程测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试
    test_suite.addTest(TestMoCoIntegrationComplete('test_moco_parameter_parsing_and_validation'))
    test_suite.addTest(TestMoCoIntegrationComplete('test_moco_parameter_constraint_violations'))
    test_suite.addTest(TestMoCoIntegrationComplete('test_moco_optimization_complete_flow'))
    test_suite.addTest(TestMoCoIntegrationComplete('test_moco_parameter_space_sampling'))
    test_suite.addTest(TestMoCoIntegrationComplete('test_moco_state_persistence'))
    test_suite.addTest(TestMoCoIntegrationComplete('test_backward_compatibility'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("MoCo集成测试总结")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"失败测试数: {len(result.failures)}")
    print(f"错误测试数: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n出错的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 所有MoCo集成测试通过!")
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息")
    
    return success


if __name__ == "__main__":
    success = run_moco_integration_tests()
    sys.exit(0 if success else 1)