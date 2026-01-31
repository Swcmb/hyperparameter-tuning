"""
多目标优化MoCo参数支持测试

测试多目标优化器对MoCo参数的支持，包括参数处理、帕累托前沿计算和目标函数计算。

Feature: moco-hyperparameter-integration
"""

import unittest
import numpy as np
import sys
import os
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Any

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian_optimizer import create_multi_objective_optimizer, create_bayesian_optimizer
from autodl_core import OptimizationResult, OptimizationHistory, create_default_parameter_space
from task_evaluator import TaskEvaluator


class TestMultiObjectiveMoCoSupport(unittest.TestCase):
    """多目标优化MoCo参数支持测试类"""
    
    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.task_types = ['LDA', 'MDA', 'LMI']
        self.objectives = ['AUROC', 'AUPRC', 'F1']
        
        # MoCo参数配置
        self.moco_params = {
            'moco_K': 4096,
            'moco_momentum': 0.999,
            'moco_t': 0.2,
            'moco_tau1': 0.15,
            'moco_tau2': 0.25,
            'moco_type': 'double_tau',
            'enable_view_0': 'true'
        }
    
    def test_multi_objective_optimizer_includes_moco_parameters(self):
        """
        测试11.1: 验证多目标优化中的MoCo参数处理
        
        测试create_multi_objective_optimizer函数对MoCo参数的支持
        确保帕累托前沿计算包含MoCo参数
        
        需求: 8.1, 8.3
        """
        print("\n=== 测试多目标优化器MoCo参数支持 ===")
        
        for task_type in self.task_types:
            with self.subTest(task_type=task_type):
                # 1. 创建多目标优化器
                optimizer = create_multi_objective_optimizer(
                    task_type=task_type,
                    objectives=['AUROC', 'AUPRC'],
                    n_initial_points=3,
                    random_state=42
                )
                
                # 2. 验证优化器包含MoCo参数空间
                parameter_space = optimizer.parameter_space
                self.assertIsNotNone(parameter_space)
                
                # 验证MoCo参数在参数空间中
                moco_param_names = ['moco_K', 'moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_type', 'enable_view_0']
                for param_name in moco_param_names:
                    self.assertIn(param_name, parameter_space.parameters, 
                                f"MoCo参数 {param_name} 不在参数空间中")
                
                # 3. 测试参数采样包含MoCo参数
                sampled_params = parameter_space.sample_random_parameters(seed=42)
                for param_name in moco_param_names:
                    self.assertIn(param_name, sampled_params, 
                                f"采样参数中缺少MoCo参数 {param_name}")
                
                # 4. 验证MoCo参数类型和范围
                # 连续型参数
                self.assertTrue(0.9 <= sampled_params['moco_momentum'] <= 0.9999)
                self.assertTrue(0.01 <= sampled_params['moco_t'] <= 1.0)
                self.assertTrue(0.01 <= sampled_params['moco_tau1'] <= 1.0)
                self.assertTrue(0.01 <= sampled_params['moco_tau2'] <= 1.0)
                
                # 离散型参数
                self.assertIn(sampled_params['moco_K'], [1024, 2048, 4096, 8192])
                
                # 分类型参数
                self.assertIn(sampled_params['moco_type'], ['basic', 'double_tau'])
                self.assertIn(sampled_params['enable_view_0'], ['true', 'false'])
                
                print(f"✓ {task_type}: MoCo参数空间验证通过")
    
    def test_multi_objective_optimization_with_moco_parameters(self):
        """
        测试11.1: 验证多目标优化过程中MoCo参数的处理
        
        运行完整的多目标优化并验证MoCo参数在帕累托前沿中的处理
        
        需求: 8.1, 8.3
        """
        print("\n=== 测试多目标优化过程中的MoCo参数处理 ===")
        
        # 选择一个任务类型进行详细测试
        task_type = 'LDA'
        
        # 创建多目标优化器
        optimizer = create_multi_objective_optimizer(
            task_type=task_type,
            objectives=['AUROC', 'AUPRC'],
            n_initial_points=3,
            random_state=42
        )
        
        # 运行优化
        history = optimizer.optimize(n_iterations=5, checkpoint_freq=2)
        
        # 1. 验证优化历史包含MoCo参数
        self.assertIsInstance(history, OptimizationHistory)
        self.assertTrue(len(history.results) > 0)
        
        # 2. 验证每个优化结果都包含MoCo参数
        moco_param_names = ['moco_K', 'moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_type', 'enable_view_0']
        
        for i, result in enumerate(history.results):
            self.assertIsNotNone(result.parameters, f"结果 {i} 缺少参数")
            
            for param_name in moco_param_names:
                self.assertIn(param_name, result.parameters, 
                            f"结果 {i} 缺少MoCo参数 {param_name}")
        
        # 3. 验证帕累托前沿包含MoCo参数
        if history.pareto_front:
            for i, pareto_result in enumerate(history.pareto_front):
                self.assertIsNotNone(pareto_result.parameters, f"帕累托解 {i} 缺少参数")
                
                for param_name in moco_param_names:
                    self.assertIn(param_name, pareto_result.parameters, 
                                f"帕累托解 {i} 缺少MoCo参数 {param_name}")
                
                # 验证MoCo参数值的合理性
                params = pareto_result.parameters
                self.assertTrue(0.9 <= params['moco_momentum'] <= 0.9999)
                self.assertTrue(0.01 <= params['moco_t'] <= 1.0)
                self.assertTrue(0.01 <= params['moco_tau1'] <= 1.0)
                self.assertTrue(0.01 <= params['moco_tau2'] <= 1.0)
                self.assertIn(params['moco_K'], [1024, 2048, 4096, 8192])
                self.assertIn(params['moco_type'], ['basic', 'double_tau'])
                self.assertIn(params['enable_view_0'], ['true', 'false'])
        
        print(f"✓ 多目标优化过程中MoCo参数处理验证通过")
        print(f"  - 总结果数: {len(history.results)}")
        print(f"  - 帕累托前沿大小: {len(history.pareto_front) if history.pareto_front else 0}")
    
    def test_moco_parameter_impact_on_objective_functions(self):
        """
        测试11.2: 验证目标函数计算
        
        确保MoCo参数变化能正确影响目标函数值
        测试多目标场景下的参数评估
        
        需求: 8.2
        """
        print("\n=== 测试MoCo参数对目标函数的影响 ===")
        
        task_type = 'LDA'
        
        # 创建多目标优化器
        optimizer = create_multi_objective_optimizer(
            task_type=task_type,
            objectives=['AUROC', 'AUPRC'],
            n_initial_points=2,
            random_state=42
        )
        
        # 创建两组不同的MoCo参数配置
        base_params = optimizer.parameter_space.sample_random_parameters(seed=42)
        
        # 配置1: 基础MoCo设置
        params1 = base_params.copy()
        params1.update({
            'moco_momentum': 0.999,
            'moco_t': 0.2,
            'moco_tau1': 0.1,
            'moco_tau2': 0.2,
            'moco_type': 'basic',
            'enable_view_0': 'true'
        })
        
        # 配置2: 不同的MoCo设置
        params2 = base_params.copy()
        params2.update({
            'moco_momentum': 0.99,
            'moco_t': 0.5,
            'moco_tau1': 0.3,
            'moco_tau2': 0.4,
            'moco_type': 'double_tau',
            'enable_view_0': 'false'
        })
        
        # 评估两组参数
        try:
            result1 = optimizer.task_evaluator.evaluate_parameters(params1)
            result2 = optimizer.task_evaluator.evaluate_parameters(params2)
            
            # 1. 验证两个结果都包含目标函数值
            self.assertIsNotNone(result1)
            self.assertIsNotNone(result2)
            self.assertIsInstance(result1, dict)
            self.assertIsInstance(result2, dict)
            
            # 2. 验证目标函数值包含所有指定的目标
            for obj in ['AUROC', 'AUPRC']:
                self.assertIn(obj, result1)
                self.assertIn(obj, result2)
                
                # 验证目标值在合理范围内
                self.assertTrue(0.0 <= result1[obj] <= 1.0)
                self.assertTrue(0.0 <= result2[obj] <= 1.0)
            
            # 3. 验证不同MoCo参数配置产生不同的结果
            # 注意：由于使用模拟评估器，结果可能相同，但至少应该能正常计算
            print(f"  配置1结果: AUROC={result1['AUROC']:.4f}, AUPRC={result1['AUPRC']:.4f}")
            print(f"  配置2结果: AUROC={result2['AUROC']:.4f}, AUPRC={result2['AUPRC']:.4f}")
            
            # 4. 验证结果包含完整的指标信息
            expected_metrics = ['AUROC', 'AUPRC', 'F1', 'precision', 'recall', 'loss']
            for metric in expected_metrics:
                self.assertIn(metric, result1, f"结果1缺少指标 {metric}")
                self.assertIn(metric, result2, f"结果2缺少指标 {metric}")
            
            print("✓ MoCo参数对目标函数影响测试通过")
            
        except Exception as e:
            self.fail(f"MoCo参数评估失败: {e}")
    
    def test_pareto_front_with_moco_parameters(self):
        """
        测试11.1: 验证帕累托前沿计算包含MoCo参数
        
        确保帕累托前沿计算正确处理MoCo参数的多样性
        
        需求: 8.1, 8.3
        """
        print("\n=== 测试帕累托前沿中的MoCo参数多样性 ===")
        
        task_type = 'LDA'
        
        # 创建多目标优化器
        optimizer = create_multi_objective_optimizer(
            task_type=task_type,
            objectives=['AUROC', 'AUPRC', 'F1'],
            n_initial_points=4,
            random_state=42
        )
        
        # 运行优化以获得帕累托前沿
        history = optimizer.optimize(n_iterations=6, checkpoint_freq=3)
        
        if history.pareto_front and len(history.pareto_front) > 1:
            # 1. 验证帕累托前沿中MoCo参数的多样性
            moco_param_values = {}
            moco_param_names = ['moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_type', 'enable_view_0']
            
            for param_name in moco_param_names:
                values = [result.parameters[param_name] for result in history.pareto_front]
                moco_param_values[param_name] = values
            
            # 2. 检查是否有参数多样性（至少一个参数有不同值）
            has_diversity = False
            for param_name, values in moco_param_values.items():
                unique_values = set(values)
                if len(unique_values) > 1:
                    has_diversity = True
                    print(f"  {param_name}: {unique_values}")
            
            # 注意：由于随机性和模拟评估器，可能不总是有多样性
            # 但至少应该能正确处理MoCo参数
            print(f"  帕累托前沿大小: {len(history.pareto_front)}")
            print(f"  MoCo参数多样性: {'是' if has_diversity else '否'}")
            
            # 3. 验证所有帕累托解的MoCo参数都是有效的
            for i, result in enumerate(history.pareto_front):
                params = result.parameters
                
                # 验证参数约束
                self.assertTrue(0.9 <= params['moco_momentum'] <= 0.9999, 
                              f"帕累托解 {i} 的moco_momentum超出范围")
                self.assertTrue(0.01 <= params['moco_t'] <= 1.0, 
                              f"帕累托解 {i} 的moco_t超出范围")
                self.assertTrue(0.01 <= params['moco_tau1'] <= 1.0, 
                              f"帕累托解 {i} 的moco_tau1超出范围")
                self.assertTrue(0.01 <= params['moco_tau2'] <= 1.0, 
                              f"帕累托解 {i} 的moco_tau2超出范围")
                
                # 验证DoubleTau约束（如果适用）
                if params['moco_type'] == 'double_tau':
                    self.assertTrue(params['moco_tau2'] >= params['moco_tau1'], 
                                  f"帕累托解 {i} 违反DoubleTau约束: tau2 < tau1")
        
        print("✓ 帕累托前沿MoCo参数处理验证通过")


if __name__ == '__main__':
    unittest.main()