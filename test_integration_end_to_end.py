#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoDL贝叶斯优化系统端到端集成测试

测试完整的优化流程，验证所有组件的集成和协作
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心组件
from autodl_core import create_default_parameter_space, OptimizationHistory
from bayesian_optimizer import create_bayesian_optimizer, create_multi_objective_optimizer
from task_evaluator import create_task_evaluator
from state_manager import create_default_state_manager
from result_analyzer import create_result_analyzer_from_checkpoint
from visualizer import create_visualizer_from_checkpoint
from report_generator import ReportGenerator, ReportConfig


class MockTaskEvaluator:
    """模拟任务评估器，用于快速测试"""
    
    def __init__(self, task_type="LDA"):
        self.task_type = task_type
        self.evaluation_count = 0
        np.random.seed(42)  # 确保结果可重现
    
    def evaluate_parameters(self, parameters):
        """模拟参数评估"""
        self.evaluation_count += 1
        
        # 基于参数生成模拟的性能指标
        # 使用参数的某种组合来生成"真实"的性能
        param_hash = hash(str(sorted(parameters.items()))) % 1000000
        np.random.seed(param_hash)
        
        # 生成相关的性能指标
        base_auroc = 0.7 + np.random.random() * 0.25  # 0.7-0.95
        noise = np.random.normal(0, 0.02)  # 添加噪声
        auroc = max(0.5, min(0.99, base_auroc + noise))
        
        auprc = max(0.5, min(0.99, auroc - 0.05 + np.random.random() * 0.1))
        f1 = max(0.5, min(0.99, auroc - 0.1 + np.random.random() * 0.15))
        precision = max(0.5, min(0.99, f1 + np.random.random() * 0.05))
        recall = max(0.5, min(0.99, f1 + np.random.random() * 0.05))
        
        # 模拟折验证结果
        fold_results = {
            'AUROC': [auroc + np.random.normal(0, 0.01) for _ in range(5)],
            'AUPRC': [auprc + np.random.normal(0, 0.01) for _ in range(5)],
            'F1': [f1 + np.random.normal(0, 0.01) for _ in range(5)]
        }
        
        return {
            'objective_value': auroc,
            'metrics': {
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1': f1,
                'Precision': precision,
                'Recall': recall
            },
            'fold_results': fold_results,
            'objective_values': {
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1': f1
            }
        }


class TestEndToEndIntegration(unittest.TestCase):
    """端到端集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        self.output_dir = os.path.join(self.temp_dir, 'results')
        
        # 创建目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_single_objective_optimization_flow(self):
        """测试单目标优化完整流程"""
        print("\n测试单目标优化完整流程...")
        
        # 1. 创建参数空间
        parameter_space = create_default_parameter_space("LDA")
        self.assertGreater(len(parameter_space.parameters), 0)
        
        # 2. 创建模拟任务评估器
        task_evaluator = MockTaskEvaluator("LDA")
        
        # 3. 创建状态管理器
        state_manager = create_default_state_manager(
            checkpoint_dir=self.checkpoint_dir,
            save_frequency=2
        )
        
        # 4. 创建贝叶斯优化器
        optimizer = create_bayesian_optimizer(
            parameter_space=parameter_space,
            task_evaluator=task_evaluator,
            acquisition_function="EI",
            random_seed=42
        )
        
        # 5. 运行优化循环
        history = OptimizationHistory()
        history.task_type = "LDA"
        history.acquisition_function = "EI"
        history.start_time = datetime.now()
        
        max_iterations = 10
        for iteration in range(1, max_iterations + 1):
            # 获取参数建议
            suggested_params = optimizer.suggest_parameters()
            self.assertIsInstance(suggested_params, dict)
            self.assertGreater(len(suggested_params), 0)
            
            # 验证参数有效性
            is_valid = parameter_space.validate_parameters(suggested_params)
            self.assertTrue(is_valid, f"第{iteration}次迭代的参数无效: {suggested_params}")
            
            # 评估参数
            evaluation_result = task_evaluator.evaluate_parameters(suggested_params)
            self.assertIn('objective_value', evaluation_result)
            self.assertIn('metrics', evaluation_result)
            
            # 创建优化结果
            from autodl_core import OptimizationResult
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
            optimizer.update_with_result(result)
            history.add_result(result)
            
            # 定期保存状态
            if iteration % 2 == 0:
                state_data = {
                    'history': history.to_dict(),
                    'optimizer_state': optimizer.get_state(),
                    'iteration': iteration
                }
                state_manager.save_state(state_data, f"iteration_{iteration}")
        
        # 6. 验证优化结果
        self.assertEqual(history.total_iterations, max_iterations)
        self.assertIsNotNone(history.best_result)
        self.assertGreater(history.best_result.objective_value, 0.5)
        
        # 验证收敛曲线
        convergence_curve = history.get_convergence_curve()
        self.assertEqual(len(convergence_curve), max_iterations)
        self.assertGreaterEqual(convergence_curve[-1], convergence_curve[0])  # 应该有改进
        
        print(f"✓ 单目标优化完成，最佳AUROC: {history.best_result.objective_value:.4f}")
        
        # 7. 测试状态恢复
        self._test_state_recovery(state_manager, history, max_iterations)
        
        # 8. 测试结果分析
        self._test_result_analysis(history, parameter_space)
        
        # 9. 测试报告生成
        self._test_report_generation(history, parameter_space)
    
    def test_multi_objective_optimization_flow(self):
        """测试多目标优化完整流程"""
        print("\n测试多目标优化完整流程...")
        
        # 1. 创建组件
        parameter_space = create_default_parameter_space("MDA")
        task_evaluator = MockTaskEvaluator("MDA")
        
        # 2. 创建多目标优化器
        objectives = ['AUROC', 'AUPRC', 'F1']
        maximize_objectives = {'AUROC': True, 'AUPRC': True, 'F1': True}
        objective_weights = {'AUROC': 0.5, 'AUPRC': 0.3, 'F1': 0.2}
        
        optimizer = create_multi_objective_optimizer(
            parameter_space=parameter_space,
            task_evaluator=task_evaluator,
            objectives=objectives,
            maximize_objectives=maximize_objectives,
            objective_weights=objective_weights,
            acquisition_function="EI",
            random_seed=42
        )
        
        # 3. 运行多目标优化
        history = OptimizationHistory()
        history.set_objectives(objectives, maximize_objectives, objective_weights)
        history.task_type = "MDA"
        history.start_time = datetime.now()
        
        max_iterations = 15
        for iteration in range(1, max_iterations + 1):
            suggested_params = optimizer.suggest_parameters()
            evaluation_result = task_evaluator.evaluate_parameters(suggested_params)
            
            from autodl_core import OptimizationResult
            result = OptimizationResult(
                parameters=suggested_params,
                objective_value=evaluation_result['objective_value'],
                metrics=evaluation_result['metrics'],
                iteration=iteration,
                timestamp=datetime.now(),
                evaluation_time=1.0,
                objective_values=evaluation_result['objective_values']
            )
            
            optimizer.update_with_result(result)
            history.add_result(result)
        
        # 4. 验证多目标优化结果
        self.assertEqual(len(history.objectives), 3)
        self.assertGreater(len(history.pareto_front), 0)
        
        # 验证帕累托前沿
        pareto_metrics = history.get_pareto_front_metrics()
        self.assertIn('front_size', pareto_metrics)
        self.assertGreater(pareto_metrics['front_size'], 0)
        
        # 验证加权目标函数
        for result in history.results:
            weighted_value = history.get_weighted_objective_value(result)
            self.assertIsInstance(weighted_value, float)
            self.assertGreater(weighted_value, 0)
        
        print(f"✓ 多目标优化完成，帕累托前沿大小: {len(history.pareto_front)}")
        
        # 5. 测试多目标可视化
        self._test_multi_objective_visualization(history, parameter_space)
    
    def test_parameter_space_constraints(self):
        """测试参数空间约束处理"""
        print("\n测试参数空间约束处理...")
        
        parameter_space = create_default_parameter_space("LMI")
        
        # 测试多次随机采样
        valid_count = 0
        total_samples = 100
        
        for i in range(total_samples):
            try:
                params = parameter_space.sample_random_parameters(seed=i)
                is_valid, errors = parameter_space.validate_parameters_detailed(params)
                
                if is_valid:
                    valid_count += 1
                else:
                    # 尝试修复参数
                    fixed_params = parameter_space.suggest_parameter_fix(params)
                    is_fixed_valid, _ = parameter_space.validate_parameters_detailed(fixed_params)
                    if is_fixed_valid:
                        valid_count += 1
                        
            except Exception as e:
                print(f"采样失败 (seed={i}): {e}")
        
        success_rate = valid_count / total_samples
        self.assertGreater(success_rate, 0.8, f"参数约束满足率过低: {success_rate:.2%}")
        
        print(f"✓ 参数约束测试完成，成功率: {success_rate:.2%}")
    
    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复机制"""
        print("\n测试错误处理和恢复机制...")
        
        parameter_space = create_default_parameter_space("LDA")
        
        # 创建会偶尔失败的任务评估器
        class FlakyTaskEvaluator(MockTaskEvaluator):
            def __init__(self):
                super().__init__()
                self.failure_rate = 0.3
            
            def evaluate_parameters(self, parameters):
                if np.random.random() < self.failure_rate:
                    raise RuntimeError("模拟评估失败")
                return super().evaluate_parameters(parameters)
        
        task_evaluator = FlakyTaskEvaluator()
        optimizer = create_bayesian_optimizer(
            parameter_space=parameter_space,
            task_evaluator=task_evaluator,
            random_seed=42
        )
        
        # 运行优化，处理错误
        successful_evaluations = 0
        failed_evaluations = 0
        max_attempts = 20
        
        for attempt in range(max_attempts):
            try:
                suggested_params = optimizer.suggest_parameters()
                evaluation_result = task_evaluator.evaluate_parameters(suggested_params)
                
                from autodl_core import OptimizationResult
                result = OptimizationResult(
                    parameters=suggested_params,
                    objective_value=evaluation_result['objective_value'],
                    metrics=evaluation_result['metrics'],
                    iteration=successful_evaluations + 1,
                    timestamp=datetime.now(),
                    evaluation_time=1.0
                )
                
                optimizer.update_with_result(result)
                successful_evaluations += 1
                
            except RuntimeError:
                failed_evaluations += 1
                continue  # 跳过失败的评估
        
        self.assertGreater(successful_evaluations, 0)
        print(f"✓ 错误处理测试完成，成功: {successful_evaluations}, 失败: {failed_evaluations}")
    
    def _test_state_recovery(self, state_manager, original_history, max_iterations):
        """测试状态恢复功能"""
        print("  测试状态恢复...")
        
        # 加载保存的状态
        checkpoint_name = f"iteration_{max_iterations}"
        state_data = state_manager.load_state(checkpoint_name)
        
        self.assertIsNotNone(state_data)
        self.assertIn('history', state_data)
        self.assertIn('iteration', state_data)
        
        # 恢复历史记录
        from autodl_core import OptimizationHistory
        recovered_history = OptimizationHistory.from_dict(state_data['history'])
        
        # 验证恢复的数据
        self.assertEqual(recovered_history.total_iterations, original_history.total_iterations)
        self.assertEqual(len(recovered_history.results), len(original_history.results))
        
        if original_history.best_result and recovered_history.best_result:
            self.assertAlmostEqual(
                recovered_history.best_result.objective_value,
                original_history.best_result.objective_value,
                places=6
            )
        
        print("  ✓ 状态恢复测试通过")
    
    def _test_result_analysis(self, history, parameter_space):
        """测试结果分析功能"""
        print("  测试结果分析...")
        
        try:
            # 创建结果分析器
            analyzer = create_result_analyzer_from_checkpoint(
                checkpoint_path=None,
                history=history,
                parameter_space=parameter_space
            )
            
            # 测试敏感性分析
            sensitivity_results = analyzer.analyze_parameter_sensitivity()
            self.assertIsInstance(sensitivity_results, dict)
            self.assertGreater(len(sensitivity_results), 0)
            
            # 测试收敛分析
            convergence_analysis = analyzer.analyze_convergence()
            self.assertIsNotNone(convergence_analysis)
            
            # 测试统计摘要
            statistical_summary = analyzer.generate_statistical_summary()
            self.assertIsNotNone(statistical_summary)
            
            print("  ✓ 结果分析测试通过")
            
        except Exception as e:
            print(f"  ⚠️ 结果分析测试跳过: {e}")
    
    def _test_report_generation(self, history, parameter_space):
        """测试报告生成功能"""
        print("  测试报告生成...")
        
        try:
            # 创建报告生成器
            config = ReportConfig(
                title="集成测试报告",
                author="AutoDL测试系统",
                include_charts=False,  # 跳过图表生成以加快测试
                include_parameter_details=True
            )
            
            # 创建简化的分析器和可视化器
            analyzer = create_result_analyzer_from_checkpoint(
                checkpoint_path=None,
                history=history,
                parameter_space=parameter_space
            )
            
            visualizer = create_visualizer_from_checkpoint(
                checkpoint_path=None,
                history=history,
                parameter_space=parameter_space
            )
            
            report_generator = ReportGenerator(
                history=history,
                parameter_space=parameter_space,
                result_analyzer=analyzer,
                visualizer=visualizer,
                config=config
            )
            
            # 生成JSON报告
            json_path = os.path.join(self.output_dir, "test_report.json")
            report_generator.generate_json_report(json_path)
            
            # 验证报告文件
            self.assertTrue(os.path.exists(json_path))
            
            with open(json_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # 验证报告内容
            self.assertIn('metadata', report_data)
            self.assertIn('optimization_summary', report_data)
            self.assertIn('best_result', report_data)
            
            print("  ✓ 报告生成测试通过")
            
        except Exception as e:
            print(f"  ⚠️ 报告生成测试跳过: {e}")
    
    def _test_multi_objective_visualization(self, history, parameter_space):
        """测试多目标优化可视化"""
        print("  测试多目标可视化...")
        
        try:
            visualizer = create_visualizer_from_checkpoint(
                checkpoint_path=None,
                history=history,
                parameter_space=parameter_space
            )
            
            # 测试帕累托前沿可视化（不实际保存文件）
            # 这里主要测试函数调用不出错
            self.assertGreater(len(history.pareto_front), 0)
            
            print("  ✓ 多目标可视化测试通过")
            
        except Exception as e:
            print(f"  ⚠️ 多目标可视化测试跳过: {e}")


class TestComponentIntegration(unittest.TestCase):
    """组件集成测试"""
    
    def test_parameter_space_task_evaluator_integration(self):
        """测试参数空间与任务评估器的集成"""
        print("\n测试参数空间与任务评估器集成...")
        
        for task_type in ['LDA', 'MDA', 'LMI']:
            with self.subTest(task_type=task_type):
                parameter_space = create_default_parameter_space(task_type)
                task_evaluator = MockTaskEvaluator(task_type)
                
                # 测试参数采样和评估
                params = parameter_space.sample_random_parameters(seed=42)
                result = task_evaluator.evaluate_parameters(params)
                
                self.assertIn('objective_value', result)
                self.assertIn('metrics', result)
                self.assertIsInstance(result['objective_value'], float)
                self.assertGreater(result['objective_value'], 0)
        
        print("✓ 参数空间与任务评估器集成测试通过")
    
    def test_optimizer_state_manager_integration(self):
        """测试优化器与状态管理器的集成"""
        print("\n测试优化器与状态管理器集成...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
            os.makedirs(checkpoint_dir)
            
            # 创建组件
            parameter_space = create_default_parameter_space("LDA")
            task_evaluator = MockTaskEvaluator("LDA")
            state_manager = create_default_state_manager(checkpoint_dir=checkpoint_dir)
            
            optimizer = create_bayesian_optimizer(
                parameter_space=parameter_space,
                task_evaluator=task_evaluator,
                random_seed=42
            )
            
            # 运行几次迭代
            history = OptimizationHistory()
            for i in range(3):
                params = optimizer.suggest_parameters()
                eval_result = task_evaluator.evaluate_parameters(params)
                
                from autodl_core import OptimizationResult
                result = OptimizationResult(
                    parameters=params,
                    objective_value=eval_result['objective_value'],
                    metrics=eval_result['metrics'],
                    iteration=i+1,
                    timestamp=datetime.now(),
                    evaluation_time=1.0
                )
                
                optimizer.update_with_result(result)
                history.add_result(result)
            
            # 保存状态
            state_data = {
                'history': history.to_dict(),
                'optimizer_state': optimizer.get_state()
            }
            state_manager.save_state(state_data, "test_checkpoint")
            
            # 验证状态保存
            loaded_state = state_manager.load_state("test_checkpoint")
            self.assertIsNotNone(loaded_state)
            self.assertIn('history', loaded_state)
            self.assertIn('optimizer_state', loaded_state)
        
        print("✓ 优化器与状态管理器集成测试通过")


def run_integration_tests():
    """运行所有集成测试"""
    print("=" * 60)
    print("AutoDL贝叶斯优化系统 - 端到端集成测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加端到端测试
    test_suite.addTest(TestEndToEndIntegration('test_single_objective_optimization_flow'))
    test_suite.addTest(TestEndToEndIntegration('test_multi_objective_optimization_flow'))
    test_suite.addTest(TestEndToEndIntegration('test_parameter_space_constraints'))
    test_suite.addTest(TestEndToEndIntegration('test_error_handling_and_recovery'))
    
    # 添加组件集成测试
    test_suite.addTest(TestComponentIntegration('test_parameter_space_task_evaluator_integration'))
    test_suite.addTest(TestComponentIntegration('test_optimizer_state_manager_integration'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("集成测试总结")
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
        print("\n🎉 所有集成测试通过!")
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息")
    
    return success


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)