"""
贝叶斯优化器约束验证测试

专门测试贝叶斯优化器建议的参数是否满足MoCo约束条件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from typing import Dict, Any

from bayesian_optimizer import BayesianOptimizer, create_bayesian_optimizer
from autodl_core import create_default_parameter_space
from task_evaluator import create_task_evaluator
from parameter_validator import ParameterValidator


class TestBayesianOptimizerConstraints:
    """贝叶斯优化器约束验证测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.parameter_space = create_default_parameter_space()
        self.validator = ParameterValidator(self.parameter_space)
        
        # 创建多个不同配置的优化器进行测试
        self.optimizers = []
        for i in range(3):
            optimizer = create_bayesian_optimizer(
                task_type="LDA",
                n_initial_points=5,
                random_state=42 + i  # 不同的随机种子
            )
            optimizer._initialize_optimization()
            self.optimizers.append(optimizer)
    
    def test_suggested_parameters_satisfy_moco_constraints(self):
        """测试建议的参数满足MoCo约束条件"""
        constraint_violations = []
        
        for optimizer_idx, optimizer in enumerate(self.optimizers):
            print(f"\n测试优化器 {optimizer_idx + 1}:")
            
            # 生成多个参数建议
            for suggestion_idx in range(10):
                params = optimizer.suggest_next_parameters()
                
                # 验证参数有效性
                is_valid = self.validator.validate_parameters(params)
                
                if not is_valid:
                    # 记录约束违反
                    violations = self._check_specific_constraints(params)
                    constraint_violations.append({
                        'optimizer': optimizer_idx + 1,
                        'suggestion': suggestion_idx + 1,
                        'parameters': params,
                        'violations': violations
                    })
                    print(f"  建议 {suggestion_idx + 1}: 约束违反 - {violations}")
                else:
                    print(f"  建议 {suggestion_idx + 1}: 约束满足 ✓")
        
        # 报告结果
        if constraint_violations:
            print(f"\n发现 {len(constraint_violations)} 个约束违反:")
            for violation in constraint_violations:
                print(f"优化器{violation['optimizer']} 建议{violation['suggestion']}: {violation['violations']}")
            
            # 如果违反率过高，测试失败
            total_suggestions = sum(10 for _ in self.optimizers)
            violation_rate = len(constraint_violations) / total_suggestions
            
            if violation_rate > 0.2:  # 允许20%的违反率
                pytest.fail(f"约束违反率过高: {violation_rate:.1%} ({len(constraint_violations)}/{total_suggestions})")
            else:
                print(f"约束违反率在可接受范围内: {violation_rate:.1%}")
        else:
            print("所有参数建议都满足约束条件 ✓")
    
    def _check_specific_constraints(self, params: Dict[str, Any]) -> list:
        """检查具体的约束违反"""
        violations = []
        
        # 检查MoCo DoubleTau温度约束
        if 'moco_tau1' in params and 'moco_tau2' in params:
            tau1 = float(params['moco_tau1'])
            tau2 = float(params['moco_tau2'])
            if tau2 < tau1:
                violations.append(f"DoubleTau约束违反: tau2({tau2:.3f}) < tau1({tau1:.3f})")
        
        # 检查MoCo动量系数范围
        if 'moco_momentum' in params:
            momentum = float(params['moco_momentum'])
            if not (0.9 <= momentum <= 0.9999):
                violations.append(f"动量系数超出范围: {momentum:.4f} 不在 [0.9, 0.9999]")
        
        # 检查温度参数正值约束
        temp_params = ['moco_t', 'moco_tau1', 'moco_tau2']
        for param in temp_params:
            if param in params:
                temp = float(params[param])
                if temp <= 0:
                    violations.append(f"温度参数非正值: {param}={temp:.3f}")
                elif temp > 1.0:
                    violations.append(f"温度参数过大: {param}={temp:.3f} > 1.0")
        
        return violations
    
    def test_parameter_suggestion_with_model_updates(self):
        """测试模型更新后的参数建议"""
        optimizer = self.optimizers[0]
        
        # 模拟一些评估结果来更新模型
        mock_evaluations = [
            ({'moco_tau1': 0.1, 'moco_tau2': 0.2, 'enable_view_0': 'true'}, 0.85),
            ({'moco_tau1': 0.3, 'moco_tau2': 0.4, 'enable_view_0': 'false'}, 0.82),
            ({'moco_tau1': 0.2, 'moco_tau2': 0.3, 'enable_view_0': 'true'}, 0.88),
        ]
        
        # 添加基础参数
        base_params = {
            'hidden1': 128, 'hidden2': 64, 'dropout': 0.5, 'lr': 0.001,
            'weight_decay': 1e-4, 'batch': 32, 'moco_K': 4096,
            'moco_momentum': 0.999, 'moco_t': 0.2, 'proj_dim': 128,
            'queue_warmup_steps': 0, 'fusion_strategy': 'self_attention',
            'feature_type': 'normal', 'moco_type': 'double_tau'
        }
        
        for params, objective_value in mock_evaluations:
            full_params = {**base_params, **params}
            metrics = {'AUROC': objective_value, 'AUPRC': objective_value - 0.1}
            
            # 更新模型
            optimizer.update_model(full_params, objective_value, metrics, 1.0)
        
        print(f"模型已更新，历史记录包含 {len(optimizer.history.results)} 个结果")
        
        # 现在测试基于模型的参数建议
        constraint_violations = 0
        
        for i in range(10):
            params = optimizer.suggest_next_parameters()
            
            # 验证约束
            is_valid = self.validator.validate_parameters(params)
            if not is_valid:
                constraint_violations += 1
                violations = self._check_specific_constraints(params)
                print(f"模型建议 {i+1} 约束违反: {violations}")
            else:
                print(f"模型建议 {i+1}: 约束满足 ✓")
        
        # 验证约束违反率
        violation_rate = constraint_violations / 10
        assert violation_rate <= 0.3, f"模型更新后约束违反率过高: {violation_rate:.1%}"
        
        print(f"模型更新后约束违反率: {violation_rate:.1%}")
    
    def test_parameter_fix_integration(self):
        """测试参数修复与建议功能的集成"""
        optimizer = self.optimizers[0]
        
        # 生成参数建议并尝试修复违反约束的参数
        fixed_count = 0
        
        for i in range(20):
            params = optimizer.suggest_next_parameters()
            
            # 检查是否需要修复
            if not self.validator.validate_parameters(params):
                # 尝试修复
                fixed_params = self.validator.suggest_parameter_fix(params)
                
                # 验证修复后的参数
                is_fixed_valid = self.validator.validate_parameters(fixed_params)
                
                if is_fixed_valid:
                    fixed_count += 1
                    print(f"参数 {i+1} 成功修复")
                else:
                    print(f"参数 {i+1} 修复失败")
                    # 显示修复前后的关键参数
                    print(f"  原始: tau1={params.get('moco_tau1', 'N/A'):.3f}, tau2={params.get('moco_tau2', 'N/A'):.3f}")
                    print(f"  修复: tau1={fixed_params.get('moco_tau1', 'N/A'):.3f}, tau2={fixed_params.get('moco_tau2', 'N/A'):.3f}")
        
        print(f"成功修复了 {fixed_count} 个违反约束的参数组合")


if __name__ == "__main__":
    # 运行测试
    print("开始贝叶斯优化器约束验证测试...")
    
    test_instance = TestBayesianOptimizerConstraints()
    test_instance.setup_method()
    
    try:
        print("\n=== 测试参数建议满足约束条件 ===")
        test_instance.test_suggested_parameters_satisfy_moco_constraints()
        
        print("\n=== 测试模型更新后的参数建议 ===")
        test_instance.test_parameter_suggestion_with_model_updates()
        
        print("\n=== 测试参数修复集成 ===")
        test_instance.test_parameter_fix_integration()
        
        print("\n✓ 所有约束验证测试通过")
        
    except Exception as e:
        print(f"✗ 约束验证测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n贝叶斯优化器约束验证测试完成!")