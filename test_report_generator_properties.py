"""
报告生成器属性测试

本模块实现了报告生成器的属性测试，验证以下属性：
- 属性 10: 优化报告生成
- 属性 19: 报告内容完整性
"""

import pytest
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

# 导入核心组件
from autodl_core import OptimizationHistory, OptimizationResult, ParameterSpace, create_default_parameter_space
from report_generator import ReportGenerator, ReportConfig


class TestReportGenerationProperties:
    """属性 10: 优化报告生成"""
    
    @given(
        num_results=st.integers(min_value=1, max_value=50),
        task_type=st.sampled_from(['LDA', 'MDA', 'LMI']),
        acquisition_function=st.sampled_from(['EI', 'PI', 'UCB']),
        include_charts=st.booleans(),
        include_errors=st.booleans()
    )
    @settings(max_examples=20, deadline=30000)
    def test_report_generation_completeness(self, num_results, task_type, acquisition_function, 
                                          include_charts, include_errors):
        """
        属性 10: 优化报告生成
        验证报告生成器能够为任何有效的优化历史生成完整的报告
        """
        # 创建测试数据
        parameter_space = create_default_parameter_space()
        history = self._create_test_history(
            num_results, task_type, acquisition_function, include_errors
        )
        
        # 创建报告生成器
        config = ReportConfig(include_charts=include_charts)
        generator = ReportGenerator(history, parameter_space, config=config)
        
        # 生成报告数据
        report_data = generator.generate_report_data()
        
        # 验证报告结构完整性
        self._verify_report_structure(report_data)
        
        # 验证报告内容一致性
        self._verify_report_consistency(report_data, history, parameter_space)
        
        # 验证JSON格式可序列化
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            try:
                generator.save_json_report(f.name)
                # 验证生成的JSON文件可以正确读取
                with open(f.name, 'r', encoding='utf-8') as rf:
                    loaded_data = json.load(rf)
                    assert isinstance(loaded_data, dict)
                    assert 'metadata' in loaded_data
                    assert 'optimization_summary' in loaded_data
            finally:
                os.unlink(f.name)
        
        # 验证HTML格式生成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            try:
                generator.save_html_report(f.name)
                # 验证生成的HTML文件存在且非空
                assert os.path.exists(f.name)
                assert os.path.getsize(f.name) > 0
                
                # 验证HTML内容包含基本结构
                with open(f.name, 'r', encoding='utf-8') as rf:
                    html_content = rf.read()
                    assert '<html' in html_content
                    assert '<title>' in html_content
                    assert '</html>' in html_content
            finally:
                os.unlink(f.name)
    
    def _create_test_history(self, num_results: int, task_type: str, 
                           acquisition_function: str, include_errors: bool) -> OptimizationHistory:
        """创建测试用的优化历史"""
        history = OptimizationHistory()
        history.task_type = task_type
        history.acquisition_function = acquisition_function
        history.start_time = datetime.now() - timedelta(hours=2)
        
        parameter_space = create_default_parameter_space()
        
        # 生成测试结果
        np.random.seed(42)  # 确保可重现性
        for i in range(num_results):
            params = parameter_space.sample_random_parameters(seed=42+i)
            
            # 模拟目标函数值
            obj_value = 0.6 + 0.3 * np.random.random()
            
            # 可选地添加错误
            error_info = None
            if include_errors and np.random.random() < 0.1:
                error_info = f"测试错误 {i}"
            
            result = OptimizationResult(
                parameters=params,
                objective_value=obj_value,
                metrics={'AUROC': obj_value, 'AUPRC': obj_value - 0.02},
                iteration=i + 1,
                timestamp=datetime.now() - timedelta(hours=2-i*0.1),
                evaluation_time=60.0 + 30 * np.random.random(),
                error_info=error_info
            )
            
            history.add_result(result)
        
        history.end_time = datetime.now()
        history.total_time = 7200.0
        
        return history
    
    def _verify_report_structure(self, report_data: Dict[str, Any]):
        """验证报告结构完整性"""
        required_sections = [
            'metadata',
            'experiment_configuration', 
            'optimization_summary',
            'statistical_summary',
            'best_parameters',
            'convergence_analysis',
            'parameter_analysis',
            'performance_metrics',
            'optimization_history',
            'recommendations'
        ]
        
        for section in required_sections:
            assert section in report_data, f"缺少必需的报告部分: {section}"
            assert report_data[section] is not None, f"报告部分 {section} 为空"
        
        # 验证元数据结构
        metadata = report_data['metadata']
        assert 'title' in metadata
        assert 'author' in metadata
        assert 'generation_time' in metadata
        assert 'system_info' in metadata
        
        # 验证系统信息
        system_info = metadata['system_info']
        assert 'task_type' in system_info
        assert 'acquisition_function' in system_info
        assert 'total_iterations' in system_info
    
    def _verify_report_consistency(self, report_data: Dict[str, Any], 
                                 history: OptimizationHistory, 
                                 parameter_space: ParameterSpace):
        """验证报告内容与原始数据的一致性"""
        # 验证基本统计信息
        opt_summary = report_data['optimization_summary']
        assert opt_summary['total_evaluations'] == len(history.results)
        
        if history.best_result:
            assert opt_summary['best_objective_value'] == history.best_result.objective_value
            assert opt_summary['best_iteration'] == history.best_result.iteration
        
        # 验证实验配置
        exp_config = report_data['experiment_configuration']
        assert exp_config['task_type'] == history.task_type
        assert exp_config['acquisition_function'] == history.acquisition_function
        
        # 验证参数空间信息
        if parameter_space:
            param_summary = exp_config['parameter_space_summary']
            assert param_summary['total_parameters'] == parameter_space.get_parameter_count()
        
        # 验证最佳参数信息
        best_params = report_data['best_parameters']
        if history.best_result and best_params['best_single_result']:
            best_single = best_params['best_single_result']
            assert best_single['iteration'] == history.best_result.iteration
            assert best_single['objective_value'] == history.best_result.objective_value
            assert best_single['parameters'] == history.best_result.parameters


class TestReportContentIntegrityProperties:
    """属性 19: 报告内容完整性"""
    
    @given(
        num_results=st.integers(min_value=5, max_value=30),
        error_rate=st.floats(min_value=0.0, max_value=0.3),
        include_metrics=st.booleans(),
        config_variations=st.dictionaries(
            keys=st.sampled_from(['include_charts', 'include_parameter_details', 
                                'include_convergence_analysis', 'include_sensitivity_analysis']),
            values=st.booleans(),
            min_size=1,
            max_size=4
        )
    )
    @settings(max_examples=15, deadline=30000)
    def test_report_content_integrity(self, num_results, error_rate, include_metrics, config_variations):
        """
        属性 19: 报告内容完整性
        验证报告内容在不同配置下的完整性和一致性
        """
        # 创建测试数据
        history = self._create_comprehensive_history(num_results, error_rate, include_metrics)
        parameter_space = create_default_parameter_space()
        
        # 创建配置
        config = ReportConfig(**config_variations)
        generator = ReportGenerator(history, parameter_space, config=config)
        
        # 生成报告数据
        report_data = generator.generate_report_data()
        
        # 验证内容完整性
        self._verify_content_completeness(report_data, config)
        
        # 验证数据一致性
        self._verify_data_consistency(report_data, history)
        
        # 验证统计计算正确性
        self._verify_statistical_accuracy(report_data, history)
        
        # 验证配置影响
        self._verify_config_impact(report_data, config)
        
        # 验证多格式一致性
        self._verify_multi_format_consistency(generator)
    
    def _create_comprehensive_history(self, num_results: int, error_rate: float, 
                                    include_metrics: bool) -> OptimizationHistory:
        """创建综合测试历史"""
        history = OptimizationHistory()
        history.task_type = "LDA"
        history.acquisition_function = "EI"
        history.start_time = datetime.now() - timedelta(hours=3)
        
        parameter_space = create_default_parameter_space()
        
        np.random.seed(123)  # 确保可重现性
        
        for i in range(num_results):
            params = parameter_space.sample_random_parameters(seed=123+i)
            
            # 创建有趋势的目标函数值（模拟优化过程）
            base_value = 0.5 + 0.4 * (1 - np.exp(-i/10))  # 指数改进
            noise = 0.05 * np.random.normal()
            obj_value = max(0.1, min(0.99, base_value + noise))
            
            # 根据错误率添加错误
            error_info = None
            if np.random.random() < error_rate:
                error_info = f"模拟错误类型{i % 3}"
            
            # 可选地添加指标
            metrics = None
            if include_metrics:
                metrics = {
                    'AUROC': obj_value,
                    'AUPRC': max(0.1, obj_value - 0.05),
                    'F1': max(0.1, obj_value - 0.1),
                    'Precision': max(0.1, obj_value - 0.02),
                    'Recall': max(0.1, obj_value - 0.03)
                }
            
            result = OptimizationResult(
                parameters=params,
                objective_value=obj_value,
                metrics=metrics,
                iteration=i + 1,
                timestamp=datetime.now() - timedelta(hours=3-i*0.1),
                evaluation_time=90.0 + 60 * np.random.random(),
                error_info=error_info
            )
            
            history.add_result(result)
        
        history.end_time = datetime.now()
        history.total_time = 10800.0
        
        return history
    
    def _verify_content_completeness(self, report_data: Dict[str, Any], config: ReportConfig):
        """验证内容完整性"""
        # 基本部分应该总是存在
        basic_sections = ['metadata', 'experiment_configuration', 'optimization_summary']
        for section in basic_sections:
            assert section in report_data
            assert report_data[section] is not None
        
        # 验证配置相关的内容
        if config.include_convergence_analysis:
            assert 'convergence_analysis' in report_data
            convergence = report_data['convergence_analysis']
            assert 'is_converged' in convergence
            assert 'convergence_curve' in convergence
        
        if config.include_sensitivity_analysis:
            assert 'parameter_analysis' in report_data
            param_analysis = report_data['parameter_analysis']
            assert 'parameter_sensitivity' in param_analysis
            assert 'importance_ranking' in param_analysis
        
        if config.include_parameter_details:
            exp_config = report_data['experiment_configuration']
            if 'parameter_space_summary' in exp_config:
                param_summary = exp_config['parameter_space_summary']
                assert 'parameter_details' in param_summary
        
        # 验证图表数据（如果启用）
        if config.include_charts:
            # 图表可能存在也可能不存在（取决于依赖）
            if 'charts' in report_data:
                charts = report_data['charts']
                assert isinstance(charts, dict)
    
    def _verify_data_consistency(self, report_data: Dict[str, Any], history: OptimizationHistory):
        """验证数据一致性"""
        # 验证基本计数
        opt_summary = report_data['optimization_summary']
        assert opt_summary['total_evaluations'] == len(history.results)
        
        successful_results = [r for r in history.results if r.error_info is None]
        failed_results = [r for r in history.results if r.error_info is not None]
        
        assert opt_summary['successful_evaluations'] == len(successful_results)
        assert opt_summary['failed_evaluations'] == len(failed_results)
        
        if len(history.results) > 0:
            expected_success_rate = len(successful_results) / len(history.results)
            assert abs(opt_summary['success_rate'] - expected_success_rate) < 1e-6
        
        # 验证最佳结果一致性
        if history.best_result:
            best_params = report_data['best_parameters']
            if best_params['best_single_result']:
                best_single = best_params['best_single_result']
                assert best_single['objective_value'] == history.best_result.objective_value
                assert best_single['iteration'] == history.best_result.iteration
        
        # 验证时间信息
        exp_config = report_data['experiment_configuration']
        if history.start_time:
            assert exp_config['optimization_start_time'] == history.start_time.isoformat()
        if history.end_time:
            assert exp_config['optimization_end_time'] == history.end_time.isoformat()
        assert exp_config['total_duration'] == history.total_time
    
    def _verify_statistical_accuracy(self, report_data: Dict[str, Any], history: OptimizationHistory):
        """验证统计计算的准确性"""
        if not history.results:
            return
        
        stats_summary = report_data['statistical_summary']
        obj_values = [r.objective_value for r in history.results]
        
        # 验证基本统计量
        assert abs(stats_summary['objective_statistics']['best'] - max(obj_values)) < 1e-6
        assert abs(stats_summary['objective_statistics']['worst'] - min(obj_values)) < 1e-6
        assert abs(stats_summary['objective_statistics']['mean'] - np.mean(obj_values)) < 1e-6
        
        # 验证评估时间统计
        eval_times = [r.evaluation_time for r in history.results]
        opt_summary = report_data['optimization_summary']
        eval_time_stats = opt_summary['evaluation_times']
        
        assert abs(eval_time_stats['total_time'] - sum(eval_times)) < 1e-6
        assert abs(eval_time_stats['average_time'] - np.mean(eval_times)) < 1e-6
        assert abs(eval_time_stats['min_time'] - min(eval_times)) < 1e-6
        assert abs(eval_time_stats['max_time'] - max(eval_times)) < 1e-6
    
    def _verify_config_impact(self, report_data: Dict[str, Any], config: ReportConfig):
        """验证配置对报告内容的影响"""
        # 验证图表配置影响
        if config.include_charts:
            # 如果启用图表，应该尝试生成图表数据
            # 但实际是否存在取决于依赖是否可用
            pass
        
        # 验证参数详情配置影响
        if config.include_parameter_details:
            exp_config = report_data['experiment_configuration']
            if 'parameter_space_summary' in exp_config:
                param_summary = exp_config['parameter_space_summary']
                assert 'parameter_details' in param_summary
        
        # 验证收敛分析配置影响
        if config.include_convergence_analysis:
            assert 'convergence_analysis' in report_data
            convergence = report_data['convergence_analysis']
            assert len(convergence) > 0  # 应该包含分析结果
        
        # 验证敏感性分析配置影响
        if config.include_sensitivity_analysis:
            assert 'parameter_analysis' in report_data
            param_analysis = report_data['parameter_analysis']
            assert 'parameter_sensitivity' in param_analysis
    
    def _verify_multi_format_consistency(self, generator: ReportGenerator):
        """验证多格式输出的一致性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 生成JSON报告
            json_path = os.path.join(temp_dir, "test.json")
            generator.save_json_report(json_path)
            
            # 生成HTML报告
            html_path = os.path.join(temp_dir, "test.html")
            generator.save_html_report(html_path)
            
            # 验证文件存在且非空
            assert os.path.exists(json_path)
            assert os.path.exists(html_path)
            assert os.path.getsize(json_path) > 0
            assert os.path.getsize(html_path) > 0
            
            # 验证JSON内容可以正确解析
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert isinstance(json_data, dict)
                assert 'metadata' in json_data
                assert 'optimization_summary' in json_data
            
            # 验证HTML内容包含基本结构
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                assert '<html' in html_content
                assert '</html>' in html_content
                assert 'optimization_summary' in html_content.lower() or '优化摘要' in html_content


class ReportGeneratorStateMachine(RuleBasedStateMachine):
    """报告生成器状态机测试"""
    
    def __init__(self):
        super().__init__()
        self.history = OptimizationHistory()
        self.parameter_space = create_default_parameter_space()
        self.results_added = 0
        
    @initialize()
    def setup_history(self):
        """初始化优化历史"""
        self.history.task_type = "LDA"
        self.history.acquisition_function = "EI"
        self.history.start_time = datetime.now()
        
    @rule(
        obj_value=st.floats(min_value=0.1, max_value=0.99),
        has_error=st.booleans(),
        has_metrics=st.booleans()
    )
    def add_result(self, obj_value, has_error, has_metrics):
        """添加优化结果"""
        params = self.parameter_space.sample_random_parameters(seed=self.results_added)
        
        error_info = f"错误 {self.results_added}" if has_error else None
        metrics = {'AUROC': obj_value} if has_metrics else None
        
        result = OptimizationResult(
            parameters=params,
            objective_value=obj_value,
            metrics=metrics,
            iteration=self.results_added + 1,
            timestamp=datetime.now(),
            evaluation_time=60.0,
            error_info=error_info
        )
        
        self.history.add_result(result)
        self.results_added += 1
    
    @rule()
    def generate_report(self):
        """生成报告并验证"""
        assume(self.results_added > 0)  # 至少需要一个结果
        
        config = ReportConfig()
        generator = ReportGenerator(self.history, self.parameter_space, config=config)
        
        # 生成报告数据
        report_data = generator.generate_report_data()
        
        # 验证基本结构
        assert 'metadata' in report_data
        assert 'optimization_summary' in report_data
        
        # 验证数据一致性
        opt_summary = report_data['optimization_summary']
        assert opt_summary['total_evaluations'] == len(self.history.results)
    
    @invariant()
    def history_consistency(self):
        """历史记录一致性不变量"""
        assert len(self.history.results) == self.results_added
        if self.history.results:
            assert all(r.iteration > 0 for r in self.history.results)
            assert all(0.1 <= r.objective_value <= 0.99 for r in self.history.results)


# 运行状态机测试
TestReportGeneratorStateMachine = ReportGeneratorStateMachine.TestCase


def test_basic_functionality():
    """基本功能测试"""
    print("运行报告生成器基本功能测试...")
    
    # 创建测试数据
    from autodl_core import create_default_parameter_space
    import numpy as np
    
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    history.start_time = datetime.now()
    
    # 添加一些测试结果
    np.random.seed(42)
    for i in range(10):
        params = parameter_space.sample_random_parameters(seed=42+i)
        obj_value = 0.7 + 0.2 * np.random.random()
        
        result = OptimizationResult(
            parameters=params,
            objective_value=obj_value,
            metrics={'AUROC': obj_value},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=60.0,
            error_info=None
        )
        history.add_result(result)
    
    history.end_time = datetime.now()
    history.total_time = 600.0
    
    # 创建报告生成器
    config = ReportConfig(include_charts=False)  # 禁用图表以加快测试
    generator = ReportGenerator(history, parameter_space, config=config)
    
    # 测试报告数据生成
    report_data = generator.generate_report_data()
    assert 'metadata' in report_data
    assert 'optimization_summary' in report_data
    assert report_data['optimization_summary']['total_evaluations'] == 10
    
    print("✓ 报告数据生成测试通过")
    
    # 测试JSON格式保存
    import tempfile
    import time
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        generator.save_json_report(json_path)
        assert os.path.exists(json_path)
        assert os.path.getsize(json_path) > 0
        print("✓ JSON报告生成测试通过")
    finally:
        try:
            time.sleep(0.1)  # 等待文件释放
            os.unlink(json_path)
        except:
            pass
    
    # 测试HTML格式保存
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_path = f.name
    
    try:
        generator.save_html_report(html_path)
        assert os.path.exists(html_path)
        assert os.path.getsize(html_path) > 0
        print("✓ HTML报告生成测试通过")
    finally:
        try:
            time.sleep(0.1)  # 等待文件释放
            os.unlink(html_path)
        except:
            pass
    
    print("所有基本功能测试完成!")


if __name__ == "__main__":
    test_basic_functionality()