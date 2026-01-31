"""
报告生成器单元测试

测试ReportGenerator的核心功能
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
import numpy as np

from autodl_core import OptimizationHistory, OptimizationResult, create_default_parameter_space
from report_generator import ReportGenerator, ReportConfig


def test_basic_report_generation():
    """测试基本报告生成功能"""
    print("测试基本报告生成功能...")
    
    # 创建测试数据
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    
    # 添加一些测试结果
    for i in range(10):
        params = parameter_space.sample_random_parameters(seed=42+i)
        result = OptimizationResult(
            parameters=params,
            objective_value=0.8 + 0.1 * np.random.random(),
            metrics={'AUROC': 0.85, 'AUPRC': 0.78},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=100.0
        )
        history.add_result(result)
    
    # 创建报告生成器
    generator = ReportGenerator(history, parameter_space)
    
    # 测试报告数据生成
    report_data = generator.generate_report_data()
    
    # 验证报告结构
    expected_sections = [
        'metadata', 'experiment_configuration', 'optimization_summary',
        'statistical_summary', 'best_parameters', 'convergence_analysis',
        'parameter_analysis', 'performance_metrics', 'optimization_history',
        'recommendations'
    ]
    
    for section in expected_sections:
        assert section in report_data, f"缺少报告部分: {section}"
    
    # 验证关键数据
    assert report_data['metadata']['title'] is not None
    assert report_data['optimization_summary']['total_evaluations'] == 10
    assert report_data['best_parameters']['best_single_result'] is not None
    
    print("✓ 基本报告生成功能测试通过")


def test_json_report_generation():
    """测试JSON报告生成"""
    print("测试JSON报告生成...")
    
    # 创建测试数据
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    history.task_type = "MDA"
    
    # 添加测试结果
    for i in range(5):
        params = parameter_space.sample_random_parameters(seed=100+i)
        result = OptimizationResult(
            parameters=params,
            objective_value=0.75 + 0.15 * np.random.random(),
            metrics={'AUROC': 0.82, 'F1': 0.75},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=90.0
        )
        history.add_result(result)
    
    # 创建报告生成器
    config = ReportConfig(title="测试JSON报告", include_charts=False)
    generator = ReportGenerator(history, parameter_space, config=config)
    
    # 生成JSON报告
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = os.path.join(temp_dir, "test_report.json")
        generator.save_json_report(json_path)
        
        # 验证文件存在
        assert os.path.exists(json_path), "JSON报告文件未生成"
        
        # 验证JSON格式
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert data['metadata']['title'] == "测试JSON报告"
        assert data['optimization_summary']['total_evaluations'] == 5
    
    print("✓ JSON报告生成测试通过")


def test_html_report_generation():
    """测试HTML报告生成"""
    print("测试HTML报告生成...")
    
    # 创建测试数据
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    history.task_type = "LMI"
    
    # 添加测试结果
    for i in range(8):
        params = parameter_space.sample_random_parameters(seed=200+i)
        result = OptimizationResult(
            parameters=params,
            objective_value=0.7 + 0.2 * np.random.random(),
            metrics={'AUROC': 0.80, 'AUPRC': 0.76, 'F1': 0.72},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=110.0
        )
        history.add_result(result)
    
    # 创建报告生成器
    config = ReportConfig(title="测试HTML报告", include_charts=True)
    generator = ReportGenerator(history, parameter_space, config=config)
    
    # 生成HTML报告
    with tempfile.TemporaryDirectory() as temp_dir:
        html_path = os.path.join(temp_dir, "test_report.html")
        generator.save_html_report(html_path)
        
        # 验证文件存在
        assert os.path.exists(html_path), "HTML报告文件未生成"
        
        # 验证HTML内容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert '<html' in html_content
        assert '测试HTML报告' in html_content
        assert 'LMI' in html_content
    
    print("✓ HTML报告生成测试通过")


def test_custom_config():
    """测试自定义配置"""
    print("测试自定义配置...")
    
    # 创建测试数据
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    
    # 添加测试结果
    for i in range(3):
        params = parameter_space.sample_random_parameters(seed=300+i)
        result = OptimizationResult(
            parameters=params,
            objective_value=0.85,
            metrics={'AUROC': 0.85},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=80.0
        )
        history.add_result(result)
    
    # 创建自定义配置
    config = ReportConfig(
        title="自定义配置测试",
        author="测试用户",
        include_charts=False,
        include_parameter_details=False,
        include_convergence_analysis=False,
        include_sensitivity_analysis=True
    )
    
    generator = ReportGenerator(history, parameter_space, config=config)
    report_data = generator.generate_report_data()
    
    # 验证配置生效
    assert report_data['metadata']['title'] == "自定义配置测试"
    assert report_data['metadata']['author'] == "测试用户"
    assert 'charts' not in report_data  # 图表被禁用
    assert not report_data['experiment_configuration']['parameter_space_summary']['parameter_details']  # 参数详情被禁用
    assert not report_data['convergence_analysis']  # 收敛分析被禁用
    assert report_data['parameter_analysis']  # 敏感性分析启用
    
    print("✓ 自定义配置测试通过")


def test_error_handling():
    """测试错误处理"""
    print("测试错误处理...")
    
    # 测试空历史数据
    empty_history = OptimizationHistory()
    generator = ReportGenerator(empty_history)
    
    report_data = generator.generate_report_data()
    
    # 验证空数据处理
    assert report_data['optimization_summary']['total_evaluations'] == 0
    assert report_data['best_parameters']['best_single_result'] is None
    assert report_data['statistical_summary']['total_evaluations'] == 0
    
    print("✓ 错误处理测试通过")


def test_report_data_access():
    """测试报告数据访问"""
    print("测试报告数据访问...")
    
    # 创建测试数据
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    
    # 添加测试结果
    best_params = parameter_space.sample_random_parameters(seed=400)
    best_result = OptimizationResult(
        parameters=best_params,
        objective_value=0.95,  # 最高分
        metrics={'AUROC': 0.95, 'AUPRC': 0.90, 'F1': 0.88},
        iteration=1,
        timestamp=datetime.now(),
        evaluation_time=120.0
    )
    history.add_result(best_result)
    
    # 添加其他结果
    for i in range(2, 6):
        params = parameter_space.sample_random_parameters(seed=400+i)
        result = OptimizationResult(
            parameters=params,
            objective_value=0.8 + 0.1 * np.random.random(),
            metrics={'AUROC': 0.82, 'AUPRC': 0.78, 'F1': 0.75},
            iteration=i,
            timestamp=datetime.now(),
            evaluation_time=100.0
        )
        history.add_result(result)
    
    generator = ReportGenerator(history, parameter_space)
    report_data = generator.generate_report_data()
    
    # 验证最佳结果
    best_single = report_data['best_parameters']['best_single_result']
    assert best_single is not None
    assert best_single['objective_value'] == 0.95
    assert best_single['iteration'] == 1
    
    # 验证参数重要性
    importance_ranking = report_data['parameter_analysis']['importance_ranking']
    assert len(importance_ranking) > 0
    assert all(isinstance(item, (list, tuple)) and len(item) == 2 for item in importance_ranking)
    
    print("✓ 报告数据访问测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行报告生成器测试...")
    print("=" * 50)
    
    try:
        test_basic_report_generation()
        test_json_report_generation()
        test_html_report_generation()
        test_custom_config()
        test_error_handling()
        test_report_data_access()
        
        print("=" * 50)
        print("✅ 所有测试通过!")
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)