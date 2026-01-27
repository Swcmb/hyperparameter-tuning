"""
结果分析器测试

测试ResultAnalyzer的核心功能
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from result_analyzer import ResultAnalyzer, ParameterSensitivityResult, ConvergenceAnalysisResult, StatisticalSummary, create_result_analyzer_from_checkpoint
from autodl_core import OptimizationHistory, OptimizationResult, create_default_parameter_space


class TestResultAnalyzer:
    """结果分析器测试类"""
    
    @pytest.fixture
    def sample_history(self):
        """创建示例优化历史"""
        history = OptimizationHistory()
        history.task_type = "LDA"
        history.acquisition_function = "EI"
        history.start_time = datetime.now() - timedelta(hours=1)
        
        # 创建一些示例结果
        np.random.seed(42)
        for i in range(20):
            params = {
                'lr': 0.001 * (1 + i * 0.1),
                'dimensions': 128 + i * 10,
                'batch': 32 if i % 2 == 0 else 64,
                'fusion_strategy': 'co_attention' if i % 3 == 0 else 'self_attention'
            }
            
            # 模拟目标函数值
            obj_value = 0.8 + 0.1 * np.random.random() + (0.02 if params['fusion_strategy'] == 'co_attention' else 0)
            
            result = OptimizationResult(
                parameters=params,
                objective_value=obj_value,
                metrics={'AUROC': obj_value, 'AUPRC': obj_value - 0.02},
                iteration=i + 1,
                timestamp=history.start_time + timedelta(minutes=i*5),
                evaluation_time=100 + 20 * np.random.random()
            )
            
            history.add_result(result)
        
        history.end_time = datetime.now()
        return history
    
    @pytest.fixture
    def analyzer(self, sample_history):
        """创建结果分析器"""
        parameter_space = create_default_parameter_space()
        return ResultAnalyzer(sample_history, parameter_space)
    
    def test_initialization(self, sample_history):
        """测试初始化"""
        analyzer = ResultAnalyzer(sample_history)
        assert analyzer.history == sample_history
        assert len(analyzer.results_df) == 20
        assert 'objective_value' in analyzer.results_df.columns
        assert 'param_lr' in analyzer.results_df.columns
    
    def test_statistical_summary(self, analyzer):
        """测试统计摘要"""
        summary = analyzer.get_statistical_summary()
        
        assert isinstance(summary, StatisticalSummary)
        assert summary.total_evaluations == 20
        assert 0.8 <= summary.best_objective_value <= 1.0
        assert 0.7 <= summary.mean_objective_value <= 1.0
        assert summary.std_objective_value >= 0
        assert summary.success_rate == 1.0  # 没有错误
        assert summary.average_evaluation_time > 0
    
    def test_parameter_sensitivity_analysis(self, analyzer):
        """测试参数敏感性分析"""
        results = analyzer.analyze_parameter_sensitivity()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, ParameterSensitivityResult)
            assert result.parameter_name in ['lr', 'dimensions', 'batch', 'fusion_strategy']
            assert 0 <= result.importance_rank <= len(results)
            assert -1 <= result.correlation_coefficient <= 1
            assert 0 <= result.p_value <= 1
    
    def test_convergence_analysis(self, analyzer):
        """测试收敛性分析"""
        result = analyzer.analyze_convergence()
        
        assert isinstance(result, ConvergenceAnalysisResult)
        assert isinstance(result.is_converged, bool)
        assert result.convergence_threshold == 0.001
        assert isinstance(result.improvement_rate, float)
        assert result.plateau_length >= 0
        
        if result.is_converged:
            assert result.convergence_iteration is not None
            assert result.convergence_iteration >= 0
    
    def test_parameter_importance_ranking(self, analyzer):
        """测试参数重要性排序"""
        ranking = analyzer.get_parameter_importance_ranking()
        
        assert isinstance(ranking, list)
        assert len(ranking) > 0
        
        # 检查排序是否正确（按重要性降序）
        scores = [score for _, score in ranking]
        assert scores == sorted(scores, reverse=True)
        
        # 检查参数名称
        param_names = [name for name, _ in ranking]
        expected_params = ['lr', 'dimensions', 'batch', 'fusion_strategy']
        for param in param_names:
            assert param in expected_params
    
    def test_best_parameters_analysis(self, analyzer):
        """测试最佳参数分析"""
        analysis = analyzer.get_best_parameters_analysis(top_k=5)
        
        assert isinstance(analysis, dict)
        assert 'top_k' in analysis
        assert 'best_objective_value' in analysis
        assert 'mean_top_k_objective' in analysis
        assert 'parameter_statistics' in analysis
        
        assert analysis['top_k'] == 5
        assert analysis['best_objective_value'] >= analysis['mean_top_k_objective']
        
        # 检查参数统计
        param_stats = analysis['parameter_statistics']
        assert 'lr' in param_stats
        assert 'dimensions' in param_stats
        
        for param_name, stats in param_stats.items():
            assert 'type' in stats
            assert stats['type'] in ['categorical', 'numerical']
    
    def test_parameter_correlations(self, analyzer):
        """测试参数相关性分析"""
        correlation_matrix = analyzer.analyze_parameter_correlations()
        
        if not correlation_matrix.empty:
            # 检查矩阵是对称的
            assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
            
            # 检查对角线为1
            for i in range(len(correlation_matrix)):
                assert abs(correlation_matrix.iloc[i, i] - 1.0) < 1e-10
            
            # 检查相关系数在[-1, 1]范围内
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    corr = correlation_matrix.iloc[i, j]
                    assert -1 <= corr <= 1
    
    def test_convergence_curve(self, analyzer):
        """测试收敛曲线"""
        curve = analyzer.get_convergence_curve()
        
        assert isinstance(curve, list)
        assert len(curve) == 20
        
        # 检查收敛曲线是非递减的
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i-1]
    
    def test_parameter_history(self, analyzer):
        """测试参数历史"""
        lr_history = analyzer.get_parameter_history('lr')
        
        assert isinstance(lr_history, list)
        assert len(lr_history) == 20
        
        # 检查学习率值的合理性
        for lr in lr_history:
            assert lr is not None
            assert 0.001 <= lr <= 0.003  # 基于我们的测试数据
    
    def test_generate_analysis_report(self, analyzer):
        """测试分析报告生成"""
        report = analyzer.generate_analysis_report()
        
        assert isinstance(report, dict)
        
        # 检查必需的部分
        required_sections = [
            'analysis_timestamp',
            'optimization_summary',
            'statistical_summary',
            'parameter_sensitivity',
            'convergence_analysis',
            'parameter_importance_ranking',
            'best_parameters_analysis'
        ]
        
        for section in required_sections:
            assert section in report
        
        # 检查优化摘要
        opt_summary = report['optimization_summary']
        assert opt_summary['task_type'] == 'LDA'
        assert opt_summary['acquisition_function'] == 'EI'
        assert opt_summary['total_iterations'] == 20
    
    def test_identify_parameter_patterns(self, analyzer):
        """测试参数模式识别"""
        patterns = analyzer.identify_parameter_patterns()
        
        assert isinstance(patterns, dict)
        
        if 'warning' not in patterns:
            # 检查模式分析结果
            for param_name, pattern in patterns.items():
                assert isinstance(pattern, dict)
                assert 'trend_correlation' in pattern
                assert 'correlation_stability' in pattern
                assert 'trend_direction' in pattern
                
                assert pattern['trend_direction'] in ['increasing', 'decreasing', 'stable']
                assert -1 <= pattern['trend_correlation'] <= 1
                assert pattern['correlation_stability'] >= 0
    
    def test_empty_history(self):
        """测试空历史记录的处理"""
        empty_history = OptimizationHistory()
        analyzer = ResultAnalyzer(empty_history)
        
        # 统计摘要应该返回默认值
        summary = analyzer.get_statistical_summary()
        assert summary.total_evaluations == 0
        assert summary.best_objective_value == 0.0
        
        # 敏感性分析应该返回空列表
        sensitivity = analyzer.analyze_parameter_sensitivity()
        assert len(sensitivity) == 0
        
        # 重要性排序应该返回空列表
        ranking = analyzer.get_parameter_importance_ranking()
        assert len(ranking) == 0
    
    def test_caching(self, analyzer):
        """测试结果缓存"""
        # 第一次调用
        result1 = analyzer.analyze_parameter_sensitivity()
        
        # 第二次调用应该返回缓存的结果
        result2 = analyzer.analyze_parameter_sensitivity()
        
        assert result1 is result2  # 应该是同一个对象（缓存）
    
    def test_save_and_load_report(self, analyzer, tmp_path):
        """测试报告保存和加载"""
        report_path = tmp_path / "test_report.json"
        
        # 保存报告
        analyzer.save_analysis_report(str(report_path))
        
        # 检查文件是否存在
        assert report_path.exists()
        
        # 加载并验证内容
        import json
        with open(report_path, 'r', encoding='utf-8') as f:
            loaded_report = json.load(f)
        
        assert isinstance(loaded_report, dict)
        assert 'analysis_timestamp' in loaded_report
        assert 'optimization_summary' in loaded_report


def test_create_result_analyzer_from_checkpoint():
    """测试从检查点创建分析器"""
    # 测试不存在的文件
    analyzer = create_result_analyzer_from_checkpoint("nonexistent.pkl")
    assert analyzer is None


if __name__ == "__main__":
    # 运行基本测试
    print("运行结果分析器测试...")
    
    # 创建测试数据
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    
    np.random.seed(42)
    for i in range(10):
        params = {
            'lr': 0.001 + i * 0.0001,
            'dimensions': 128 + i * 10,
            'batch': 32
        }
        
        result = OptimizationResult(
            parameters=params,
            objective_value=0.8 + 0.1 * np.random.random(),
            metrics={'AUROC': 0.85},
            iteration=i + 1,
            timestamp=datetime.now(),
            evaluation_time=100.0
        )
        
        history.add_result(result)
    
    # 创建分析器并测试
    analyzer = ResultAnalyzer(history)
    
    print(f"✓ 创建分析器成功，包含 {len(analyzer.results_df)} 个结果")
    
    # 测试统计摘要
    summary = analyzer.get_statistical_summary()
    print(f"✓ 统计摘要: 最佳值={summary.best_objective_value:.4f}")
    
    # 测试敏感性分析
    sensitivity = analyzer.analyze_parameter_sensitivity()
    print(f"✓ 敏感性分析: 找到 {len(sensitivity)} 个参数")
    
    # 测试收敛性分析
    convergence = analyzer.analyze_convergence()
    print(f"✓ 收敛性分析: 收敛={convergence.is_converged}")
    
    # 测试报告生成
    report = analyzer.generate_analysis_report()
    print(f"✓ 报告生成: 包含 {len(report)} 个部分")
    
    print("所有基本测试通过！")