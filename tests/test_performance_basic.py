"""
基础性能测试脚本

简化版本的性能测试，用于验证核心功能
"""

import time
import logging
from datetime import datetime
from performance_benchmark import PerformanceBenchmark, BenchmarkResult
from performance_optimizer import PerformanceOptimizer


def test_basic_performance():
    """测试基础性能功能"""
    print("开始基础性能测试...")
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark("test_benchmark_results")
    
    # 运行单个测试
    print("\n1. 运行基础日志测试...")
    basic_result = benchmark.benchmark_basic_logging()
    print(f"   操作数: {basic_result.operations_count}")
    print(f"   耗时: {basic_result.duration_seconds:.2f} 秒")
    print(f"   吞吐量: {basic_result.operations_per_second:.1f} ops/s")
    print(f"   成功率: {basic_result.success_rate:.2f}%")
    print(f"   内存使用: {basic_result.peak_memory_mb:.1f} MB")
    
    print("\n2. 运行标签处理测试...")
    tag_result = benchmark.benchmark_tag_processing()
    print(f"   操作数: {tag_result.operations_count}")
    print(f"   耗时: {tag_result.duration_seconds:.2f} 秒")
    print(f"   吞吐量: {tag_result.operations_per_second:.1f} ops/s")
    print(f"   成功率: {tag_result.success_rate:.2f}%")
    
    print("\n3. 运行格式化测试...")
    format_result = benchmark.benchmark_formatting_performance()
    print(f"   操作数: {format_result.operations_count}")
    print(f"   耗时: {format_result.duration_seconds:.2f} 秒")
    print(f"   吞吐量: {format_result.operations_per_second:.1f} ops/s")
    print(f"   成功率: {format_result.success_rate:.2f}%")
    
    # 收集结果
    results = {
        'basic_logging': basic_result,
        'tag_processing': tag_result,
        'formatting_performance': format_result
    }
    
    return results


def test_performance_optimization():
    """测试性能优化功能"""
    print("\n开始性能优化测试...")
    
    # 创建模拟的基准测试结果
    mock_results = {
        'basic_logging': BenchmarkResult(
            test_name="basic_logging",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=10.0,
            operations_count=10000,
            operations_per_second=800,  # 较低的性能，需要优化
            memory_usage_mb=150.0,
            peak_memory_mb=200.0,
            cpu_usage_percent=45.0,
            disk_io_read_mb=5.0,
            disk_io_write_mb=25.0,
            success_rate=98.5,
            error_count=15,
            additional_metrics={}
        ),
        'high_volume_logging': BenchmarkResult(
            test_name="high_volume_logging",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=50.0,
            operations_count=100000,
            operations_per_second=2000,  # 需要优化
            memory_usage_mb=300.0,
            peak_memory_mb=450.0,  # 内存使用较高
            cpu_usage_percent=65.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=120.0,
            success_rate=99.2,
            error_count=80,
            additional_metrics={}
        )
    }
    
    # 创建性能优化器
    optimizer = PerformanceOptimizer("test_optimization_results")
    
    # 分析基准测试结果
    print("\n1. 分析基准测试结果...")
    analysis = optimizer.analyze_benchmark_results(mock_results)
    
    print(f"   发现性能瓶颈: {len(analysis['performance_bottlenecks'])}")
    print(f"   发现内存问题: {len(analysis['memory_issues'])}")
    print(f"   优化机会: {len(analysis['optimization_opportunities'])}")
    
    # 显示分析结果
    if analysis['performance_bottlenecks']:
        print("\n   性能瓶颈详情:")
        for bottleneck in analysis['performance_bottlenecks']:
            print(f"     - {bottleneck['test_name']}: {bottleneck['ops_per_second']:.1f} ops/s ({bottleneck['severity']})")
    
    if analysis['memory_issues']:
        print("\n   内存问题详情:")
        for issue in analysis['memory_issues']:
            print(f"     - {issue['test_name']}: 峰值 {issue['peak_memory_mb']:.1f} MB ({issue['severity']})")
    
    if analysis['optimization_opportunities']:
        print("\n   优化机会:")
        for opp in analysis['optimization_opportunities']:
            print(f"     - {opp['type']}: {opp['description']} (优先级: {opp['priority']})")
    
    return analysis


def main():
    """主函数"""
    print("="*60)
    print("性能测试和优化验证")
    print("="*60)
    
    try:
        # 测试基础性能
        performance_results = test_basic_performance()
        
        # 测试性能优化
        optimization_analysis = test_performance_optimization()
        
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        
        print("\n性能测试结果:")
        for test_name, result in performance_results.items():
            print(f"  {test_name}:")
            print(f"    吞吐量: {result.operations_per_second:.1f} ops/s")
            print(f"    成功率: {result.success_rate:.2f}%")
            print(f"    内存峰值: {result.peak_memory_mb:.1f} MB")
        
        print(f"\n优化分析结果:")
        print(f"  性能瓶颈数量: {len(optimization_analysis['performance_bottlenecks'])}")
        print(f"  内存问题数量: {len(optimization_analysis['memory_issues'])}")
        print(f"  优化机会数量: {len(optimization_analysis['optimization_opportunities'])}")
        
        print("\n✅ 所有测试完成！性能测试和优化系统运行正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()