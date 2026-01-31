"""
运行性能优化测试

基于基准测试结果优化性能瓶颈，调整缓冲区大小和刷新策略，
优化文件轮转和清理算法。
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from performance_benchmark import PerformanceBenchmark
from performance_optimizer import PerformanceOptimizer, OptimizationConfig


def run_baseline_benchmark():
    """运行基线基准测试"""
    print("="*60)
    print("第一步：运行基线基准测试")
    print("="*60)
    
    benchmark = PerformanceBenchmark("baseline_results")
    
    # 运行关键的基准测试
    print("\n运行基础日志测试...")
    basic_result = benchmark.benchmark_basic_logging()
    print(f"✓ 基础日志测试完成: {basic_result.operations_per_second:.1f} ops/s")
    
    print("\n运行大量日志测试...")
    volume_result = benchmark.benchmark_high_volume_logging()
    print(f"✓ 大量日志测试完成: {volume_result.operations_per_second:.1f} ops/s")
    
    print("\n运行文件操作测试...")
    file_result = benchmark.benchmark_file_operations()
    print(f"✓ 文件操作测试完成: {file_result.operations_per_second:.1f} ops/s")
    
    print("\n运行标签处理测试...")
    tag_result = benchmark.benchmark_tag_processing()
    print(f"✓ 标签处理测试完成: {tag_result.operations_per_second:.1f} ops/s")
    
    print("\n运行格式化测试...")
    format_result = benchmark.benchmark_formatting_performance()
    print(f"✓ 格式化测试完成: {format_result.operations_per_second:.1f} ops/s")
    
    results = {
        'basic_logging': basic_result,
        'high_volume_logging': volume_result,
        'file_operations': file_result,
        'tag_processing': tag_result,
        'formatting_performance': format_result
    }
    
    return results


def analyze_and_optimize(baseline_results):
    """分析基线结果并执行优化"""
    print("\n" + "="*60)
    print("第二步：分析性能瓶颈并执行优化")
    print("="*60)
    
    optimizer = PerformanceOptimizer("optimization_results")
    
    # 分析基准测试结果
    print("\n分析基准测试结果...")
    analysis = optimizer.analyze_benchmark_results(baseline_results)
    
    print(f"✓ 发现 {len(analysis['performance_bottlenecks'])} 个性能瓶颈")
    print(f"✓ 发现 {len(analysis['memory_issues'])} 个内存问题")
    print(f"✓ 发现 {len(analysis['optimization_opportunities'])} 个优化机会")
    
    # 显示详细分析结果
    if analysis['performance_bottlenecks']:
        print("\n性能瓶颈详情:")
        for bottleneck in analysis['performance_bottlenecks']:
            print(f"  - {bottleneck['test_name']}: {bottleneck['ops_per_second']:.1f} ops/s ({bottleneck['severity']})")
    
    if analysis['memory_issues']:
        print("\n内存问题详情:")
        for issue in analysis['memory_issues']:
            print(f"  - {issue['test_name']}: 峰值 {issue['peak_memory_mb']:.1f} MB ({issue['severity']})")
    
    if analysis['optimization_opportunities']:
        print("\n优化机会:")
        for opp in analysis['optimization_opportunities']:
            print(f"  - {opp['type']}: {opp['description']} (优先级: {opp['priority']})")
    
    # 执行优化
    print("\n执行性能优化...")
    optimization_results = []
    
    # 缓冲区优化
    if any(b['ops_per_second'] < 5000 for b in analysis['performance_bottlenecks']):
        print("\n执行缓冲区优化...")
        buffer_result = optimizer.optimize_buffer_configuration(baseline_results)
        optimization_results.append(buffer_result)
        print(f"✓ 缓冲区优化完成: 性能提升 {buffer_result.performance_improvement:.2f}%")
    
    # 内存优化
    if analysis['memory_issues']:
        print("\n执行内存优化...")
        memory_result = optimizer.optimize_memory_usage(baseline_results)
        optimization_results.append(memory_result)
        print(f"✓ 内存优化完成: 内存减少 {memory_result.memory_reduction:.2f}%")
    
    # 文件操作优化
    file_test = baseline_results.get('file_operations')
    if file_test and file_test.operations_per_second < 2000:
        print("\n执行文件操作优化...")
        file_result = optimizer.optimize_file_operations(baseline_results)
        optimization_results.append(file_result)
        print(f"✓ 文件操作优化完成: 性能提升 {file_result.performance_improvement:.2f}%")
    
    return analysis, optimization_results


def generate_optimization_recommendations(analysis, optimization_results):
    """生成优化建议"""
    print("\n" + "="*60)
    print("第三步：生成优化建议和配置")
    print("="*60)
    
    recommendations = []
    
    # 基于分析结果生成建议
    if analysis['performance_bottlenecks']:
        recommendations.extend([
            "缓冲区优化建议:",
            "  - 将缓冲区大小增加到 5000-10000",
            "  - 将刷新间隔调整为 2-3 秒",
            "  - 启用批量写入模式"
        ])
    
    if analysis['memory_issues']:
        recommendations.extend([
            "内存优化建议:",
            "  - 启用压缩以减少内存占用",
            "  - 设置内存阈值为 300MB",
            "  - 实施对象池和复用机制"
        ])
    
    if analysis['io_bottlenecks']:
        recommendations.extend([
            "I/O优化建议:",
            "  - 增加文件大小限制到 200MB",
            "  - 启用异步I/O操作",
            "  - 优化文件轮转策略"
        ])
    
    # 基于优化结果生成具体配置
    if optimization_results:
        best_config = OptimizationConfig()
        
        for result in optimization_results:
            if result.performance_improvement > 10:  # 如果性能提升超过10%
                # 合并有效的配置
                if result.optimized_config.buffer_size > best_config.buffer_size:
                    best_config.buffer_size = result.optimized_config.buffer_size
                
                if result.optimized_config.flush_interval > best_config.flush_interval:
                    best_config.flush_interval = result.optimized_config.flush_interval
                
                if result.optimized_config.compression_enabled:
                    best_config.compression_enabled = True
                
                if result.optimized_config.max_file_size > best_config.max_file_size:
                    best_config.max_file_size = result.optimized_config.max_file_size
        
        recommendations.extend([
            "",
            "推荐的最佳配置:",
            f"  - buffer_size: {best_config.buffer_size}",
            f"  - flush_interval: {best_config.flush_interval} 秒",
            f"  - max_file_size: {best_config.max_file_size // (1024*1024)} MB",
            f"  - compression_enabled: {best_config.compression_enabled}",
            f"  - concurrent_workers: {best_config.concurrent_workers}",
            f"  - memory_threshold_mb: {best_config.memory_threshold_mb}"
        ])
        
        # 保存最佳配置
        config_file = Path("optimization_results") / f"best_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(best_config.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 最佳配置已保存到: {config_file}")
    
    # 显示建议
    print("\n优化建议:")
    for rec in recommendations:
        print(rec)
    
    return recommendations


def validate_optimizations(baseline_results, optimization_results):
    """验证优化效果"""
    print("\n" + "="*60)
    print("第四步：验证优化效果")
    print("="*60)
    
    if not optimization_results:
        print("⚠️  没有执行任何优化")
        return
    
    print("\n优化效果总结:")
    
    total_performance_improvement = 0
    total_memory_reduction = 0
    
    for result in optimization_results:
        print(f"\n{result.optimization_name}:")
        print(f"  性能提升: {result.performance_improvement:.2f}%")
        print(f"  内存减少: {result.memory_reduction:.2f}%")
        print(f"  错误减少: {result.error_reduction:.2f}%")
        
        total_performance_improvement += result.performance_improvement
        total_memory_reduction += result.memory_reduction
        
        print("  主要建议:")
        for rec in result.recommendations[:3]:  # 显示前3个建议
            print(f"    - {rec}")
    
    avg_performance_improvement = total_performance_improvement / len(optimization_results)
    avg_memory_reduction = total_memory_reduction / len(optimization_results)
    
    print(f"\n总体优化效果:")
    print(f"  平均性能提升: {avg_performance_improvement:.2f}%")
    print(f"  平均内存减少: {avg_memory_reduction:.2f}%")
    print(f"  执行优化数量: {len(optimization_results)}")
    
    # 评估优化成功度
    if avg_performance_improvement > 20:
        print("✅ 优化效果显著！")
    elif avg_performance_improvement > 10:
        print("✅ 优化效果良好")
    elif avg_performance_improvement > 0:
        print("⚠️  优化效果一般")
    else:
        print("❌ 优化效果不明显")


def main():
    """主函数"""
    print("性能优化和调整测试")
    print("基于基准测试结果优化性能瓶颈，调整缓冲区大小和刷新策略")
    print("优化文件轮转和清理算法")
    
    start_time = datetime.now()
    
    try:
        # 第一步：运行基线基准测试
        baseline_results = run_baseline_benchmark()
        
        # 第二步：分析和优化
        analysis, optimization_results = analyze_and_optimize(baseline_results)
        
        # 第三步：生成优化建议
        recommendations = generate_optimization_recommendations(analysis, optimization_results)
        
        # 第四步：验证优化效果
        validate_optimizations(baseline_results, optimization_results)
        
        # 总结
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("性能优化测试完成")
        print("="*60)
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"基线测试数量: {len(baseline_results)}")
        print(f"执行优化数量: {len(optimization_results)}")
        print(f"生成建议数量: {len(recommendations)}")
        
        print("\n✅ 性能优化和调整任务完成！")
        print("📁 结果文件保存在 baseline_results/ 和 optimization_results/ 目录")
        
    except Exception as e:
        print(f"\n❌ 性能优化测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()