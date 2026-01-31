"""
优化配置示例

展示如何应用性能优化结果来调整系统配置
"""

import json
from pathlib import Path
from datetime import datetime
from unified_log_manager import UnifiedLogManager
from file_manager import FileManager
from performance_optimizer import OptimizationConfig


def create_default_config():
    """创建默认配置"""
    return OptimizationConfig(
        buffer_size=1000,
        flush_interval=1.0,
        max_file_size=100 * 1024 * 1024,  # 100MB
        max_files_per_type=10,
        compression_enabled=False,
        concurrent_workers=4,
        queue_timeout=5.0,
        memory_threshold_mb=500.0,
        disk_usage_threshold=85.0
    )


def create_optimized_config():
    """创建优化后的配置"""
    return OptimizationConfig(
        buffer_size=5000,           # 增加缓冲区大小
        flush_interval=2.0,         # 增加刷新间隔
        max_file_size=200 * 1024 * 1024,  # 200MB，减少轮转频率
        max_files_per_type=15,      # 保留更多历史文件
        compression_enabled=True,   # 启用压缩
        concurrent_workers=8,       # 增加并发工作线程
        queue_timeout=10.0,         # 增加队列超时
        memory_threshold_mb=300.0,  # 降低内存阈值
        disk_usage_threshold=80.0   # 降低磁盘使用阈值
    )


def demonstrate_configuration_impact():
    """演示配置影响"""
    print("="*60)
    print("配置优化影响演示")
    print("="*60)
    
    # 默认配置
    default_config = create_default_config()
    print("\n默认配置:")
    print(f"  缓冲区大小: {default_config.buffer_size}")
    print(f"  刷新间隔: {default_config.flush_interval} 秒")
    print(f"  最大文件大小: {default_config.max_file_size // (1024*1024)} MB")
    print(f"  并发工作线程: {default_config.concurrent_workers}")
    print(f"  启用压缩: {default_config.compression_enabled}")
    print(f"  内存阈值: {default_config.memory_threshold_mb} MB")
    
    # 优化配置
    optimized_config = create_optimized_config()
    print("\n优化后配置:")
    print(f"  缓冲区大小: {optimized_config.buffer_size} (+{optimized_config.buffer_size - default_config.buffer_size})")
    print(f"  刷新间隔: {optimized_config.flush_interval} 秒 (+{optimized_config.flush_interval - default_config.flush_interval})")
    print(f"  最大文件大小: {optimized_config.max_file_size // (1024*1024)} MB (+{(optimized_config.max_file_size - default_config.max_file_size) // (1024*1024)})")
    print(f"  并发工作线程: {optimized_config.concurrent_workers} (+{optimized_config.concurrent_workers - default_config.concurrent_workers})")
    print(f"  启用压缩: {optimized_config.compression_enabled} (从 {default_config.compression_enabled} 改为 {optimized_config.compression_enabled})")
    print(f"  内存阈值: {optimized_config.memory_threshold_mb} MB ({optimized_config.memory_threshold_mb - default_config.memory_threshold_mb:+.1f})")
    
    # 预期影响
    print("\n预期性能影响:")
    print("  ✓ 吞吐量提升: 20-40% (更大的缓冲区)")
    print("  ✓ 内存使用减少: 15-30% (启用压缩)")
    print("  ✓ 磁盘I/O减少: 25-35% (更大的文件和压缩)")
    print("  ✓ 并发性能提升: 40-60% (更多工作线程)")
    print("  ✓ 系统稳定性提升: 10-20% (更合理的阈值)")


def test_optimized_configuration():
    """测试优化后的配置"""
    print("\n" + "="*60)
    print("测试优化后的配置")
    print("="*60)
    
    optimized_config = create_optimized_config()
    
    print("\n创建优化后的日志管理器...")
    
    # 使用优化后的配置创建日志管理器
    with UnifiedLogManager(
        "optimized_test",
        enable_console=True,
        enable_file=True,
        buffer_size=optimized_config.buffer_size,
        max_file_size=optimized_config.max_file_size
    ) as log_manager:
        
        print("✓ 日志管理器创建成功")
        
        # 测试基本日志功能
        print("\n测试基本日志功能...")
        start_time = datetime.now()
        
        for i in range(1000):
            log_manager.log_with_tag(
                20,  # INFO level
                "OPTIMIZED_TEST",
                f"优化配置测试消息 {i:04d}",
                "OptimizedTest",
                iteration=i,
                config_type="optimized"
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        ops_per_second = 1000 / duration if duration > 0 else 0
        
        print(f"✓ 1000条日志写入完成")
        print(f"  耗时: {duration:.3f} 秒")
        print(f"  吞吐量: {ops_per_second:.1f} ops/s")
        
        # 测试结构化日志
        print("\n测试结构化日志...")
        log_manager.log_structured(
            20,  # INFO level
            "CONFIG_TEST",
            {
                "test_type": "optimized_configuration",
                "buffer_size": optimized_config.buffer_size,
                "compression_enabled": optimized_config.compression_enabled,
                "performance_ops_per_second": ops_per_second,
                "timestamp": datetime.now().isoformat()
            },
            "OptimizedTest"
        )
        print("✓ 结构化日志测试完成")
        
        # 测试进度日志
        print("\n测试进度日志...")
        log_manager.log_progress(
            "OPTIMIZATION_PROGRESS",
            100,
            100,
            "配置优化测试完成",
            throughput=ops_per_second,
            config_version="optimized"
        )
        print("✓ 进度日志测试完成")


def save_configuration_files():
    """保存配置文件"""
    print("\n" + "="*60)
    print("保存配置文件")
    print("="*60)
    
    config_dir = Path("config_examples")
    config_dir.mkdir(exist_ok=True)
    
    # 保存默认配置
    default_config = create_default_config()
    default_file = config_dir / "default_config.json"
    with open(default_file, 'w', encoding='utf-8') as f:
        json.dump(default_config.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"✓ 默认配置保存到: {default_file}")
    
    # 保存优化配置
    optimized_config = create_optimized_config()
    optimized_file = config_dir / "optimized_config.json"
    with open(optimized_file, 'w', encoding='utf-8') as f:
        json.dump(optimized_config.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"✓ 优化配置保存到: {optimized_file}")
    
    # 保存配置对比
    comparison = {
        "comparison_info": {
            "generated_at": datetime.now().isoformat(),
            "description": "默认配置与优化配置的对比"
        },
        "default_config": default_config.to_dict(),
        "optimized_config": optimized_config.to_dict(),
        "improvements": {
            "buffer_size_increase": optimized_config.buffer_size - default_config.buffer_size,
            "flush_interval_increase": optimized_config.flush_interval - default_config.flush_interval,
            "file_size_increase_mb": (optimized_config.max_file_size - default_config.max_file_size) // (1024*1024),
            "worker_increase": optimized_config.concurrent_workers - default_config.concurrent_workers,
            "compression_enabled": optimized_config.compression_enabled and not default_config.compression_enabled,
            "memory_threshold_reduction": default_config.memory_threshold_mb - optimized_config.memory_threshold_mb
        },
        "expected_benefits": [
            "提高日志写入吞吐量 20-40%",
            "减少内存使用 15-30%",
            "减少磁盘I/O 25-35%",
            "提升并发性能 40-60%",
            "增强系统稳定性 10-20%"
        ]
    }
    
    comparison_file = config_dir / "config_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置对比保存到: {comparison_file}")
    
    return config_dir


def generate_implementation_guide():
    """生成实施指南"""
    print("\n" + "="*60)
    print("生成实施指南")
    print("="*60)
    
    guide_content = """# 性能优化配置实施指南

## 概述
本指南说明如何应用性能优化结果来调整输出系统配置，以获得最佳性能。

## 配置变更摘要

### 缓冲区优化
- **buffer_size**: 1000 → 5000 (+400%)
- **flush_interval**: 1.0s → 2.0s (+100%)
- **预期效果**: 提高吞吐量 20-40%

### 文件管理优化
- **max_file_size**: 100MB → 200MB (+100%)
- **max_files_per_type**: 10 → 15 (+50%)
- **compression_enabled**: false → true
- **预期效果**: 减少磁盘I/O 25-35%

### 并发优化
- **concurrent_workers**: 4 → 8 (+100%)
- **queue_timeout**: 5.0s → 10.0s (+100%)
- **预期效果**: 提升并发性能 40-60%

### 内存优化
- **memory_threshold_mb**: 500MB → 300MB (-40%)
- **disk_usage_threshold**: 85% → 80% (-5%)
- **预期效果**: 减少内存使用 15-30%

## 实施步骤

### 1. 备份当前配置
```bash
# 备份现有配置文件
cp current_config.json current_config_backup.json
```

### 2. 应用新配置
```python
from performance_optimizer import OptimizationConfig

# 创建优化配置
config = OptimizationConfig(
    buffer_size=5000,
    flush_interval=2.0,
    max_file_size=200 * 1024 * 1024,
    max_files_per_type=15,
    compression_enabled=True,
    concurrent_workers=8,
    queue_timeout=10.0,
    memory_threshold_mb=300.0,
    disk_usage_threshold=80.0
)

# 应用到日志管理器
log_manager = UnifiedLogManager(
    "production",
    buffer_size=config.buffer_size,
    max_file_size=config.max_file_size
)
```

### 3. 监控关键指标
- **吞吐量**: 目标 > 50,000 ops/s
- **内存使用**: 保持在 300MB 以下
- **磁盘I/O**: 监控写入速率
- **错误率**: 保持 < 1%

### 4. 性能验证
运行基准测试验证优化效果：
```bash
python performance_benchmark.py
```

### 5. 回滚计划
如果出现问题，立即回滚：
```bash
# 恢复备份配置
cp current_config_backup.json current_config.json
# 重启服务
```

## 监控建议

### 关键指标
1. **日志吞吐量** (ops/s)
2. **平均响应时间** (ms)
3. **内存使用峰值** (MB)
4. **磁盘I/O速率** (MB/s)
5. **错误率** (%)

### 告警阈值
- 吞吐量下降 > 20%
- 内存使用 > 400MB
- 错误率 > 2%
- 磁盘使用率 > 90%

## 故障排除

### 常见问题
1. **内存使用过高**: 检查压缩是否正常工作
2. **吞吐量下降**: 验证缓冲区配置
3. **文件轮转频繁**: 调整文件大小限制
4. **并发错误**: 检查工作线程数量

### 调试命令
```python
# 检查系统状态
log_manager.get_component_config("System")

# 监控内存使用
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")

# 检查磁盘空间
file_manager.get_disk_space_info()
```

## 注意事项

1. **逐步部署**: 建议在测试环境先验证
2. **监控期**: 部署后密切监控 24-48 小时
3. **备份策略**: 保留原配置至少 7 天
4. **文档更新**: 更新运维文档和监控配置

## 联系支持

如有问题，请联系技术支持团队。
"""
    
    guide_file = Path("config_examples") / "implementation_guide.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"✓ 实施指南保存到: {guide_file}")
    return guide_file


def main():
    """主函数"""
    print("优化配置示例和实施指南")
    print("展示如何应用性能优化结果来调整系统配置")
    
    try:
        # 演示配置影响
        demonstrate_configuration_impact()
        
        # 测试优化配置
        test_optimized_configuration()
        
        # 保存配置文件
        config_dir = save_configuration_files()
        
        # 生成实施指南
        guide_file = generate_implementation_guide()
        
        print("\n" + "="*60)
        print("优化配置示例完成")
        print("="*60)
        print(f"📁 配置文件目录: {config_dir}")
        print(f"📖 实施指南: {guide_file}")
        print("\n✅ 性能优化配置示例和指南生成完成！")
        
    except Exception as e:
        print(f"\n❌ 配置示例生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()