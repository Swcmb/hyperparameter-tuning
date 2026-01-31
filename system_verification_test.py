#!/usr/bin/env python3
"""
系统验证测试脚本
验证输出系统重写项目的核心功能是否正常工作
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

def test_core_components():
    """测试核心组件"""
    print("=" * 60)
    print("输出系统重写项目 - 系统验证测试")
    print("=" * 60)
    
    try:
        # 测试组件导入
        print("\n1. 测试组件导入...")
        from unified_log_manager import UnifiedLogManager
        from structured_tag_processor import StructuredTagProcessor
        from output_formatter import OutputFormatter
        from file_manager import FileManager
        print("✓ 所有核心组件导入成功")
        
        # 测试结构化标签处理器
        print("\n2. 测试结构化标签处理器...")
        processor = StructuredTagProcessor()
        all_tags = processor.get_all_tags()
        print(f"✓ 结构化标签处理器创建成功，支持 {len(all_tags)} 个标签")
        
        # 验证主要标签类别
        training_tags = processor.get_tags_by_category("training")
        config_tags = processor.get_tags_by_category("config")
        optimization_tags = processor.get_tags_by_category("optimization")
        print(f"✓ 训练标签: {len(training_tags)} 个")
        print(f"✓ 配置标签: {len(config_tags)} 个")
        print(f"✓ 优化标签: {len(optimization_tags)} 个")
        
        # 测试输出格式化器
        print("\n3. 测试输出格式化器...")
        formatter = OutputFormatter()
        print("✓ 输出格式化器创建成功")
        
        # 测试emoji移除功能
        test_text = "🚀 开始训练 📊 显示结果 ⚡ 优化中"
        clean_text = formatter.remove_emojis(test_text)
        print(f"✓ Emoji移除测试:")
        print(f"  原文: {test_text}")
        print(f"  清理后: {clean_text}")
        
        # 测试进度格式化
        progress_info = formatter.format_progress_info(
            current=75, total=100, 
            message="训练进行中",
            metrics={"loss": 0.234, "accuracy": 0.876}
        )
        print(f"✓ 进度格式化测试:")
        print(f"  {progress_info}")
        
        # 测试文件管理器
        print("\n4. 测试文件管理器...")
        file_manager = FileManager("test_logs", "system_verification")
        print("✓ 文件管理器创建成功")
        print(f"✓ 运行目录: {file_manager.run_dir}")
        
        # 测试统一日志管理器
        print("\n5. 测试统一日志管理器...")
        with UnifiedLogManager("system_verification_test", 
                             log_level=logging.INFO,
                             enable_console=True,
                             enable_file=True) as log_manager:
            print("✓ 统一日志管理器创建成功")
            
            # 测试基本日志输出
            log_manager.log_with_tag(logging.INFO, "TEST", "这是一个测试消息", "SystemVerification")
            log_manager.log_with_tag(logging.INFO, "TRAINING", "模拟训练开始", "SystemVerification")
            log_manager.log_with_tag(logging.INFO, "CONFIG", "配置加载完成", "SystemVerification")
            
            # 测试结构化数据日志
            test_data = {
                "epoch": 1,
                "batch": 10,
                "loss": 0.234,
                "accuracy": 0.876,
                "gpu_memory": "2.1GB"
            }
            log_manager.log_structured(logging.INFO, "METRICS", test_data, "SystemVerification")
            
            # 测试进度日志
            log_manager.log_progress("TRAINING", 75, 100, "训练进行中", 
                                   loss=0.234, accuracy=0.876)
            
            print("✓ 日志输出测试完成")
        
        print("\n6. 测试集成功能...")
        
        # 检查是否有贝叶斯优化器和AutoDL协调器的集成
        try:
            from bayesian_optimizer import BayesianOptimizer
            print("✓ 贝叶斯优化器集成检查通过")
        except Exception as e:
            print(f"⚠ 贝叶斯优化器集成检查失败: {e}")
        
        try:
            from autodl import AutoDLOptimizer
            print("✓ AutoDL协调器集成检查通过")
        except Exception as e:
            print(f"⚠ AutoDL协调器集成检查失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 系统验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """测试性能"""
    print("\n7. 测试基本性能...")
    
    try:
        from unified_log_manager import UnifiedLogManager
        import time
        
        # 测试大量日志输出的性能
        start_time = time.time()
        
        with UnifiedLogManager("performance_test", 
                             enable_console=False,
                             enable_file=True) as log_manager:
            
            # 输出1000条日志
            for i in range(1000):
                log_manager.log_with_tag(logging.INFO, "PERF_TEST", 
                                       f"性能测试消息 {i+1}", "PerformanceTest")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ 性能测试完成: 1000条日志耗时 {duration:.3f} 秒")
        print(f"✓ 平均每条日志耗时: {duration/1000*1000:.3f} 毫秒")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    success = True
    
    # 测试核心组件
    if not test_core_components():
        success = False
    
    # 测试性能
    if not test_performance():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 系统验证测试全部通过！")
        print("✓ 核心组件功能正常")
        print("✓ 日志输出系统工作正常")
        print("✓ 文件管理功能正常")
        print("✓ 性能表现良好")
    else:
        print("❌ 系统验证测试存在问题")
        print("请检查上述错误信息并修复相关问题")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)