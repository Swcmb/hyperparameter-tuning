#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoDL贝叶斯优化系统快速开始示例

本脚本演示如何使用autodl.py进行超参数优化
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def create_sample_config():
    """创建示例配置文件"""
    config = {
        "task_type": "LDA",
        "max_iterations": 10,  # 快速示例，只运行10次迭代
        "max_time_hours": 1,   # 最多运行1小时
        "random_seed": 42,
        
        "acquisition_function": "EI",
        "acquisition_params": {"xi": 0.01},
        
        "objectives": ["AUROC"],
        
        "cv_folds": 3,  # 减少折数以加快速度
        
        "checkpoint_dir": "demo_checkpoints",
        "save_frequency": 2,
        
        "output_dir": "demo_results",
        "log_dir": "demo_logs",
        
        "generate_report": True,
        "generate_html": True,
        "generate_charts": True
    }
    
    config_path = "demo_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 示例配置文件已创建: {config_path}")
    return config_path

def run_basic_optimization():
    """运行基本的贝叶斯优化示例"""
    print("=" * 60)
    print("AutoDL贝叶斯优化系统 - 快速开始示例")
    print("=" * 60)
    
    # 检查autodl.py是否存在
    if not os.path.exists("autodl.py"):
        print("❌ 错误: 找不到autodl.py文件")
        print("请确保在hyperparameter-tuning目录中运行此脚本")
        return False
    
    # 创建示例配置
    config_path = create_sample_config()
    
    print("\n1. 基本优化示例（单目标）")
    print("-" * 40)
    
    try:
        # 运行基本优化
        cmd = [
            sys.executable, "autodl.py",
            "--config", config_path,
            "--task_type", "LDA",
            "--max_iterations", "5"  # 快速演示
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("开始运行优化...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ 基本优化完成")
            print("标准输出:")
            print(result.stdout[-1000:])  # 显示最后1000个字符
        else:
            print("❌ 优化失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ 优化超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False
    
    return True

def run_multi_objective_example():
    """运行多目标优化示例"""
    print("\n2. 多目标优化示例")
    print("-" * 40)
    
    try:
        # 运行多目标优化
        cmd = [
            sys.executable, "autodl.py",
            "--task_type", "LDA",
            "--max_iterations", "5",
            "--objectives", "AUROC", "AUPRC", "F1",
            "--objective_weights", "0.5", "0.3", "0.2",
            "--output_dir", "demo_multi_results"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("开始运行多目标优化...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ 多目标优化完成")
            print("标准输出:")
            print(result.stdout[-1000:])
        else:
            print("❌ 多目标优化失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ 多目标优化超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False
    
    return True

def show_results():
    """显示结果文件"""
    print("\n3. 查看生成的结果")
    print("-" * 40)
    
    result_dirs = ["demo_results", "demo_multi_results"]
    
    for result_dir in result_dirs:
        if os.path.exists(result_dir):
            print(f"\n{result_dir} 目录内容:")
            for item in os.listdir(result_dir):
                item_path = os.path.join(result_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    print(f"  📄 {item} ({size} bytes)")
                elif os.path.isdir(item_path):
                    file_count = len(os.listdir(item_path))
                    print(f"  📁 {item}/ ({file_count} files)")
        else:
            print(f"❌ 结果目录不存在: {result_dir}")

def cleanup_demo_files():
    """清理演示文件"""
    print("\n4. 清理演示文件")
    print("-" * 40)
    
    demo_files = [
        "demo_config.json",
        "demo_checkpoints",
        "demo_results", 
        "demo_multi_results",
        "demo_logs"
    ]
    
    for item in demo_files:
        if os.path.exists(item):
            if os.path.isfile(item):
                os.remove(item)
                print(f"✓ 删除文件: {item}")
            elif os.path.isdir(item):
                import shutil
                shutil.rmtree(item)
                print(f"✓ 删除目录: {item}")

def show_usage_examples():
    """显示使用示例"""
    print("\n5. 更多使用示例")
    print("-" * 40)
    
    examples = [
        {
            "描述": "基本单目标优化",
            "命令": "python autodl.py --task_type LDA --max_iterations 30"
        },
        {
            "描述": "使用配置文件",
            "命令": "python autodl.py --config my_config.json"
        },
        {
            "描述": "多目标优化",
            "命令": "python autodl.py --objectives AUROC AUPRC F1 --objective_weights 0.5 0.3 0.2"
        },
        {
            "描述": "恢复之前的优化",
            "命令": "python autodl.py --resume --checkpoint_name iteration_20"
        },
        {
            "描述": "使用UCB采集函数",
            "命令": 'python autodl.py --acquisition_function UCB --acquisition_params \'{"beta": 2.0}\''
        },
        {
            "描述": "长时间运行",
            "命令": "python autodl.py --max_iterations 100 --max_time_hours 48"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}: {example['描述']}")
        print(f"命令: {example['命令']}")

def main():
    """主函数"""
    print("欢迎使用AutoDL贝叶斯优化系统快速开始指南!")
    print("\n选择要运行的示例:")
    print("1. 运行基本优化示例")
    print("2. 运行多目标优化示例") 
    print("3. 查看使用示例")
    print("4. 清理演示文件")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == "1":
                success = run_basic_optimization()
                if success:
                    show_results()
                break
                
            elif choice == "2":
                success = run_multi_objective_example()
                if success:
                    show_results()
                break
                
            elif choice == "3":
                show_usage_examples()
                break
                
            elif choice == "4":
                confirm = input("确定要清理所有演示文件吗? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    cleanup_demo_files()
                break
                
            elif choice == "5":
                print("再见!")
                break
                
            else:
                print("无效选择，请输入1-5")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，再见!")
            break
        except Exception as e:
            print(f"出错: {e}")
            break

if __name__ == "__main__":
    main()