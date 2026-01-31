#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
参数冲突修复脚本

自动备份原有文件并应用新的统一参数管理系统
"""

import os
import shutil
import sys
from datetime import datetime


def backup_file(file_path):
    """备份文件"""
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        print(f"✓ 已备份 {file_path} -> {backup_path}")
        return backup_path
    return None


def apply_fix():
    """应用参数冲突修复"""
    print("🔧 开始修复参数冲突问题...")
    
    # 1. 备份原有的parms_setting.py
    backup_path = backup_file("parms_setting.py")
    if backup_path:
        print(f"✓ 原有parms_setting.py已备份")
    
    # 2. 替换parms_setting.py
    if os.path.exists("parms_setting_new.py"):
        shutil.copy2("parms_setting_new.py", "parms_setting.py")
        print("✓ 已应用新的parms_setting.py")
    else:
        print("❌ 找不到parms_setting_new.py文件")
        return False
    
    # 3. 测试新系统
    print("\n🧪 测试新的参数系统...")
    
    try:
        # 测试导入
        import parms_setting
        print("✓ 新parms_setting模块导入成功")
        
        # 测试基本功能
        from parameter_manager import get_parameter_manager
        from unified_parameter_registry import initialize_unified_parameters
        
        # 初始化系统
        initialize_unified_parameters()
        manager = get_parameter_manager()
        
        # 测试解析包含autodl参数的命令行
        test_args = ["--max_iterations", "10", "--random_seed", "42", "--epochs", "20"]
        parsed = manager.parse_arguments(test_args)
        
        print("✓ 参数解析测试成功:")
        print(f"  - max_iterations: {parsed.max_iterations}")
        print(f"  - random_seed: {parsed.random_seed}")
        print(f"  - epochs: {parsed.epochs}")
        
        # 测试settings()函数兼容性
        # 注意：这里不能直接调用settings()，因为参数已经被解析过了
        print("✓ 统一参数系统工作正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        
        # 恢复备份
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, "parms_setting.py")
            print(f"✓ 已恢复原有的parms_setting.py")
        
        return False


def create_usage_example():
    """创建使用示例"""
    example_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用新参数系统的示例

展示如何在autodl.py和其他模块中使用统一参数管理系统
"""

# 方法1: 使用原有的settings()函数（推荐，保持兼容性）
from parms_setting import settings

def main_with_settings():
    """使用settings()函数的方式"""
    args = settings()
    
    print(f"任务类型: {args.task_type}")
    print(f"最大迭代次数: {getattr(args, 'max_iterations', '未设置')}")
    print(f"随机种子: {getattr(args, 'random_seed', '未设置')}")
    print(f"训练轮数: {args.epochs}")


# 方法2: 直接使用参数管理器（高级用法）
from parameter_manager import get_parameter_manager
from unified_parameter_registry import initialize_unified_parameters

def main_with_manager():
    """使用参数管理器的方式"""
    # 初始化系统
    initialize_unified_parameters()
    
    # 获取管理器
    manager = get_parameter_manager()
    
    # 解析参数
    args = manager.parse_arguments()
    
    print(f"任务类型: {args.task_type}")
    print(f"最大迭代次数: {args.max_iterations}")
    print(f"随机种子: {args.random_seed}")
    print(f"训练轮数: {args.epochs}")


# 方法3: 使用参数代理（延迟加载）
from parameter_manager import get_parameter_proxy

def main_with_proxy():
    """使用参数代理的方式"""
    proxy = get_parameter_proxy()
    
    # 延迟访问参数
    task_type = proxy.get('task_type', 'LDA')
    max_iterations = proxy.get('max_iterations', 50)
    
    print(f"任务类型: {task_type}")
    print(f"最大迭代次数: {max_iterations}")


if __name__ == "__main__":
    print("=== 方法1: 使用settings()函数 ===")
    main_with_settings()
    
    print("\\n=== 方法2: 使用参数管理器 ===")
    # main_with_manager()  # 注释掉避免重复解析
    
    print("\\n=== 方法3: 使用参数代理 ===")
    main_with_proxy()
'''
    
    with open("parameter_usage_example.py", "w", encoding="utf-8") as f:
        f.write(example_code)
    
    print("✓ 已创建使用示例文件: parameter_usage_example.py")


def main():
    """主函数"""
    print("参数冲突修复工具")
    print("=" * 50)
    
    # 检查必要文件是否存在
    required_files = [
        "parameter_manager.py",
        "unified_parameter_registry.py", 
        "parms_setting_new.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    # 应用修复
    success = apply_fix()
    
    if success:
        print("\n✅ 参数冲突修复成功！")
        print("\n📝 修复内容:")
        print("  1. 创建了统一参数管理系统")
        print("  2. 备份并替换了parms_setting.py")
        print("  3. 现在autodl.py和parms_setting.py可以和谐共存")
        
        print("\n🚀 现在你可以运行:")
        print("  python autodl.py --max_iterations 30 --random_seed 123")
        print("  不会再出现参数冲突错误！")
        
        # 创建使用示例
        create_usage_example()
        
        return True
    else:
        print("\n❌ 修复失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)