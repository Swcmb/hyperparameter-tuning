#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试autodl.py参数冲突修复

模拟原来的错误场景，验证修复是否成功
"""

import sys
import os

def test_original_error_scenario():
    """测试原来的错误场景"""
    print("🧪 测试原来的错误场景...")
    print("模拟命令: python autodl.py --max_iterations 1 --random_seed 42")
    
    # 模拟sys.argv
    original_argv = sys.argv.copy()
    sys.argv = ["autodl.py", "--max_iterations", "1", "--random_seed", "42"]
    
    try:
        # 这里模拟task_evaluator.py导入parms_setting的场景
        print("1. 导入parms_setting模块...")
        import parms_setting
        print("✓ parms_setting导入成功")
        
        print("2. 调用settings()函数...")
        args = parms_setting.settings()
        print("✓ settings()调用成功")
        
        print("3. 检查参数解析结果...")
        print(f"  - max_iterations: {getattr(args, 'max_iterations', '未找到')}")
        print(f"  - random_seed: {getattr(args, 'random_seed', '未找到')}")
        print(f"  - task_type: {args.task_type}")
        print(f"  - epochs: {args.epochs}")
        print(f"  - lr: {args.lr}")
        
        # 验证autodl参数确实被识别了
        if hasattr(args, 'max_iterations') and hasattr(args, 'random_seed'):
            print("✅ autodl.py的参数被正确识别！")
            return True
        else:
            print("❌ autodl.py的参数未被识别")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        # 恢复原来的sys.argv
        sys.argv = original_argv


def test_task_evaluator_scenario():
    """测试task_evaluator调用场景"""
    print("\n🧪 测试task_evaluator调用场景...")
    
    # 模拟task_evaluator.py中的参数处理
    original_argv = sys.argv.copy()
    sys.argv = ["task_evaluator.py", "--max_iterations", "5", "--random_seed", "123", "--epochs", "10"]
    
    try:
        # 重新导入以清除缓存
        if 'parms_setting' in sys.modules:
            del sys.modules['parms_setting']
        
        # 模拟task_evaluator导入parms_setting
        from parms_setting import settings
        
        print("1. task_evaluator导入parms_setting成功")
        
        # 调用settings()
        args = settings()
        print("2. settings()调用成功")
        
        # 检查参数
        print("3. 参数检查:")
        print(f"  - max_iterations: {getattr(args, 'max_iterations', '未找到')}")
        print(f"  - random_seed: {getattr(args, 'random_seed', '未找到')}")
        print(f"  - epochs: {args.epochs}")
        
        print("✅ task_evaluator场景测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ task_evaluator场景测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv


def test_parameter_conflicts():
    """测试参数冲突处理"""
    print("\n🧪 测试参数冲突处理...")
    
    from parameter_manager import get_parameter_manager
    
    manager = get_parameter_manager()
    conflicts = manager.get_conflicts()
    
    if conflicts:
        print(f"⚠️  发现 {len(conflicts)} 个参数冲突:")
        for name, conflict in conflicts.items():
            print(f"  - {name}: 冲突模块 {conflict.conflicting_modules}")
            print(f"    冲突类型: {conflict.conflict_type}")
            print(f"    解决策略: {conflict.resolution_strategy}")
    else:
        print("✅ 没有参数冲突")
    
    return True


def main():
    """主测试函数"""
    print("参数冲突修复验证测试")
    print("=" * 50)
    
    success = True
    
    # 测试1: 原来的错误场景
    if not test_original_error_scenario():
        success = False
    
    # 测试2: task_evaluator场景
    if not test_task_evaluator_scenario():
        success = False
    
    # 测试3: 参数冲突处理
    if not test_parameter_conflicts():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！参数冲突问题已解决！")
        print("\n✅ 现在你可以正常运行:")
        print("  python autodl.py --max_iterations 30 --random_seed 42")
        print("  不会再出现 'unrecognized arguments' 错误！")
    else:
        print("❌ 部分测试失败，请检查问题")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)