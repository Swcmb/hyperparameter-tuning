#!/usr/bin/env python3
"""
测试检查点文件内容
"""

import os
import sys
sys.path.insert(0, '.')

from state_manager import StateManager

def test_checkpoint_files():
    checkpoint_dir = "checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
    
    state_manager = StateManager()
    
    for checkpoint_file in sorted(checkpoint_files)[-5:]:  # 测试最后5个文件
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        print(f"\n测试文件: {checkpoint_file}")
        
        try:
            state_data = state_manager.load_state(checkpoint_path)
            print(f"  键: {list(state_data.keys())}")
            
            if 'optimization_history' in state_data:
                history_data = state_data['optimization_history']
                if isinstance(history_data, dict) and 'results' in history_data:
                    print(f"  优化结果数量: {len(history_data['results'])}")
                    print(f"  ✅ 可用于生成报告")
                else:
                    print(f"  ❌ 优化历史格式不正确")
            else:
                print(f"  ❌ 没有优化历史数据")
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")

if __name__ == "__main__":
    test_checkpoint_files()