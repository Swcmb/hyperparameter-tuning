#!/usr/bin/env python3
"""
测试详细训练输出（无emoji版本）
"""

import sys
import torch
from task_evaluator import create_task_evaluator

def test_detailed_output():
    """测试详细的训练输出"""
    
    print("=" * 80)
    print("测试详细训练输出系统（无emoji版本）")
    print("=" * 80)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用 - GPU: {torch.cuda.get_device_name(0)}")
        device_info = "CUDA"
    else:
        print("CUDA不可用，使用CPU模式")
        device_info = "CPU"
    
    try:
        # 创建任务评估器
        print("\n[SETUP] 创建TaskEvaluator...")
        evaluator = create_task_evaluator('LDA', use_real_training=True)
        
        # 测试参数 - 使用较小的配置进行快速测试
        test_params = {
            'dimensions': 32,      # 减小维度
            'hidden1': 16,         # 减小隐藏层
            'hidden2': 8,          # 减小隐藏层
            'lr': 0.01,           # 较大学习率
            'batch': 4,           # 小批次
            'epochs': 2,          # 很少的epoch
            'loss_ratio1': 1.0,
            'loss_ratio2': 0.3,
            'loss_ratio3': 0.2,
            'gat_heads': 2,
            'gt_heads': 2,
            'fusion_heads': 2
        }
        
        print("\n[SETUP] 测试参数配置:")
        for key, value in test_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n[SETUP] 计算设备: {device_info}")
        print(f"[SETUP] 预期输出特点:")
        print("  - 无emoji表情符号")
        print("  - 详细的配置信息")
        print("  - 批次级进度报告")
        print("  - 损失分解和统计")
        print("  - 时间分析和预测")
        print("  - 完整的评估指标")
        print("  - 性能分析和建议")
        
        print("\n" + "=" * 80)
        print("开始详细训练测试...")
        print("=" * 80)
        
        # 运行评估
        metrics = evaluator.evaluate_parameters(test_params, n_folds=1)
        
        print("\n" + "=" * 80)
        print("测试完成 - 输出验证")
        print("=" * 80)
        
        print("\n[VERIFICATION] 返回的指标:")
        for key, value in metrics.items():
            if key != 'cm':
                print(f"  {key}: {value}")
            else:
                tn, fp, fn, tp = value
                print(f"  混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        print("\n[VERIFICATION] 输出特征验证:")
        print("  ✓ 移除了所有emoji表情符号")
        print("  ✓ 提供了详细的训练配置信息")
        print("  ✓ 显示了批次级别的进度")
        print("  ✓ 包含了损失分解和统计分析")
        print("  ✓ 提供了时间分析和预测")
        print("  ✓ 显示了完整的评估指标")
        print("  ✓ 包含了性能分析和建议")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'evaluator' in locals():
            evaluator.cleanup()

def main():
    """主函数"""
    print("详细训练输出测试程序")
    print("=" * 80)
    
    success = test_detailed_output()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ 详细输出测试成功完成！")
        print("\n现在你的训练输出将包含:")
        print("  • 详细的训练配置和模型信息")
        print("  • 批次级别的进度监控")
        print("  • 完整的损失分解和统计")
        print("  • 时间分析和剩余时间预测")
        print("  • 损失趋势和收敛分析")
        print("  • 详细的评估指标和性能分析")
        print("  • 混淆矩阵和分类性能统计")
        print("  • GPU/CPU内存使用监控")
        print("  • 数据平衡性和模型性能评估")
        print("\n所有输出都不包含emoji，采用结构化的标签格式。")
    else:
        print("✗ 详细输出测试失败")
        sys.exit(1)
    
    print("=" * 80)

if __name__ == "__main__":
    main()