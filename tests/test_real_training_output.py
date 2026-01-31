#!/usr/bin/env python3
"""
测试真实训练的增强输出
"""

import sys
import torch
from task_evaluator import RealTaskEvaluator

def test_real_training_output():
    """测试真实训练的增强输出"""
    
    print("🧪 测试真实训练的增强输出功能")
    print("=" * 80)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，将使用CPU模式进行测试")
        print("注意：真实训练通常需要CUDA以获得最佳性能")
        print()
    
    try:
        # 创建真实任务评估器
        print("📋 创建RealTaskEvaluator...")
        evaluator = RealTaskEvaluator(task_type="LDA")
        
        # 测试参数 - 使用较小的配置进行快速测试
        test_params = {
            'dimensions': 64,
            'hidden1': 32, 
            'hidden2': 16,
            'lr': 0.001,
            'batch': 8,
            'epochs': 3,  # 使用很少的epoch进行快速测试
            'loss_ratio1': 1.0,
            'loss_ratio2': 0.5,
            'loss_ratio3': 0.5,
            'gat_heads': 2,
            'gt_heads': 2,
            'fusion_heads': 2
        }
        
        print("🎯 测试参数:")
        for key, value in test_params.items():
            print(f"   {key}: {value}")
        print()
        
        print("🚀 开始真实训练测试（1折，快速模式）...")
        print("=" * 80)
        
        # 运行评估
        metrics = evaluator.evaluate_parameters(test_params, n_folds=1)
        
        print("=" * 80)
        print("✅ 真实训练测试完成！")
        print(f"📊 最终结果:")
        print(f"   AUROC: {metrics['AUROC']:.4f}")
        print(f"   AUPRC: {metrics['AUPRC']:.4f}")
        print(f"   F1: {metrics['F1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        if 'error' in metrics:
            print(f"⚠️ 错误信息: {metrics['error']}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'evaluator' in locals():
            evaluator.cleanup()
    
    return True

if __name__ == "__main__":
    success = test_real_training_output()
    if success:
        print("🎉 真实训练输出测试成功！")
        print()
        print("💡 现在当你运行真实的超参数优化时，你会看到：")
        print("   ✅ 详细的训练配置信息")
        print("   ✅ 每5个epoch的进度更新")
        print("   ✅ 批次级别的损失监控")
        print("   ✅ 损失分解（BCE、对比、对抗）")
        print("   ✅ 剩余时间估算")
        print("   ✅ 详细的最终评估结果")
        print("   ✅ 美观的格式化输出")
    else:
        print("❌ 真实训练输出测试失败")
        sys.exit(1)