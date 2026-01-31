#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试增强版TaskEvaluator的输出
"""

import sys
import logging

def test_enhanced_output():
    """测试增强版输出"""
    print("测试增强版TaskEvaluator输出...")
    
    try:
        # 设置日志级别以查看所有输出
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        from task_evaluator import TaskEvaluator
        
        # 创建评估器（测试模式）
        evaluator = TaskEvaluator(task_type="LDA", force_cuda=False)
        
        # 测试参数
        test_params = {
            'dimensions': 128,
            'hidden1': 64,
            'hidden2': 32,
            'lr': 0.001,
            'dropout': 0.1,
            'weight_decay': 0.0005,
            'batch': 16,
            'epochs': 10,  # 使用较少的epoch进行快速测试
            'gat_heads': 4,
            'gt_heads': 4,
            'fusion_heads': 4,
            'loss_ratio1': 1.0,
            'loss_ratio2': 0.5,
            'loss_ratio3': 0.5,
            'fusion_strategy': 'self_attention',
            'feature_type': 'normal',
            'moco_type': 'basic'
        }
        
        print("\n" + "="*80)
        print("🧪 开始测试增强版训练输出")
        print("="*80)
        
        # 测试评估功能（使用1折进行快速测试）
        metrics = evaluator.evaluate_parameters(test_params, n_folds=1)
        
        print("\n" + "="*80)
        print("✅ 测试完成")
        print("="*80)
        print(f"最终指标: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, F1={metrics['F1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'evaluator' in locals():
            evaluator.cleanup()


def show_expected_output():
    """显示预期的输出格式"""
    print("\n" + "="*80)
    print("📋 现在你应该看到的训练输出格式:")
    print("="*80)
    
    expected_output = """
🚀 开始完整训练实现 | 设备: cuda
🎯 开始完整模型训练 | Epochs: 50 | 设备: cuda
📊 训练配置 | 批大小: 25 | 学习率: 0.000500
⚖️  损失权重 | BCE: 1.0 | 对比: 0.5 | 对抗: 0.5
--------------------------------------------------------------------------------
📈 训练进度: Epoch 1/50 | 已用时: 0.0s
    📊 Batch 20/100 (20.0%) | 当前损失: 0.8234 | 平均损失: 0.8456
    📊 Batch 40/100 (40.0%) | 当前损失: 0.7891 | 平均损失: 0.8123
✅ Epoch 1/50 完成 | 用时: 3.2s
   📉 总损失: 0.7654 | BCE: 0.4321 | 对比: 0.2345 | 对抗: 0.0988
   ⏱️  预计剩余时间: 156.8s (2.6分钟)
------------------------------------------------------------
📈 训练进度: Epoch 5/50 | 已用时: 16.0s
✅ Epoch 5/50 完成 | 用时: 3.1s
   📉 总损失: 0.6543 | BCE: 0.3876 | 对比: 0.1987 | 对抗: 0.0680
   ⏱️  预计剩余时间: 139.5s (2.3分钟)
------------------------------------------------------------
================================================================================
🎉 训练完成 | 总用时: 158.4s (2.6分钟)
📊 平均每轮: 3.2s
🔍 开始最终评估...
================================================================================
🎯 最终评估结果:
   📈 AUROC: 0.8234
   📊 AUPRC: 0.7891
   🎯 F1-Score: 0.7456
   ✅ Precision: 0.7654
   🔍 Recall: 0.7234
   📉 Final Loss: 0.4567
   🔢 混淆矩阵: TN=1234, FP=123, FN=234, TP=2345
================================================================================
    """
    
    print(expected_output)
    
    print("\n🎉 主要改进:")
    print("  ✅ 不再显示'简化训练实现'的误导性消息")
    print("  ✅ 显示'完整训练实现'，更准确地反映实际情况")
    print("  ✅ 详细的训练配置信息（批大小、学习率、损失权重）")
    print("  ✅ 每5个epoch的详细进度和损失分解")
    print("  ✅ 批次级别的进度监控")
    print("  ✅ 剩余时间估算")
    print("  ✅ 美观的格式化输出和表情符号")
    print("  ✅ 详细的最终评估结果")


if __name__ == "__main__":
    print("增强版TaskEvaluator输出测试")
    print("="*80)
    
    # 显示预期输出格式
    show_expected_output()
    
    # 询问是否运行实际测试
    print("\n" + "="*80)
    response = input("是否运行实际测试？(y/n): ").lower().strip()
    
    if response in ['y', 'yes', '是']:
        success = test_enhanced_output()
        if success:
            print("\n🎉 测试成功！现在你的训练会显示详细的进度信息。")
        else:
            print("\n❌ 测试失败，请检查错误信息。")
    else:
        print("\n✅ 跳过实际测试。修改已应用，下次运行训练时会看到新的输出格式。")