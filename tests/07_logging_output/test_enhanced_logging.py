#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速测试增强日志功能
"""

def test_enhanced_logging():
    """测试增强日志功能"""
    print("测试增强训练日志...")
    
    try:
        from enhanced_training_logger import create_enhanced_logger
        
        # 创建测试日志记录器
        logger = create_enhanced_logger("TestLogger")
        
        # 模拟训练过程
        logger.start_training(10, {"model": "Test", "lr": 0.001})
        
        for epoch in range(3):
            logger.start_epoch(epoch, 10)
            
            # 模拟一些损失
            import time
            time.sleep(0.1)
            
            losses = {
                "total_loss": 1.0 - epoch * 0.1,
                "bce_loss": 0.6 - epoch * 0.05,
                "contrast_loss": 0.3 - epoch * 0.03,
                "adv_loss": 0.1 - epoch * 0.02
            }
            
            metrics = {"auc": 0.6 + epoch * 0.1} if epoch % 2 == 0 else None
            
            logger.end_epoch(epoch, 10, losses, metrics)
        
        logger.end_training(10, {"final_auc": 0.85})
        
        print("✅ 增强日志测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 增强日志测试失败: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_logging()
