#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
应用增强训练日志

将详细的训练日志应用到现有的task_evaluator.py中
"""

import os
import shutil
from datetime import datetime


def backup_and_patch_task_evaluator():
    """备份并修补task_evaluator.py以支持增强日志"""
    
    print("🔧 应用增强训练日志...")
    
    # 1. 备份原文件
    if os.path.exists("task_evaluator.py"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"task_evaluator.py.backup_{timestamp}"
        shutil.copy2("task_evaluator.py", backup_path)
        print(f"✓ 已备份 task_evaluator.py -> {backup_path}")
    
    # 2. 读取原文件内容
    with open("task_evaluator.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # 3. 添加增强日志导入
    import_addition = """
# 增强训练日志支持
try:
    from enhanced_training_logger import create_enhanced_logger
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
"""
    
    # 在导入部分添加
    if "from enhanced_training_logger import" not in content:
        # 找到导入部分的结尾
        import_end = content.find("class TaskEvaluator:")
        if import_end != -1:
            content = content[:import_end] + import_addition + "\n\n" + content[import_end:]
            print("✓ 已添加增强日志导入")
    
    # 4. 修改训练循环以支持更详细的输出
    enhanced_training_loop = '''        # 训练循环 - 增强版
        for epoch in range(epochs):
            # 更频繁的进度输出
            if epoch % 5 == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                self.logger.info(f"🚀 训练进度: Epoch {epoch+1}/{epochs} | 已用时: {elapsed:.1f}s")
                
            epoch_loss = 0.0
            batch_count = 0
            epoch_start = time.time()
            
            # 损失统计
            loss_stats = {"total": 0.0, "bce": 0.0, "contrast": 0.0, "adversarial": 0.0}
            
            for i, (labels, inputs) in enumerate(train_loader):
                labels = labels.to(self.device)
                optimizer.zero_grad()
                
                try:
                    # 前向传播
                    output, cla_os, cla_os_a, _, logits, log1 = model(data_o, data_a, inputs)
                    
                    # 计算损失
                    log = torch.squeeze(sigmoid(output))
                    loss1 = loss_fct(log, labels.float())
                    
                    # 对比损失
                    if float(getattr(args, "loss_ratio2", 0.0) or 0.0) > 0.0:
                        if isinstance(cla_os, (list, tuple)):
                            losses = [ce_loss(lg, tg) for lg, tg in zip(cla_os, cla_os_a)]
                            loss2 = torch.stack(losses).mean()
                        else:
                            loss2 = ce_loss(cla_os, cla_os_a)
                    else:
                        loss2 = torch.tensor(0.0, device=self.device)
                    
                    # 节点对抗损失
                    loss3 = node_loss(logits, lbl2.float())
                    
                    # 总损失
                    total_loss = (getattr(args, 'loss_ratio1', 1.0) * loss1 + 
                                 getattr(args, 'loss_ratio2', 0.5) * loss2 + 
                                 getattr(args, 'loss_ratio3', 0.5) * loss3)
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    # 统计损失
                    epoch_loss += total_loss.item()
                    loss_stats["total"] += total_loss.item()
                    loss_stats["bce"] += loss1.item()
                    loss_stats["contrast"] += loss2.item()
                    loss_stats["adversarial"] += loss3.item()
                    batch_count += 1
                    
                    # 详细批次日志（每20个batch）
                    if i % 20 == 0 and i > 0:
                        avg_loss = epoch_loss / batch_count
                        progress = (i + 1) / len(train_loader) * 100
                        self.logger.info(f"    Batch {i+1}/{len(train_loader)} ({progress:.1f}%) | "
                                       f"Loss: {total_loss.item():.4f} | Avg: {avg_loss:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"批次 {i} 训练失败: {e}")
                    continue
            
            # Epoch结束统计
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                epoch_time = time.time() - epoch_start
                
                # 计算各部分损失的平均值
                for key in loss_stats:
                    loss_stats[key] /= batch_count
                
                # 详细的epoch总结
                if epoch % 5 == 0 or epoch == epochs - 1:
                    self.logger.info(f"✅ Epoch {epoch+1}/{epochs} 完成 | 用时: {epoch_time:.1f}s")
                    self.logger.info(f"   总损失: {avg_epoch_loss:.4f} | "
                                   f"BCE: {loss_stats['bce']:.4f} | "
                                   f"对比: {loss_stats['contrast']:.4f} | "
                                   f"对抗: {loss_stats['adversarial']:.4f}")
                    
                    # 估算剩余时间
                    if epoch > 0:
                        avg_time_per_epoch = (time.time() - start_time) / (epoch + 1)
                        remaining_time = avg_time_per_epoch * (epochs - epoch - 1)
                        self.logger.info(f"   预计剩余时间: {remaining_time:.1f}s ({remaining_time/60:.1f}分钟)")
                
                # 清理GPU内存
                torch.cuda.empty_cache()'''
    
    # 5. 替换原有的训练循环
    old_loop_start = content.find("# 训练循环\n        for epoch in range(epochs):")
    if old_loop_start != -1:
        # 找到训练循环的结束位置
        old_loop_end = content.find("# 清理GPU内存", old_loop_start)
        if old_loop_end != -1:
            old_loop_end = content.find("torch.cuda.empty_cache()", old_loop_end) + len("torch.cuda.empty_cache()")
            
            # 添加开始时间记录
            start_time_addition = "        start_time = time.time()  # 记录开始时间\n        "
            
            content = (content[:old_loop_start] + 
                      start_time_addition + 
                      enhanced_training_loop + 
                      content[old_loop_end:])
            print("✓ 已替换训练循环为增强版本")
    
    # 6. 写回文件
    with open("task_evaluator.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ 增强训练日志应用完成！")
    return True


def create_quick_test():
    """创建快速测试脚本"""
    test_code = '''#!/usr/bin/env python3
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
'''
    
    with open("test_enhanced_logging.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    
    print("✓ 已创建快速测试脚本: test_enhanced_logging.py")


def main():
    """主函数"""
    print("增强训练日志应用工具")
    print("=" * 50)
    
    # 检查必要文件
    required_files = ["enhanced_training_logger.py", "task_evaluator.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    # 应用增强日志
    success = backup_and_patch_task_evaluator()
    
    if success:
        print("\n✅ 增强训练日志应用成功！")
        print("\n📝 改进内容:")
        print("  1. 更频繁的训练进度输出（每5个epoch）")
        print("  2. 详细的损失分解显示")
        print("  3. 批次级别的进度监控")
        print("  4. 剩余时间估算")
        print("  5. 更清晰的格式化输出")
        
        print("\n🚀 现在训练时会显示:")
        print("  - 每5个epoch的详细进度")
        print("  - 各种损失的分解值")
        print("  - 每20个batch的进度")
        print("  - 剩余时间估算")
        
        # 创建测试脚本
        create_quick_test()
        
        print("\n🧪 运行测试:")
        print("  python test_enhanced_logging.py")
        
        return True
    else:
        print("\n❌ 应用失败")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)