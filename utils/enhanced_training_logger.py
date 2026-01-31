#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强训练日志输出

提供更详细的训练进度和损失信息
"""

import logging
import time
from typing import Dict, Any


class EnhancedTrainingLogger:
    """增强的训练日志记录器"""
    
    def __init__(self, logger_name: str = "EnhancedTraining"):
        self.logger = logging.getLogger(logger_name)
        self.start_time = None
        self.epoch_start_time = None
        self.best_loss = float('inf')
        self.loss_history = []
        
    def start_training(self, total_epochs: int, model_info: Dict[str, Any] = None):
        """开始训练时的日志"""
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("🚀 开始训练")
        self.logger.info("=" * 60)
        self.logger.info(f"总训练轮数: {total_epochs}")
        
        if model_info:
            self.logger.info("模型配置:")
            for key, value in model_info.items():
                self.logger.info(f"  - {key}: {value}")
        
        self.logger.info("-" * 60)
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """开始每个epoch时的日志"""
        self.epoch_start_time = time.time()
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info(f"📈 Epoch {epoch+1}/{total_epochs} | 已用时: {elapsed_time:.1f}s")
    
    def log_batch_progress(self, epoch: int, batch_idx: int, total_batches: int, 
                          losses: Dict[str, float], lr: float = None):
        """记录批次进度（可选，用于详细调试）"""
        if batch_idx % 10 == 0:  # 每10个batch输出一次
            progress = (batch_idx + 1) / total_batches * 100
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            
            log_msg = f"  Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) | {loss_str}"
            if lr:
                log_msg += f" | LR: {lr:.6f}"
            
            self.logger.info(log_msg)
    
    def end_epoch(self, epoch: int, total_epochs: int, losses: Dict[str, float], 
                  metrics: Dict[str, float] = None):
        """结束每个epoch时的日志"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # 计算剩余时间估计
        avg_epoch_time = total_time / (epoch + 1)
        remaining_epochs = total_epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs
        
        # 主要损失
        main_loss = losses.get('total_loss', losses.get('loss', 0.0))
        self.loss_history.append(main_loss)
        
        # 检查是否是最佳损失
        is_best = main_loss < self.best_loss
        if is_best:
            self.best_loss = main_loss
            best_indicator = " 🌟 NEW BEST!"
        else:
            best_indicator = ""
        
        # 输出epoch总结
        self.logger.info(f"✅ Epoch {epoch+1}/{total_epochs} 完成 | 用时: {epoch_time:.1f}s | ETA: {eta:.1f}s{best_indicator}")
        
        # 输出损失信息
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        self.logger.info(f"   损失: {loss_str}")
        
        # 输出指标信息
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"   指标: {metrics_str}")
        
        # 损失趋势分析（每5个epoch分析一次）
        if (epoch + 1) % 5 == 0 and len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            trend = "📈 上升" if recent_losses[-1] > recent_losses[0] else "📉 下降"
            avg_recent = sum(recent_losses) / len(recent_losses)
            self.logger.info(f"   趋势: {trend} | 近5轮平均: {avg_recent:.4f}")
        
        self.logger.info("-" * 60)
    
    def end_training(self, total_epochs: int, final_metrics: Dict[str, float] = None):
        """结束训练时的日志"""
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_epoch_time = total_time / total_epochs
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 训练完成")
        self.logger.info("=" * 60)
        self.logger.info(f"总用时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
        self.logger.info(f"平均每轮: {avg_epoch_time:.1f}s")
        self.logger.info(f"最佳损失: {self.best_loss:.4f}")
        
        if final_metrics:
            self.logger.info("最终指标:")
            for key, value in final_metrics.items():
                self.logger.info(f"  - {key}: {value:.4f}")
        
        self.logger.info("=" * 60)


def create_enhanced_logger(name: str = "EnhancedTraining", level: int = logging.INFO) -> EnhancedTrainingLogger:
    """创建增强训练日志记录器"""
    # 设置日志格式
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return EnhancedTrainingLogger(name)


# 使用示例
if __name__ == "__main__":
    # 创建增强日志记录器
    enhanced_logger = create_enhanced_logger("TestTraining")
    
    # 模拟训练过程
    total_epochs = 10
    enhanced_logger.start_training(total_epochs, {
        "model": "MoCo",
        "batch_size": 32,
        "learning_rate": 0.001
    })
    
    for epoch in range(total_epochs):
        enhanced_logger.start_epoch(epoch, total_epochs)
        
        # 模拟训练过程
        import time
        time.sleep(0.1)  # 模拟训练时间
        
        # 模拟损失
        import random
        losses = {
            "total_loss": 1.0 - epoch * 0.08 + random.uniform(-0.05, 0.05),
            "loss1": 0.6 - epoch * 0.04 + random.uniform(-0.02, 0.02),
            "loss2": 0.3 - epoch * 0.03 + random.uniform(-0.02, 0.02),
            "loss3": 0.1 - epoch * 0.01 + random.uniform(-0.01, 0.01)
        }
        
        metrics = {
            "accuracy": 0.5 + epoch * 0.04 + random.uniform(-0.02, 0.02),
            "auc": 0.6 + epoch * 0.03 + random.uniform(-0.01, 0.01)
        }
        
        enhanced_logger.end_epoch(epoch, total_epochs, losses, metrics)
    
    enhanced_logger.end_training(total_epochs, {
        "final_accuracy": 0.85,
        "final_auc": 0.92
    })