#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版任务评估器

提供更详细的训练日志输出
"""

import torch
import torch.nn as nn
import numpy as np
import time
from enhanced_training_logger import create_enhanced_logger


def enhanced_simplified_training(self, model, optimizer, data_o, data_a, train_loader, test_loader, args):
    """
    增强版简化训练实现，提供详细的训练日志
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
    
    # 创建增强日志记录器
    enhanced_logger = create_enhanced_logger("DetailedTraining")
    
    # 设置损失函数
    loss_fct = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    ce_loss = nn.CrossEntropyLoss()
    node_loss = nn.BCEWithLogitsLoss()
    
    # 强制设置epoch为5
    epochs = 5
    
    # 准备模型信息
    model_info = {
        "总参数数": sum(p.numel() for p in model.parameters()),
        "可训练参数": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "学习率": optimizer.param_groups[0]['lr'],
        "批大小": getattr(args, 'batch', 25),
        "损失权重": f"α={getattr(args, 'loss_ratio1', 1.0)}, β={getattr(args, 'loss_ratio2', 0.5)}, γ={getattr(args, 'loss_ratio3', 0.5)}"
    }
    
    # 开始训练日志
    enhanced_logger.start_training(epochs, model_info)
    
    # 为节点级别的对抗损失创建标签
    n_nodes = int(data_o.x.size(0))
    lbl_1 = torch.ones(1, n_nodes, device=self.device)
    lbl_2 = torch.zeros(1, n_nodes, device=self.device)
    lbl2 = torch.cat((lbl_1, lbl_2), 1)
    
    model.train()
    
    # 训练循环
    for epoch in range(epochs):
        enhanced_logger.start_epoch(epoch, epochs)
        
        epoch_losses = {
            "total_loss": 0.0,
            "loss1_bce": 0.0,
            "loss2_contrast": 0.0,
            "loss3_adversarial": 0.0
        }
        batch_count = 0
        
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
                
                # 累积损失
                epoch_losses["total_loss"] += total_loss.item()
                epoch_losses["loss1_bce"] += loss1.item()
                epoch_losses["loss2_contrast"] += loss2.item()
                epoch_losses["loss3_adversarial"] += loss3.item()
                batch_count += 1
                
                # 详细批次日志（可选，用于调试）
                if i % 20 == 0:  # 每20个batch输出一次
                    batch_losses = {
                        "total": total_loss.item(),
                        "bce": loss1.item(),
                        "contrast": loss2.item(),
                        "adv": loss3.item()
                    }
                    enhanced_logger.log_batch_progress(
                        epoch, i, len(train_loader), batch_losses, 
                        optimizer.param_groups[0]['lr']
                    )
                
            except Exception as e:
                self.logger.error(f"批次 {i} 训练失败: {e}")
                continue
        
        # 计算平均损失
        if batch_count > 0:
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
        
        # 每5个epoch进行一次验证
        metrics = {}
        if (epoch + 1) % 5 == 0:
            try:
                # 简单验证
                model.eval()
                with torch.no_grad():
                    val_outputs = []
                    val_labels = []
                    
                    for labels, inputs in test_loader:
                        labels = labels.to(self.device)
                        output, _, _, _, _, _ = model(data_o, data_a, inputs)
                        
                        probs = torch.sigmoid(output).cpu().numpy()
                        val_outputs.extend(probs.flatten())
                        val_labels.extend(labels.cpu().numpy())
                    
                    if len(val_outputs) > 0:
                        val_outputs = np.array(val_outputs)
                        val_labels = np.array(val_labels)
                        
                        # 计算指标
                        auc = roc_auc_score(val_labels, val_outputs)
                        auprc = average_precision_score(val_labels, val_outputs)
                        
                        # 使用0.5作为阈值计算其他指标
                        pred_labels = (val_outputs > 0.5).astype(int)
                        f1 = f1_score(val_labels, pred_labels)
                        precision = precision_score(val_labels, pred_labels, zero_division=0)
                        recall = recall_score(val_labels, pred_labels, zero_division=0)
                        
                        metrics = {
                            "AUC": auc,
                            "AUPRC": auprc,
                            "F1": f1,
                            "Precision": precision,
                            "Recall": recall
                        }
                
                model.train()
                
            except Exception as e:
                self.logger.warning(f"验证失败: {e}")
        
        # 结束epoch日志
        enhanced_logger.end_epoch(epoch, epochs, epoch_losses, metrics)
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    # 最终评估
    model.eval()
    final_metrics = {}
    
    try:
        with torch.no_grad():
            test_outputs = []
            test_labels = []
            
            for labels, inputs in test_loader:
                labels = labels.to(self.device)
                output, _, _, _, _, _ = model(data_o, data_a, inputs)
                
                probs = torch.sigmoid(output).cpu().numpy()
                test_outputs.extend(probs.flatten())
                test_labels.extend(labels.cpu().numpy())
            
            if len(test_outputs) > 0:
                test_outputs = np.array(test_outputs)
                test_labels = np.array(test_labels)
                
                # 计算最终指标
                final_metrics = {
                    "auroc": float(roc_auc_score(test_labels, test_outputs)),
                    "auprc": float(average_precision_score(test_labels, test_outputs)),
                    "f1": float(f1_score(test_labels, (test_outputs > 0.5).astype(int))),
                    "precision": float(precision_score(test_labels, (test_outputs > 0.5).astype(int), zero_division=0)),
                    "recall": float(recall_score(test_labels, (test_outputs > 0.5).astype(int), zero_division=0)),
                    "loss": epoch_losses["total_loss"]
                }
    
    except Exception as e:
        self.logger.error(f"最终评估失败: {e}")
        # 返回默认值
        final_metrics = {
            "auroc": 0.5,
            "auprc": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "loss": float('inf')
        }
    
    # 结束训练日志
    enhanced_logger.end_training(epochs, final_metrics)
    
    return final_metrics


# 使用说明
def apply_enhanced_logging_to_task_evaluator():
    """
    将增强日志应用到TaskEvaluator
    
    使用方法：
    1. 在task_evaluator.py中导入这个函数
    2. 替换_simplified_training方法
    """
    print("要应用增强日志，请在task_evaluator.py中:")
    print("1. 导入: from task_evaluator_enhanced import enhanced_simplified_training")
    print("2. 在RealTaskEvaluator类中替换_simplified_training方法")
    print("3. 或者直接调用enhanced_simplified_training函数")


if __name__ == "__main__":
    # 运行示例
    enhanced_logger = create_enhanced_logger("TestEnhanced")
    print("增强训练日志记录器测试:")
    print("运行 python enhanced_training_logger.py 查看完整示例")
    
    apply_enhanced_logging_to_task_evaluator()