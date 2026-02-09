#!/bin/bash

# 基于optimization_report_20260205_131133的最佳模型参数
# 最佳迭代: 1
# 目标函数值: 0.9691
# AUROC: 0.9691, AUPRC: 0.9591, F1: 0.9385

python train.py --task LMI --dimensions 496 --hidden1 256 --hidden2 64 --decoder1 461.1803616634573 --lr 0.00001 --dropout 0.5 --weight_decay 0.000001 --alpha 2.0 --beta 0.1 --gamma 2.0 --moco_momentum 0.9999 --moco_t 0.01 --moco_tau1 1.0 --moco_tau2 1.0 --gat_heads 2 --gt_heads 2 --fusion_heads 2 --batch 25 --moco_K 2048 --fusion_strategy co_attention --feature_type one_hot --moco_type basic --enable_view_0 true