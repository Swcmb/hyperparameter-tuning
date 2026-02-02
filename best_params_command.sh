#!/bin/bash
# 第5次迭代最佳参数的命令行
# AUROC: 0.956639

python autodl.py --task_type LDA --dimensions 320 --hidden1 256 --hidden2 32 --decoder1 465.333573 --lr 7.34e-04 --dropout 0.467014 --weight_decay 2.19e-06 --alpha 1.295537 --beta 1.573586 --gamma 1.071743 --moco_momentum 0.974776 --moco_t 0.881687 --moco_tau1 0.504155 --moco_tau2 0.796956 --gat_heads 2 --gt_heads 4 --fusion_heads 4 --batch 32 --moco_K 4096 --fusion_strategy self_attention --feature_type one_hot --moco_type double_tau --enable_view_0 true
