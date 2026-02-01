#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一参数注册表

将autodl.py和parms_setting.py的所有参数统一注册到参数管理系统中
"""

from parameter_manager import ParameterDefinition, register_module_parameters


def register_autodl_parameters():
    """注册autodl.py的参数"""
    autodl_params = [
        # 基本优化参数
        ParameterDefinition(
            name="task_type",
            type=str,
            default="LDA",
            choices=['LDA', 'MDA', 'LMI'],
            help="任务类型"
        ),
        ParameterDefinition(
            name="max_iterations",
            type=int,
            default=50,
            help="最大迭代次数"
        ),
        ParameterDefinition(
            name="max_time_hours",
            type=float,
            default=24.0,
            help="最大运行时间（小时）"
        ),
        ParameterDefinition(
            name="random_seed",
            type=int,
            default=42,
            help="随机种子"
        ),
        
        # 优化设置
        ParameterDefinition(
            name="acquisition_function",
            type=str,
            default="EI",
            choices=['EI', 'PI', 'UCB'],
            help="采集函数类型"
        ),
        ParameterDefinition(
            name="n_initial_points",
            type=int,
            default=10,
            help="初始随机采样点数"
        ),
        ParameterDefinition(
            name="checkpoint_dir",
            type=str,
            default="checkpoints",
            help="检查点保存目录"
        ),
        
        # 数据配置
        ParameterDefinition(
            name="pos_file",
            type=str,
            default=None,
            help="正样本文件路径"
        ),
        ParameterDefinition(
            name="neg_file",
            type=str,
            default=None,
            help="负样本文件路径"
        ),
        
        # 日志配置
        ParameterDefinition(
            name="log_dir",
            type=str,
            default="logs",
            help="日志目录"
        ),
        ParameterDefinition(
            name="log_level",
            type=str,
            default="INFO",
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            help="日志级别"
        ),
        
        # 配置文件支持
        ParameterDefinition(
            name="config",
            type=str,
            default=None,
            help="配置文件路径"
        ),
        ParameterDefinition(
            name="save_config",
            type=str,
            default=None,
            help="保存配置到文件"
        ),
        
        # 多目标优化
        ParameterDefinition(
            name="multi_objective",
            type=bool,
            default=False,
            help="是否启用多目标优化"
        ),
        ParameterDefinition(
            name="objectives",
            type=str,
            default="auc",
            help="优化目标（逗号分隔）"
        )
    ]
    
    register_module_parameters("autodl", autodl_params)


def register_parms_setting_parameters():
    """注册parms_setting.py的参数"""
    parms_params = [
        # 公共参数
        ParameterDefinition(
            name="seed",
            type=int,
            default=0,
            help="随机种子，默认 0"
        ),
        ParameterDefinition(
            name="file",
            type=str,
            default="dataset1/LDA.edgelist",
            dest="in_file",
            help="正样本文件（--in_file 的别名）"
        ),
        ParameterDefinition(
            name="neg_sample",
            type=str,
            default="dataset1/non_LDA.edgelist",
            help="未知关联（负样本）文件路径"
        ),
        ParameterDefinition(
            name="validation_type",
            type=str,
            default="5_cv1",
            choices=['5_cv1', '5_cv2', '5-cv1', '5-cv2'],
            help="交叉验证类型，默认 5_cv1"
        ),
        
        # 特征构建与增强
        ParameterDefinition(
            name="feature_type",
            type=str,
            default="normal",
            choices=['one_hot', 'uniform', 'normal', 'position'],
            help="初始节点特征类型，默认 normal"
        ),
        ParameterDefinition(
            name="noise_std",
            type=float,
            default=0.01,
            help="高斯噪声标准差，默认 0.01"
        ),
        ParameterDefinition(
            name="mask_rate",
            type=float,
            default=0.1,
            help="列掩蔽比例，默认 0.1"
        ),
        ParameterDefinition(
            name="augment_seed",
            type=int,
            default=None,
            help="增强随机种子；None 时使用 seed+fold"
        ),
        ParameterDefinition(
            name="augment_mode",
            type=str,
            default="static",
            choices=['static', 'online'],
            help="增强模式：static（按折离线）/ online（训练时在线），默认 static"
        ),
        ParameterDefinition(
            name="augment",
            type=str,
            default="random_permute_features,attribute_mask,noise_then_mask",
            help="增强方式，多个增强用逗号分隔"
        ),
        
        # 训练设置
        ParameterDefinition(
            name="lr",
            type=float,
            default=5e-4,
            help="学习率，默认 5e-4"
        ),
        ParameterDefinition(
            name="learning_rate",
            type=float,
            default=5e-4,  # 添加默认值
            dest="lr",
            help="--lr 的别名"
        ),
        ParameterDefinition(
            name="dropout",
            type=float,
            default=0.1,
            help="Dropout 比例，默认 0.1"
        ),
        ParameterDefinition(
            name="weight_decay",
            type=float,
            default=5e-4,
            help="权重衰减（L2 正则），默认 5e-4"
        ),
        ParameterDefinition(
            name="batch",
            type=int,
            default=25,
            help="批大小，默认 25"
        ),
        ParameterDefinition(
            name="epochs",
            type=int,
            default=5,  # 修改为5个epoch
            help="训练轮数，默认 5"
        ),
        
        # 多任务损失权重
        ParameterDefinition(
            name="loss_ratio1",
            type=float,
            default=1.0,
            help="任务1损失权重，默认 1"
        ),
        ParameterDefinition(
            name="loss_ratio2",
            type=float,
            default=0.5,
            help="任务2损失权重，默认 0.5"
        ),
        ParameterDefinition(
            name="loss_ratio3",
            type=float,
            default=0.5,
            help="任务3损失权重，默认 0.5"
        ),
        
        # 模型结构参数
        ParameterDefinition(
            name="dimensions",
            type=int,
            default=256,
            help="初始特征维度 d，默认 256"
        ),
        ParameterDefinition(
            name="embed_dim",
            type=int,
            default=256,  # 添加默认值
            dest="dimensions",
            help="--dimensions 的别名"
        ),
        ParameterDefinition(
            name="hidden1",
            type=int,
            default=128,
            help="编码器第 1 层隐藏维度，默认 128"
        ),
        ParameterDefinition(
            name="hidden2",
            type=int,
            default=64,
            help="编码器第 2 层隐藏维度，默认 64"
        ),
        ParameterDefinition(
            name="decoder1",
            type=int,
            default=512,
            help="解码器第 1 层隐藏维度，默认 512"
        ),
        
        # 注意力头
        ParameterDefinition(
            name="gat_heads",
            type=int,
            default=4,
            help="GAT 编码器的注意力头数，默认 4"
        ),
        ParameterDefinition(
            name="gt_heads",
            type=int,
            default=4,
            help="Graph Transformer 编码器的注意力头数，默认 4"
        ),
        ParameterDefinition(
            name="fusion_heads",
            type=int,
            default=4,
            help="对偶融合的多头注意力头数，默认 4"
        ),
        ParameterDefinition(
            name="fusion_strategy",
            type=str,
            default="self_attention",
            choices=['self_attention', 'co_attention', 'hybrid', 'transformer_multihead'],
            help="两实体融合策略"
        ),
        ParameterDefinition(
            name="fusion_weight",
            type=float,
            default=0.5,
            help="混合策略中自注意力的权重(0-1)，默认 0.5"
        ),
        
        # 协作注意力参数
        ParameterDefinition(
            name="co_hidden_dim",
            type=int,
            default=None,
            help="协作注意力的隐藏维度，None时使用输入维度"
        ),
        ParameterDefinition(
            name="use_co_attention",
            type=bool,
            default=False,
            help="是否使用协作注意力，默认 False"
        ),
        ParameterDefinition(
            name="co_attention_type",
            type=str,
            default="transformer",
            help="协作注意力类型和参数"
        ),
        ParameterDefinition(
            name="attention_config",
            type=str,
            default=None,
            help="高级注意力配置字符串"
        ),
        ParameterDefinition(
            name="use_multihead",
            type=bool,
            default=False,
            help="是否使用多头注意力机制，默认 False"
        ),
        ParameterDefinition(
            name="transformer_style",
            type=bool,
            default=True,
            help="是否使用Transformer风格的注意力，默认 True"
        ),
        
        # 模型类型选择
        ParameterDefinition(
            name="model_type",
            type=str,
            default="moco",
            choices=['moco', 'byol'],
            help="自监督学习模型类型选择：moco|byol，默认 moco"
        ),
        
        # MoCo参数
        ParameterDefinition(
            name="moco_type",
            type=str,
            default="basic",
            choices=['basic', 'double_tau'],
            help="MoCo类型选择：basic|double_tau，默认 basic"
        ),
        ParameterDefinition(
            name="moco_config",
            type=str,
            default=None,
            help="高级MoCo配置字符串"
        ),
        ParameterDefinition(
            name="moco_K",
            type=int,
            default=4096,
            help="MoCo队列大小，默认 4096"
        ),
        ParameterDefinition(
            name="moco_queue",
            type=int,
            default=4096,
            help="MoCo 队列长度，默认 4096"
        ),
        ParameterDefinition(
            name="moco_momentum",
            type=float,
            default=0.999,
            help="MoCo 动量 m，默认 0.999"
        ),
        ParameterDefinition(
            name="moco_t",
            type=float,
            default=0.2,
            help="MoCo 温度 T，默认 0.2"
        ),
        ParameterDefinition(
            name="moco_tau1",
            type=float,
            default=0.2,
            help="DoubleTau MoCo正样本温度系数，默认 0.2"
        ),
        ParameterDefinition(
            name="moco_tau2",
            type=float,
            default=0.2,
            help="DoubleTau MoCo负样本温度系数，默认 0.2"
        ),
        ParameterDefinition(
            name="proj_dim",
            type=int,
            default=None,
            help="投影维度，默认随 hidden2"
        ),
        ParameterDefinition(
            name="queue_warmup_steps",
            type=int,
            default=0,
            help="队列预热步数，默认 0"
        ),
        ParameterDefinition(
            name="enable_view_0",
            type=bool,
            default=True,
            help="是否启用MoCo第0视图，默认 True"
        ),
        ParameterDefinition(
            name="num_views",
            type=int,
            default=3,
            help="MoCo多视图数量，默认3"
        ),
        
        # BYOL参数
        ParameterDefinition(
            name="byol_config",
            type=str,
            default=None,
            help="高级BYOL配置字符串"
        ),
        ParameterDefinition(
            name="byol_predictor_dim",
            type=int,
            default=256,
            help="BYOL预测器隐藏维度，默认 256"
        ),
        ParameterDefinition(
            name="byol_ema_momentum",
            type=float,
            default=0.996,
            help="BYOL目标网络EMA动量，默认 0.996"
        ),
        ParameterDefinition(
            name="byol_temperature",
            type=float,
            default=0.2,
            help="BYOL温度系数，默认 0.2"
        ),
        
        # CPU并行与数据加载
        ParameterDefinition(
            name="threads",
            type=int,
            default=32,
            help="后端线程上限，默认 32"
        ),
        ParameterDefinition(
            name="num_workers",
            type=int,
            default=-1,
            help="DataLoader workers，默认 -1"
        ),
        ParameterDefinition(
            name="prefetch_factor",
            type=int,
            default=4,
            help="DataLoader 预取因子，默认 4"
        ),
        ParameterDefinition(
            name="chunk_size",
            type=int,
            default=0,
            help="CPU 任务通用切片大小；0 自动（默认 20000）"
        ),
        
        # 其他参数
        ParameterDefinition(
            name="similarity_threshold",
            type=float,
            default=0.5,
            help="图构建中的相似度阈值，默认 0.5"
        ),
        ParameterDefinition(
            name="alpha",
            type=float,
            default=1.0,
            dest="loss_ratio1",
            help="监督任务权重（BCE），--loss_ratio1 别名"
        ),
        ParameterDefinition(
            name="beta",
            type=float,
            default=0.5,
            dest="loss_ratio2",
            help="对比任务权重（InfoNCE/CE），--loss_ratio2 别名"
        ),
        ParameterDefinition(
            name="gamma",
            type=float,
            default=0.5,
            dest="loss_ratio3",
            help="节点对抗任务权重（BCEWithLogits），--loss_ratio3 别名"
        ),
        
        # 数据保存
        ParameterDefinition(
            name="save_datasets",
            type=bool,
            default=False,
            help="是否保存构建的数据集，默认 False"
        ),
        ParameterDefinition(
            name="save_format",
            type=str,
            default="npy",
            choices=['npy', 'txt'],
            help="数据保存格式，默认 npy"
        ),
        ParameterDefinition(
            name="save_dir_prefix",
            type=str,
            default="result/data",
            help="保存目录前缀，默认 result/data"
        ),
        ParameterDefinition(
            name="run_name",
            type=str,
            default=None,
            help="运行名称，默认 None"
        ),
        ParameterDefinition(
            name="shutdown",
            type=bool,
            default=False,
            help="仅 Linux：运行结束后关机"
        ),
        
        # 对抗学习参数
        ParameterDefinition(
            name="adv_mode",
            type=str,
            default="none",
            choices=['none', 'mgraph'],
            help="对抗模式，默认 none"
        ),
        ParameterDefinition(
            name="adv_norm",
            type=str,
            default="linf",
            choices=['linf', 'l2'],
            help="对抗扰动范数，默认 linf"
        ),
        ParameterDefinition(
            name="adv_eps",
            type=float,
            default=0.01,
            help="对抗扰动幅度，默认 0.01"
        ),
        ParameterDefinition(
            name="adv_alpha",
            type=float,
            default=0.002,
            help="PGD步长，默认 0.002"
        ),
        ParameterDefinition(
            name="adv_steps",
            type=int,
            default=5,
            help="PGD迭代步数，默认 5"
        ),
        ParameterDefinition(
            name="adv_rand_init",
            type=bool,
            default=True,
            help="PGD随机初始化，默认 True"
        ),
        ParameterDefinition(
            name="adv_project",
            type=bool,
            default=True,
            help="PGD投影到约束集，默认 True"
        ),
        ParameterDefinition(
            name="adv_agg",
            type=str,
            default="mean",
            choices=['mean', 'sum', 'max'],
            help="多图对抗聚合方式，默认 mean"
        ),
        ParameterDefinition(
            name="adv_budget",
            type=str,
            default="shared",
            choices=['shared', 'independent'],
            help="多图对抗预算分配，默认 shared"
        ),
        ParameterDefinition(
            name="adv_use_amp",
            type=bool,
            default=False,
            help="对抗训练使用AMP，默认 False"
        ),
        ParameterDefinition(
            name="adv_on_moco",
            type=bool,
            default=False,
            help="对MoCo特征进行对抗，默认 False"
        ),
        ParameterDefinition(
            name="adv_seed",
            type=int,
            default=None,
            help="对抗随机种子，默认 None"
        ),
        ParameterDefinition(
            name="adv_clip_min",
            type=float,
            default=0.0,
            help="对抗扰动裁剪下界，默认 0.0"
        ),
        ParameterDefinition(
            name="adv_clip_max",
            type=float,
            default=1.0,
            help="对抗扰动裁剪上界，默认 1.0"
        ),
        ParameterDefinition(
            name="adv_warmup_end",
            type=int,
            default=3,
            help="PGD启用的起始epoch，默认 3"
        ),
        
        # 阈值扫描与温度校准
        ParameterDefinition(
            name="enable_threshold_scan",
            type=bool,
            default=True,
            help="是否启用阈值扫描，默认 True"
        ),
        ParameterDefinition(
            name="threshold_min",
            type=float,
            default=0.1,
            help="阈值扫描最小值，默认 0.1"
        ),
        ParameterDefinition(
            name="threshold_max",
            type=float,
            default=0.9,
            help="阈值扫描最大值，默认 0.9"
        ),
        ParameterDefinition(
            name="threshold_step",
            type=float,
            default=0.05,
            help="阈值扫描步长，默认 0.05"
        ),
        ParameterDefinition(
            name="enable_temp_scaling",
            type=bool,
            default=True,
            help="是否启用温度校准，默认 True"
        ),
        ParameterDefinition(
            name="temp_grid_min",
            type=float,
            default=0.1,
            help="温度校准网格最小值，默认 0.1"
        ),
        ParameterDefinition(
            name="temp_grid_max",
            type=float,
            default=5.0,
            help="温度校准网格最大值，默认 5.0"
        ),
        ParameterDefinition(
            name="temp_grid_num",
            type=int,
            default=26,
            help="温度校准网格点数，默认 26"
        ),
        
        # K折重算与缓存
        ParameterDefinition(
            name="kfold_recompute",
            type=bool,
            default=True,
            help="按折仅用训练集重算预处理/相似度/EM，默认 True"
        ),
        ParameterDefinition(
            name="kfold_cache",
            type=bool,
            default=False,
            help="按 {fold, adv_hash, epoch, iter} 可选缓存对抗样本，默认 False"
        )
    ]
    
    register_module_parameters("parms_setting", parms_params)


def initialize_unified_parameters():
    """初始化统一参数系统"""
    print("正在初始化统一参数系统...")
    
    # 注册所有模块的参数
    register_autodl_parameters()
    register_parms_setting_parameters()
    
    print("✓ autodl.py 参数已注册")
    print("✓ parms_setting.py 参数已注册")
    print("✓ 统一参数系统初始化完成")


if __name__ == "__main__":
    initialize_unified_parameters()
    
    # 测试参数解析
    from parameter_manager import get_parameter_manager
    
    manager = get_parameter_manager()
    
    # 测试解析包含autodl参数的命令行
    test_args = ["--max_iterations", "30", "--random_seed", "123", "--epochs", "100"]
    
    try:
        parsed = manager.parse_arguments(test_args)
        print(f"\n✓ 成功解析参数:")
        print(f"  max_iterations: {parsed.max_iterations}")
        print(f"  random_seed: {parsed.random_seed}")
        print(f"  epochs: {parsed.epochs}")
        print(f"  task_type: {parsed.task_type}")
        
        # 检查冲突
        conflicts = manager.get_conflicts()
        if conflicts:
            print(f"\n⚠️  发现 {len(conflicts)} 个参数冲突:")
            for name, conflict in conflicts.items():
                print(f"  - {name}: {conflict.conflicting_modules}")
        else:
            print("\n✓ 没有参数冲突")
            
    except Exception as e:
        print(f"\n❌ 参数解析失败: {e}")