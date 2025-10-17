#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数优化（HPO）脚本：三阶段流程
- 阶段B：粗随机搜索（每任务 N=60，epochs=3），以 AUPRC 为主排序，输出 top10 与历史
- 阶段C：对阶段B的 top10 在邻域精调（epochs=20），启用 online 增强与 mgraph 对抗（入口预置）
- 最终复现：对每任务 top3 配置在 seeds=[0,1,2] 下复现（入口预置）

实现策略：
- 直接复用当前目录的 load_data → Create_model → train_model，进行 5 折评估
- 为每个 trial 覆盖 layer.args 的相关字段，确保 EM 内部读取的超参与 trial 一致
- 严格生成 experiments/hpo_{TIMESTAMP}/{TASK}/ 下的 CSV/JSON 报告与错误日志
"""

import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
# 可选引入 Optuna（作为搜索后端），若未安装则退回随机搜索
# Optuna 已移除
_OPTUNA_AVAILABLE = False  # 保留标志固定为 False，避免外部引用错误

# 引入项目内部模块（与 main.py 同步）
# 安全包装：延迟导入并在导入前暂时清空 sys.argv，避免 settings() 解析 HPO 自定义 CLI
import importlib

def _safe_import(mod_name: str):
    import sys
    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.executable]
        return importlib.import_module(mod_name)
    finally:
        sys.argv = argv_backup

def _ps_settings():
    """安全调用 parms_setting.settings() 获取基础 args"""
    ps = _safe_import("parms_setting")
    import sys as _sys
    argv_backup = list(_sys.argv)
    try:
        _sys.argv = [_sys.executable]
        return ps.settings()
    finally:
        _sys.argv = argv_backup

def _init_autodl_env(args):
    return _safe_import("autodl").init_autodl_env(args)

def _init_logging(run_name=None):
    return _safe_import("log_output_manager").init_logging(run_name=run_name)

def _redirect_print(enable=True):
    return _safe_import("log_output_manager").redirect_print(enable)

def _make_result_run_dir(prefix="data"):
    return _safe_import("log_output_manager").make_result_run_dir(prefix)

def _get_run_paths():
    return _safe_import("log_output_manager").get_run_paths()

def _finalize_run():
    return _safe_import("log_output_manager").finalize_run()

def _load_data(args):
    return _safe_import("data_preprocess").load_data(args)

def _Create_model(args):
    return _safe_import("instantiation").Create_model(args)

def _train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args, fold_idx=None):
    return _safe_import("train").train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args, fold_idx=fold_idx)

# 注意：layer.py 在 import 时会执行 settings() 获取全局 args
# 我们需要在每个 trial 运行前，覆盖其中的字段以保持一致
# 延迟导入 layer，避免其在导入时调用 settings() 与 HPO CLI 冲突
_LAYER_MOD = None
def _get_layer_mod():
    """
    安全延迟导入 layer：
    - 在导入前暂时清空 sys.argv，防止 layer.settings() 解析 HPO 的 --stage 等参数
    - 导入完成后恢复 sys.argv
    - 结果缓存到全局，避免重复导入
    """
    global _LAYER_MOD
    if _LAYER_MOD is not None:
        return _LAYER_MOD
    import importlib, sys
    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.executable]
        _LAYER_MOD = importlib.import_module("layer")
    finally:
        sys.argv = argv_backup
    return _LAYER_MOD


# ============== 工具函数 ==============

def now_tag() -> str:
    """返回当前时间戳短标签"""
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    """确保目录存在"""
    p.mkdir(parents=True, exist_ok=True)

def update_layer_args(args_obj: Any) -> None:
    """将当前 trial 的关键参数写入 layer.args（全局），以便 EM 内部读取"""
    # 仅覆盖训练相关参数，避免污染不可用项
    la = getattr(_get_layer_mod(), "args", None)
    if la is None:
        return
    for key in [
        "gat_heads", "fusion_heads",
        "proj_dim", "moco_queue", "moco_momentum", "moco_t", "queue_warmup_steps",
        "noise_std", "mask_rate", "augment", "augment_mode", "augment_seed",
        "adv_mode", "adv_on_moco", "adv_eps", "adv_alpha", "adv_steps", "adv_rand_init", "adv_project",
        "adv_agg", "adv_budget", "adv_use_amp", "adv_clip_min", "adv_clip_max",
        "num_views",
    ]:
        if hasattr(args_obj, key):
            setattr(la, key, getattr(args_obj, key))


def derive_seed(*parts: int) -> int:
    """简易派生稳定种子（与 utils.derive_seed 保持一致思想）"""
    mod = 2**32
    acc = 0
    for p in parts:
        acc = (acc * 1000003 + int(p)) % mod
    return int(acc if acc != 0 else 1)


def sample_config(task: str, rng: np.random.Generator) -> Dict[str, Any]:
    """阶段B：随机采样一个配置（与用户确认的粗搜索空间一致）"""
    # 搜索空间
    embed_dim_choices = [32, 64, 128, 256]
    lr_choices = [1e-4, 5e-4, 1e-3]
    wd_choices = [0.0, 5e-5, 5e-4]
    dropout_choices = [0.0, 0.1, 0.2]
    batch_choices = [16, 32, 64, 128]
    # 预热与早停相关
    lr_warmup_epochs_choices = [0, 2, 3, 5]
    lr_min_factor_choices = [0.1, 0.2, 0.3]
    early_stop_patience_choices = [0, 3, 5]
    early_stop_min_delta_choices = [0.0, 1e-4, 5e-4]
    moco_queue_choices = [1024, 4096, 8192]
    moco_momentum_choices = [0.95, 0.99, 0.999]
    moco_t_choices = [0.07, 0.1, 0.2]
    proj_dim_choices = [32, 64, 128, None]
    augment_choices = ["none", "random_permute_features", "add_noise", "attribute_mask", "noise_then_mask"]
    noise_std_choices = [0.005, 0.01, 0.02]
    mask_rate_choices = [0.05, 0.1, 0.2]
    # 损失权重（alpha/beta/gamma）
    alpha_choices = [0.5, 1.0, 2.0]
    beta_choices = [0.1, 0.5, 1.0]
    gamma_choices = [0.0, 0.1, 0.5]
    # 视图数（MoCo）
    num_views_choices = [2, 3]

    embed_dim = rng.choice(embed_dim_choices)
    hidden1_choices = [max(32, embed_dim // 2), embed_dim, min(2 * embed_dim, 512)]
    hidden2_choices = [max(16, embed_dim // 4), max(32, embed_dim // 2), embed_dim]

    cfg = {
        "task_type": task,
        "dimensions": int(embed_dim),
        "hidden1": int(rng.choice(hidden1_choices)),
        "hidden2": int(rng.choice(hidden2_choices)),
        "decoder1": int(rng.choice([256, 512])),
        "lr": float(rng.choice(lr_choices)),
        "weight_decay": float(rng.choice(wd_choices)),
        "dropout": float(rng.choice(dropout_choices)),
        # 训练策略
        "base_batch": 32,  # 线性学习率缩放基准批次
        "lr_warmup_epochs": int(rng.choice(lr_warmup_epochs_choices)),
        "lr_min_factor": float(rng.choice(lr_min_factor_choices)),
        "early_stop_patience": int(rng.choice(early_stop_patience_choices)),
        "early_stop_min_delta": float(rng.choice(early_stop_min_delta_choices)),
        "batch": int(rng.choice(batch_choices)),
        "moco_queue": int(rng.choice(moco_queue_choices)),
        "moco_momentum": float(rng.choice(moco_momentum_choices)),
        "moco_t": float(rng.choice(moco_t_choices)),
        "proj_dim": (None if (c := rng.choice(proj_dim_choices)) is None else int(c)),
        "augment": str(rng.choice(augment_choices)),
        "augment_mode": "static",  # 阶段B固定 static
        "noise_std": float(rng.choice(noise_std_choices)),
        "mask_rate": float(rng.choice(mask_rate_choices)),
        "alpha": float(rng.choice(alpha_choices)),   # 别名到 loss_ratio1
        "beta": float(rng.choice(beta_choices)),     # 别名到 loss_ratio2
        "gamma": float(rng.choice(gamma_choices)),   # 别名到 loss_ratio3
        "num_views": int(rng.choice(num_views_choices)),
        # 阶段B关闭对抗
        "adv_mode": "none",
        "adv_on_moco": False,
        "queue_warmup_steps": int(rng.choice([0, 200])),
        "gat_heads": int(rng.choice([2, 4])),
        "fusion_heads": int(rng.choice([2, 4])),
    }
    # 约束：若 augment=none，则忽略 noise_std/mask_rate
    if cfg["augment"] in ("none", "null", ""):
        cfg["noise_std"] = 0.0
        cfg["mask_rate"] = 0.0
    # proj_dim 兜底：None 使用 hidden2
    if (cfg["proj_dim"] is None) or (int(cfg["proj_dim"]) <= 0 if cfg["proj_dim"] is not None else False):
        cfg["proj_dim"] = int(cfg["hidden2"])
    return cfg


def trial_to_args(base_args: Any, cfg: Dict[str, Any], seed: int, in_file: str, neg_file: str, epochs: int) -> Any:
    """将采样配置写入 args 对象并返回"""
    # 基本任务文件
    base_args.task_type = cfg["task_type"]
    base_args.in_file = in_file
    base_args.neg_sample = neg_file
    base_args.validation_type = "5_cv1"
    # 训练与结构
    # 线性学习率缩放：随批次线性放大
    base_batch = int(cfg.get("base_batch", 32) or 32)
    scaled_lr = float(cfg["lr"]) * (float(cfg["batch"]) / float(base_batch if base_batch > 0 else 32))
    base_args.lr = scaled_lr
    base_args.learning_rate = scaled_lr
    base_args.weight_decay = cfg["weight_decay"]
    base_args.dropout = cfg["dropout"]
    base_args.batch = cfg["batch"]
    base_args.epochs = int(epochs)
    # 预热与早停超参
    base_args.lr_warmup_epochs = int(cfg.get("lr_warmup_epochs", 0) or 0)
    base_args.lr_min_factor = float(cfg.get("lr_min_factor", 0.1) or 0.1)
    base_args.early_stop_patience = int(cfg.get("early_stop_patience", 0) or 0)
    base_args.early_stop_min_delta = float(cfg.get("early_stop_min_delta", 0.0) or 0.0)
    base_args.dimensions = cfg["dimensions"]
    base_args.embed_dim = cfg["dimensions"]
    base_args.hidden1 = cfg["hidden1"]
    base_args.hidden2 = cfg["hidden2"]
    base_args.decoder1 = cfg["decoder1"]
    base_args.proj_dim = cfg["proj_dim"]
    base_args.gat_heads = cfg["gat_heads"]
    base_args.fusion_heads = cfg["fusion_heads"]
    base_args.num_views = cfg["num_views"]
    # 损失权重
    base_args.loss_ratio1 = cfg["alpha"]
    base_args.loss_ratio2 = cfg["beta"]
    base_args.loss_ratio3 = cfg["gamma"]
    base_args.alpha = cfg["alpha"]
    base_args.beta = cfg["beta"]
    base_args.gamma = cfg["gamma"]
    # 增强
    base_args.augment_mode = cfg["augment_mode"]
    base_args.augment = cfg["augment"]
    base_args.noise_std = cfg["noise_std"]
    base_args.mask_rate = cfg["mask_rate"]
    base_args.augment_seed = seed
    base_args.feature_type = "one_hot"  # 与用户基线一致
    base_args.similarity_threshold = 0.5
    # MoCo
    base_args.moco_queue = cfg["moco_queue"]
    base_args.moco_momentum = cfg["moco_momentum"]
    base_args.moco_t = cfg["moco_t"]
    base_args.queue_warmup_steps = cfg["queue_warmup_steps"]
    # 对抗
    base_args.adv_mode = cfg["adv_mode"]
    base_args.adv_on_moco = cfg["adv_on_moco"]
    # 其他
    base_args.seed = seed
    base_args.cuda = True
    base_args.run_name = None  # 由外层设置
    # 并行与数据保存
    base_args.num_workers = -1
    base_args.threads = 32
    base_args.save_datasets = False
    return base_args


def parse_epoch_csv_loss(last_csv_path: Path) -> Tuple[float, float]:
    """
    从每折的 CSV 中解析首/末 epoch 的 loss_train，返回 (first_loss, last_loss)
    若无法读取，返回 (math.nan, math.nan)
    """
    try:
        txt = last_csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(txt) <= 1:
            return (math.nan, math.nan)
        # 第一数据行与最后数据行
        first = txt[1].split(",")
        last = txt[-1].split(",")
        # CSV 列固定顺序：epoch,loss_train,...
        return (float(first[1]), float(last[1]))
    except Exception:
        return (math.nan, math.nan)


def run_one_trial(task: str, trial_id: int, cfg: Dict[str, Any], epochs: int, exp_task_dir: Path, in_file: str, neg_file: str, fixed_seed: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    执行单个 trial：返回 (metrics, meta)
    metrics：包含 AUROC/AUPRC/F1/Loss 的 mean/std
    meta：包含 run_name、log 路径、时间、是否重试、错误信息等
    """
    # 运行名：task_trial_{trial_id}
    run_name = f"{task}_trial_{trial_id}"
    meta: Dict[str, Any] = {"run_name": run_name, "trial_id": trial_id, "retry": False, "error": None}
    start_t = time.time()

    # 构造 args
    args_obj = _ps_settings()
    # 派生种子（确保每 trial 稳定）
    seed = int(fixed_seed) if fixed_seed is not None else derive_seed(int(getattr(args_obj, "seed", 0)), trial_id)
    args_obj = trial_to_args(args_obj, cfg, seed, in_file, neg_file, epochs)
    args_obj.run_name = run_name

    # 覆盖到 layer.args
    update_layer_args(args_obj)

    # 初始化环境与日志
    _init_autodl_env(args_obj)
    logger = _init_logging(run_name=args_obj.run_name)
    # 根据环境变量控制是否重定向日志到文件
    if os.environ.get("HPO_NO_REDIRECT") == "1":
        _redirect_print(False)
    else:
        _redirect_print(True)
    _make_result_run_dir("data")
    # 写 params.json（含seed与augment_seed等全部参数）到当前运行的 metrics 目录
    try:
        paths_p = _get_run_paths()
        run_dir = Path(paths_p.get("run_result_dir") or "")
        metrics_dir = run_dir / "metrics"
        ensure_dir(metrics_dir)
        params_path = metrics_dir / "params.json"
        payload = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)) for k, v in vars(args_obj).items()}
        payload["seed"] = int(getattr(args_obj, "seed", 0))
        payload["augment_seed"] = int(getattr(args_obj, "augment_seed", payload["seed"]))
        params_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # 加载数据与 5 折评估
    data_o_folds, data_a_folds, train_loaders, test_loaders = _load_data(args_obj)

    # 逐折训练与测试
    all_fold_results: List[Dict[str, Any]] = []
    for fold in range(5):
        model, optimizer = _Create_model(args_obj)
        fold_res = _train_model(model, optimizer, data_o_folds[fold], data_a_folds[fold], train_loaders[fold], test_loaders[fold], args_obj, fold_idx=fold+1)
        all_fold_results.append(fold_res)

    # 计算汇总
    def _collect(key: str) -> Tuple[float, float]:
        xs = [float(r[key]) for r in all_fold_results]
        return (float(np.mean(xs)), float(np.std(xs)))

    auroc_mean, auroc_std = _collect("auroc")
    auprc_mean, auprc_std = _collect("auprc")
    f1_mean, f1_std = _collect("f1")
    loss_mean, loss_std = _collect("loss")

    # 故障检测：读取最后一个折的 per-epoch CSV 判断发散
    paths = _get_run_paths()
    run_id = paths.get("run_id") or ""
    last_csv = exp_task_dir / f"{run_name}_fold_last_metrics_{run_id}.marker"  # 我们在本地创建一个标记文件，同时解析 OUTPUT 中真实 CSV
    try:
        # 找到一个 CSV 文件（例如 fold_1）
        csv_dir = Path(paths.get("run_result_dir") or "")
        csv_sample = None
        for fold_idx in range(1, 6):
            p = csv_dir / "metrics" / f"train_epoch_metrics_fold_{fold_idx}_{run_id}.csv"
            if p.exists():
                csv_sample = p
                break
        if csv_sample is not None:
            first_loss, last_loss = parse_epoch_csv_loss(csv_sample)
            # 发散判定：首末比 > 10 且末值更大
            if (not math.isnan(first_loss)) and (not math.isnan(last_loss)) and (first_loss > 0.0) and (last_loss / first_loss > 10.0):
                meta["divergent"] = True
            else:
                meta["divergent"] = False
        else:
            meta["divergent"] = False
        # 标记文件（便于排查）
        ensure_dir(exp_task_dir)
        last_csv.write_text(f"csv_checked={csv_sample is not None}\n", encoding="utf-8")
    except Exception:
        meta["divergent"] = False

    # 输出本次 trial 的 CSV 行（历史文件由外层写）
    _finalize_run()
    meta["time_s"] = round(time.time() - start_t, 3)
    meta["log_path"] = str(Path(paths.get("log_dir")) / f"EM_{run_name}_{run_id}.log") if run_id else None

    metrics = {
        "AUROC_mean": auroc_mean, "AUROC_std": auroc_std,
        "AUPRC_mean": auprc_mean, "AUPRC_std": auprc_std,
        "F1_mean": f1_mean, "F1_std": f1_std,
        "Loss_mean": loss_mean, "Loss_std": loss_std,
        "run_id": run_id,
    }
    return metrics, meta


def run_trial_with_retry(task: str, trial_id: int, cfg: Dict[str, Any], epochs: int, exp_task_dir: Path, in_file: str, neg_file: str, fixed_seed: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """包裹一次重试逻辑：NaN/Inf 或发散时将 lr*=0.5 重试一次；OOM/其他异常记录错误并返回"""
    try:
        metrics, meta = run_one_trial(task, trial_id, cfg, epochs, exp_task_dir, in_file, neg_file, fixed_seed=fixed_seed)
        # 简单 NaN/Inf 检查与发散检查
        def _bad(x: float) -> bool:
            return (x is None) or (not math.isfinite(x)) or (x < 0)
        if _bad(metrics["AUPRC_mean"]) or meta.get("divergent", False):
            # 重试一次：lr *= 0.5
            cfg_retry = dict(cfg)
            cfg_retry["lr"] = float(cfg["lr"]) * 0.5
            meta["retry"] = True
            metrics, meta2 = run_one_trial(task, trial_id, cfg_retry, epochs, exp_task_dir, in_file, neg_file, fixed_seed=fixed_seed)
            # 合并关键信息
            meta.update({"retry_run_name": meta2.get("run_name"), "retry_log_path": meta2.get("log_path")})
        return metrics, meta
    except RuntimeError as e:
        # 可能是 OOM
        msg = str(e)
        meta = {"run_name": f"{task}_trial_{trial_id}", "trial_id": trial_id, "retry": False, "error": "OOM" if "out of memory" in msg.lower() else "RuntimeError", "time_s": None}
        # 写 error report
        ensure_dir(exp_task_dir)
        (exp_task_dir / "error_report.txt").write_text(f"{meta['run_name']}, {meta['error']}, detail={msg}\n", encoding="utf-8")
        return {
            "AUROC_mean": 0.0, "AUROC_std": 0.0,
            "AUPRC_mean": 0.0, "AUPRC_std": 0.0,
            "F1_mean": 0.0, "F1_std": 0.0,
            "Loss_mean": 0.0, "Loss_std": 0.0,
            "run_id": None,
        }, meta
    except Exception as e:
        meta = {"run_name": f"{task}_trial_{trial_id}", "trial_id": trial_id, "retry": False, "error": "Exception", "time_s": None}
        ensure_dir(exp_task_dir)
        # 保存错误
        err_path = exp_task_dir / f"{task}_trial_{trial_id}.err"
        err_path.write_text("".join(traceback.format_exception(e)), encoding="utf-8")
        (exp_task_dir / "error_report.txt").write_text(f"{meta['run_name']}, Exception, err_path={err_path}\n", encoding="utf-8")
        return {
            "AUROC_mean": 0.0, "AUROC_std": 0.0,
            "AUPRC_mean": 0.0, "AUPRC_std": 0.0,
            "F1_mean": 0.0, "F1_std": 0.0,
            "Loss_mean": 0.0, "Loss_std": 0.0,
            "run_id": None,
        }, meta


def sort_and_top(trials: List[Dict[str, Any]], topn: int = 10) -> List[Dict[str, Any]]:
    """按 AUPRC_mean 降序，tie-breaker: AUROC_mean、F1_mean"""
    def _key(d: Dict[str, Any]) -> Tuple[float, float, float]:
        return (float(d["AUPRC_mean"]), float(d["AUROC_mean"]), float(d["F1_mean"]))
    trials_sorted = sorted(trials, key=_key, reverse=True)
    return trials_sorted[:topn]


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["LDA", "MDA", "LMI"], choices=["LDA", "MDA", "LMI"], help="选择需要优化的任务集合")
    parser.add_argument("--search_backend", type=str, default="random", choices=["random"], help="选择搜索后端：仅支持 random（已移除 Optuna）")
    parser.add_argument("--stage", type=str, default="auto", choices=["auto", "B", "C", "final"], help="选择阶段：auto 先执行B后自动进入C；B 粗随机搜索；C 精调；final 最终复现")
    parser.add_argument("--trials", type=int, default=60, help="阶段B每任务试验数")
    parser.add_argument("--epochs", type=int, default=3, help="阶段B训练轮数（建议 3）")
    parser.add_argument("--seed_base", type=int, default=0, help="基础种子")
    # 阶段C/最终复现参数（入口预置）
    parser.add_argument("--epochs_refine", type=int, default=10, help="阶段C精调的训练轮数（默认10，启用早停）")
    parser.add_argument("--final_seeds", nargs="+", type=int, default=[0, 1, 2], help="最终复现的 seeds 列表")
    parser.add_argument("--no_redirect", action="store_true", help="禁用训练过程的日志重定向（打印到控制台）")
    args_cli = parser.parse_args()
    # 控制日志重定向：--no_redirect 时不重定向到文件
    if getattr(args_cli, "no_redirect", False):
        os.environ["HPO_NO_REDIRECT"] = "1"
    else:
        os.environ.pop("HPO_NO_REDIRECT", None)
    # 关键运行信息打印
    print(f"[HPO] stage={args_cli.stage} backend={args_cli.search_backend} tasks={args_cli.tasks} trials={args_cli.trials} epochs={args_cli.epochs} epochs_refine={args_cli.epochs_refine}")
    sys.stdout.flush()

    # 根目录与 experiments 目录
    proj_root = Path(__file__).resolve().parent.parent
    exp_root = proj_root / "experiments" / f"hpo_{now_tag()}"
    ensure_dir(exp_root)

    # 基线文件路径（与用户命令一致的数据文件）
    dataset_dir = Path(__file__).resolve().parent / "dataset1"
    file_map = {
        "LDA": {"in_file": str(dataset_dir / "LDA.edgelist"), "neg_file": str(dataset_dir / "non_LDA.edgelist")},
        "MDA": {"in_file": str(dataset_dir / "MDA.edgelist"), "neg_file": str(dataset_dir / "non_MDA.edgelist")},
        "LMI": {"in_file": str(dataset_dir / "LMI.edgelist"), "neg_file": str(dataset_dir / "non_LMI.edgelist")},
    }

    # 阶段B：粗随机搜索（支持 auto 先执行B）
    if args_cli.stage in ("B", "auto"):
        print(f"[HPO][B] 开始阶段B（随机搜索），任务集合: {args_cli.tasks}")
        for task in args_cli.tasks:
            # 若选择 Optuna 且可用：该分支原为阶段C精调逻辑，现已禁用以避免混淆（使用下方 Optuna 阶段B分支）
            if (str(getattr(args_cli, "search_backend", "optuna")).lower() == "optuna") and _OPTUNA_AVAILABLE and False:
                print(f"[HPO][B][{task}] 使用Optuna进行粗随机搜索（Optuna）")
                # 兜底获取最近的阶段B输出目录，避免 latest_exp 未定义导致异常
                try:
                    _exp_dir_root = Path(__file__).resolve().parent.parent / "experiments"
                    _hpo_dirs = sorted([p for p in _exp_dir_root.glob("hpo_*") if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                    _latest_exp = _hpo_dirs[0] if _hpo_dirs else None
                except Exception:
                    _latest_exp = None
                if _latest_exp is None:
                    print(f"[HPO][C][{task}] 未找到阶段B输出目录 experiments/hpo_*，跳过该任务")
                    continue
                src_task_dir = _latest_exp / task
                if not src_task_dir.exists():
                    print(f"[HPO][C][{task}] 未找到来源目录: {src_task_dir}，跳过该任务")
                    continue
                top_csv = src_task_dir / f"configs_top10_task_{task}.csv"
                if not top_csv.exists():
                    print(f"[HPO][C][{task}] 未找到 top10 CSV: {top_csv}，请确认阶段B已生成")
                    continue

                dst_task_dir = exp_root / task
                ensure_dir(dst_task_dir)

                # 读取 top10 配置作为局部精调的基准
                lines = top_csv.read_text(encoding="utf-8").strip().splitlines()
                header = lines[0].split(",")
                try:
                    cfg_idx = header.index("config_json")
                except ValueError:
                    print(f"[HPO][C][{task}] CSV 不包含 config_json 列，跳过")
                    continue
                base_cfgs = []
                for i in range(1, min(11, len(lines))):
                    row = lines[i].split(",", maxsplit=len(header)-1)
                    try:
                        base_cfgs.append(json.loads(row[cfg_idx]))
                    except Exception:
                        pass
                if not base_cfgs:
                    print(f"[HPO][C][{task}] 无可用top10配置，跳过")
                    continue

                # 数据文件映射
                dataset_dir = Path(__file__).resolve().parent / "dataset1"
                in_file = str(dataset_dir / f"{task}.edgelist")
                neg_file = str(dataset_dir / f"non_{task}.edgelist")

                def _clip(v, lo, hi):
                    return max(lo, min(hi, v))

                history_rows_opt: List[List[Any]] = []
                trial_metrics_opt: List[Dict[str, Any]] = []

                def _objective(trial: "optuna.trial.Trial") -> float:
                    # 选择一个基准配置索引
                    base_idx = int(trial.suggest_int("base_idx", 0, len(base_cfgs) - 1))
                    base = dict(base_cfgs[base_idx])

                    # 邻域采样（离散微调）
                    lr_scale = float(trial.suggest_categorical("lr_scale", [0.75, 1.0, 1.25]))
                    dropout_delta = float(trial.suggest_categorical("dropout_delta", [-0.05, 0.0, 0.05]))
                    alpha_scale = float(trial.suggest_categorical("alpha_scale", [0.75, 1.0, 1.25]))
                    beta_scale = float(trial.suggest_categorical("beta_scale", [0.75, 1.0, 1.25]))
                    gamma_choice = float(trial.suggest_categorical("gamma_choice", [base.get("gamma", 0.0), 0.1]))
                    proj_choice = trial.suggest_categorical("proj_choice", ["keep", "hidden2"])

                    cfg = dict(base)
                    cfg["lr"] = float(_clip(base["lr"] * lr_scale, 1e-5, 5e-2))
                    cfg["dropout"] = float(_clip(base.get("dropout", 0.0) + dropout_delta, 0.0, 0.5))
                    cfg["alpha"] = float(_clip(base.get("alpha", 1.0) * alpha_scale, 0.1, 3.0))
                    cfg["beta"] = float(_clip(base.get("beta", 0.5) * beta_scale, 0.05, 2.0))
                    cfg["gamma"] = float(gamma_choice)
                    if proj_choice == "hidden2":
                        cfg["proj_dim"] = int(base.get("hidden2", cfg.get("proj_dim", 64)))
                    else:
                        cfg["proj_dim"] = int(cfg.get("proj_dim", base.get("proj_dim", base.get("hidden2", 64))))

                    # 强制精调设定：online + mgraph
                    cfg["augment_mode"] = "online"
                    cfg["adv_mode"] = "mgraph"
                    cfg["adv_on_moco"] = True

                    trial_id = int(trial.number + 1)
                    metrics, meta = run_trial_with_retry(task, trial_id, cfg, args_cli.epochs_refine, dst_task_dir, in_file, neg_file)

                    # 历史记录（含 trial_number 与 seed）
                    history_rows_opt.append([
                        meta.get("run_name"),
                        json.dumps(cfg, ensure_ascii=False),
                        metrics["AUPRC_mean"], metrics["AUPRC_std"],
                        metrics["AUROC_mean"], metrics["AUROC_std"],
                        metrics["F1_mean"], metrics["F1_std"],
                        metrics["Loss_mean"], metrics["Loss_std"],
                        meta.get("time_s"), meta.get("log_path"), meta.get("retry"), meta.get("error"),
                        f"trial_number={trial.number}",
                        f"seed={trial_id}"
                    ])
                    tm = dict(metrics)
                    tm["run_name"] = meta.get("run_name")
                    tm["config_json"] = json.dumps(cfg, ensure_ascii=False)
                    tm["time_s"] = meta.get("time_s")
                    tm["log_path"] = meta.get("log_path")
                    trial_metrics_opt.append(tm)
                    # 控制台 trial 摘要打印（阶段B-Optuna）
                    print(f"[HPO][B][{task}][trial {trial_id}] AUPRC={metrics['AUPRC_mean']:.4f} AUROC={metrics['AUROC_mean']:.4f} F1={metrics['F1_mean']:.4f} time_s={meta.get('time_s')} retry={meta.get('retry')} error={meta.get('error')}")
                    return float(metrics["AUPRC_mean"])

                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=int(args_cli.seed_base)),
                    pruner=optuna.pruners.MedianPruner()
                )
                study.optimize(_objective, n_trials=int(args_cli.trials))

                # 写精调历史CSV（含trial_number与seed）
                write_csv(
                    dst_task_dir / f"opt_history_refine_task_{task}.csv",
                    header=[
                        "run_name", "config_json",
                        "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                        "time_s", "log_path", "retry", "error", "trial_number", "seed"
                    ],
                    rows=history_rows_opt
                )

                # Top3与多seed复验
                top3 = sort_and_top(trial_metrics_opt, topn=3)
                write_csv(
                    dst_task_dir / f"configs_top3_refine_task_{task}.csv",
                    header=[
                        "run_name", "config_json",
                        "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                        "time_s", "log_path"
                    ],
                    rows=[[m["run_name"], m["config_json"], m["AUPRC_mean"], m["AUPRC_std"], m["AUROC_mean"], m["AUROC_std"], m["F1_mean"], m["F1_std"], m["Loss_mean"], m["Loss_std"], m["time_s"], m["log_path"]] for m in top3]
                )

                seeds = list(args_cli.final_seeds)
                final_payload = {"task": task, "top3_final": []}
                for m in top3:
                    cfg_final = json.loads(m["config_json"])
                    run_stats = []
                    for sd in seeds:
                        trial_id = len(history_rows_opt) + 1
                        metrics_sd, meta_sd = run_trial_with_retry(task, trial_id, cfg_final, args_cli.epochs_refine, dst_task_dir, in_file, neg_file, fixed_seed=int(sd))
                        run_stats.append({
                            "seed": int(sd),
                            "metrics": metrics_sd,
                            "run_name": meta_sd.get("run_name"),
                            "log_path": meta_sd.get("log_path")
                        })
                    def agg(key):
                        vals = [float(rs["metrics"][key]) for rs in run_stats]
                        return float(np.mean(vals)), float(np.std(vals))
                    final_payload["top3_final"].append({
                        "config": cfg_final,
                        "seeds": seeds,
                        "aggregate": {
                            "AUPRC_mean": agg("AUPRC_mean")[0], "AUPRC_std": agg("AUPRC_mean")[1],
                            "AUROC_mean": agg("AUROC_mean")[0], "AUROC_std": agg("AUROC_mean")[1],
                            "F1_mean": agg("F1_mean")[0], "F1_std": agg("F1_mean")[1]
                        },
                        "runs": run_stats
                    })
                (dst_task_dir / "best_configs_final.json").write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                (dst_task_dir / f"summary_task_{task}.md").write_text(
                    f"# 阶段C精调与复验结果（Optuna）\n- 基于最近一次阶段B的top10进行Optuna局部精调（epochs={args_cli.epochs_refine}，online+mgraph）。\n- 对精调Top3进行多个seed的完整5-fold复验，并聚合mean/std。\n来源：{str(src_task_dir)}",
                    encoding="utf-8"
                )
                # 完成该任务的Optuna精调，进入下一个任务
                continue
            exp_task_dir = exp_root / task
            ensure_dir(exp_task_dir)

            rng = np.random.default_rng(args_cli.seed_base + hash(task) % 10000)

            # Optuna 已移除：禁用该分支，阶段B固定为随机搜索
            if False and (str(args_cli.search_backend).lower() == "optuna"):
                def _build_cfg_from_trial(trial) -> Dict[str, Any]:
                    # 离散空间与条件与 sample_config 对齐
                    embed_dim = trial.suggest_categorical("dimensions", [32, 64, 128, 256])
                    hidden1_choices = [max(32, embed_dim // 2), embed_dim, min(2 * embed_dim, 512)]
                    hidden2_choices = [max(16, embed_dim // 4), max(32, embed_dim // 2), embed_dim]
                    cfg = {
                        "task_type": task,
                        "dimensions": int(embed_dim),
                        "hidden1": int(trial.suggest_categorical("hidden1", hidden1_choices)),
                        "hidden2": int(trial.suggest_categorical("hidden2", hidden2_choices)),
                        "decoder1": int(trial.suggest_categorical("decoder1", [256, 512])),
                        "lr": float(trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3])),
                        "weight_decay": float(trial.suggest_categorical("weight_decay", [0.0, 5e-5, 5e-4])),
                        "dropout": float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])),
                        "base_batch": 32,
                        "lr_warmup_epochs": int(trial.suggest_categorical("lr_warmup_epochs", [0, 2, 3, 5])),
                        "lr_min_factor": float(trial.suggest_categorical("lr_min_factor", [0.1, 0.2, 0.3])),
                        "early_stop_patience": int(trial.suggest_categorical("early_stop_patience", [0, 3, 5])),
                        "early_stop_min_delta": float(trial.suggest_categorical("early_stop_min_delta", [0.0, 1e-4, 5e-4])),
                        "batch": int(trial.suggest_categorical("batch", [16, 32, 64, 128])),
                        "moco_queue": int(trial.suggest_categorical("moco_queue", [1024, 4096, 8192])),
                        "moco_momentum": float(trial.suggest_categorical("moco_momentum", [0.95, 0.99, 0.999])),
                        "moco_t": float(trial.suggest_categorical("moco_t", [0.07, 0.1, 0.2])),
                        "proj_dim": trial.suggest_categorical("proj_dim", [32, 64, 128, None]),
                        "augment": str(trial.suggest_categorical("augment", ["none", "random_permute_features", "add_noise", "attribute_mask", "noise_then_mask"])),
                        "augment_mode": "static",
                        "noise_std": float(trial.suggest_categorical("noise_std", [0.005, 0.01, 0.02])),
                        "mask_rate": float(trial.suggest_categorical("mask_rate", [0.05, 0.1, 0.2])),
                        "alpha": float(trial.suggest_categorical("alpha", [0.5, 1.0, 2.0])),
                        "beta": float(trial.suggest_categorical("beta", [0.1, 0.5, 1.0])),
                        "gamma": float(trial.suggest_categorical("gamma", [0.0, 0.1, 0.5])),
                        "num_views": int(trial.suggest_categorical("num_views", [2, 3])),
                        "adv_mode": "none",
                        "adv_on_moco": False,
                        "queue_warmup_steps": int(trial.suggest_categorical("queue_warmup_steps", [0, 200])),
                        "gat_heads": int(trial.suggest_categorical("gat_heads", [2, 4])),
                        "fusion_heads": int(trial.suggest_categorical("fusion_heads", [2, 4])),
                    }
                    if cfg["augment"] in ("none", "null", ""):
                        cfg["noise_std"] = 0.0
                        cfg["mask_rate"] = 0.0
                    if (cfg["proj_dim"] is None) or (cfg["proj_dim"] is not None and int(cfg["proj_dim"]) <= 0):
                        cfg["proj_dim"] = int(cfg["hidden2"])
                    return cfg

                history_rows_opt: List[List[Any]] = []
                trial_metrics_opt: List[Dict[str, Any]] = []

                def _objective(trial: "optuna.trial.Trial") -> float:
                    # 使用 trial.number 作为 trial_id；记录 trial_number 与 seed
                    cfg = _build_cfg_from_trial(trial)
                    trial_id = int(trial.number + 1)
                    metrics, meta = run_trial_with_retry(task, trial_id, cfg, args_cli.epochs, exp_task_dir, file_map[task]["in_file"], file_map[task]["neg_file"])
                    # 记录历史
                    history_rows_opt.append([
                        meta.get("run_name"),
                        json.dumps(cfg, ensure_ascii=False),
                        metrics["AUPRC_mean"], metrics["AUPRC_std"],
                        metrics["AUROC_mean"], metrics["AUROC_std"],
                        metrics["F1_mean"], metrics["F1_std"],
                        metrics["Loss_mean"], metrics["Loss_std"],
                        meta.get("time_s"), meta.get("log_path"), meta.get("retry"), meta.get("error"),
                        f"trial_number={trial.number}",  # 额外记录 trial_number
                        f"seed={trial_id}"  # 与 derive_seed 相关联（外层以 trial_id 派生）
                    ])
                    tm = dict(metrics)
                    tm["run_name"] = meta.get("run_name")
                    tm["config_json"] = json.dumps(cfg, ensure_ascii=False)
                    tm["time_s"] = meta.get("time_s")
                    tm["log_path"] = meta.get("log_path")
                    trial_metrics_opt.append(tm)
                    # 以 AUPRC_mean 为优化目标
                    return float(metrics["AUPRC_mean"])

                study = optuna.create_study(direction="maximize",
                                            sampler=optuna.samplers.TPESampler(seed=int(args_cli.seed_base)),
                                            pruner=optuna.pruners.MedianPruner())
                study.optimize(_objective, n_trials=int(args_cli.trials))

                # 写历史 CSV（含额外两列）
                write_csv(
                    exp_task_dir / f"opt_history_task_{task}.csv",
                    header=[
                        "run_name", "config_json",
                        "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                        "time_s", "log_path", "retry", "error", "trial_number", "seed"
                    ],
                    rows=history_rows_opt
                )

                # Top10
                top10_opt = sort_and_top(trial_metrics_opt, topn=10)
                top_rows = [[
                    m["run_name"],
                    m["config_json"],
                    m["AUPRC_mean"], m["AUPRC_std"],
                    m["AUROC_mean"], m["AUROC_std"],
                    m["F1_mean"], m["F1_std"],
                    m["Loss_mean"], m["Loss_std"],
                    m["time_s"], m["log_path"]
                ] for m in top10_opt]
                write_csv(
                    exp_task_dir / f"configs_top10_task_{task}.csv",
                    header=[
                        "run_name", "config_json",
                        "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                        "time_s", "log_path"
                    ],
                    rows=top_rows
                )
                # 控制台打印 top10 汇总（阶段B-Optuna）
                print(f"[HPO][B][{task}] Top10 生成（Optuna）:")
                for m in top10_opt:
                    print(f"  - {m['run_name']}: AUPRC={m['AUPRC_mean']:.4f} AUROC={m['AUROC_mean']:.4f} F1={m['F1_mean']:.4f}")

                # best_configs_final.json（取 top3）
                best3 = top10_opt[:3]
                best_payload = {
                    "task": task,
                    "top3": [{
                        "run_name": m["run_name"],
                        "config": json.loads(m["config_json"]),
                        "metrics": {
                            "AUPRC_mean": m["AUPRC_mean"], "AUROC_mean": m["AUROC_mean"], "F1_mean": m["F1_mean"]
                        }
                    } for m in best3],
                    "note": "阶段B（Optuna）搜索结果，阶段C将围绕这些配置做精调（epochs=10，online+mgraph）"
                }
                (exp_task_dir / "best_configs_final.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                # 初版报告
                (exp_task_dir / f"summary_task_{task}.md").write_text("# 阶段B（Optuna）结果\n\n- 已生成 top10 与历史CSV；待阶段C与最终复现补充。\n", encoding="utf-8")

                # 当前任务完成，继续下一个
                continue

            history_rows: List[List[Any]] = []
            trial_metrics: List[Dict[str, Any]] = []

            for trial_id in range(1, args_cli.trials + 1):
                cfg = sample_config(task, rng)
                metrics, meta = run_trial_with_retry(task, trial_id, cfg, args_cli.epochs, exp_task_dir, file_map[task]["in_file"], file_map[task]["neg_file"])

                # 记录历史行
                history_rows.append([
                    meta.get("run_name"),
                    json.dumps(cfg, ensure_ascii=False),
                    metrics["AUPRC_mean"], metrics["AUPRC_std"],
                    metrics["AUROC_mean"], metrics["AUROC_std"],
                    metrics["F1_mean"], metrics["F1_std"],
                    metrics["Loss_mean"], metrics["Loss_std"],
                    meta.get("time_s"), meta.get("log_path"), meta.get("retry"), meta.get("error")
                ])
                # trial_metrics 用于排序
                tm = dict(metrics)
                tm["run_name"] = meta.get("run_name")
                tm["config_json"] = json.dumps(cfg, ensure_ascii=False)
                tm["time_s"] = meta.get("time_s")
                tm["log_path"] = meta.get("log_path")
                trial_metrics.append(tm)
                # 控制台 trial 摘要打印（阶段B-随机）
                print(f"[HPO][B][{task}][trial {trial_id}] AUPRC={metrics['AUPRC_mean']:.4f} AUROC={metrics['AUROC_mean']:.4f} F1={metrics['F1_mean']:.4f} time_s={meta.get('time_s')} retry={meta.get('retry')} error={meta.get('error')}")

            # 写历史 CSV
            write_csv(
                exp_task_dir / f"opt_history_task_{task}.csv",
                header=[
                    "run_name", "config_json",
                    "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                    "time_s", "log_path", "retry", "error"
                ],
                rows=history_rows
            )

            # Top10
            top10 = sort_and_top(trial_metrics, topn=10)
            top_rows = [[
                m["run_name"],
                m["config_json"],
                m["AUPRC_mean"], m["AUPRC_std"],
                m["AUROC_mean"], m["AUROC_std"],
                m["F1_mean"], m["F1_std"],
                m["Loss_mean"], m["Loss_std"],
                m["time_s"], m["log_path"]
            ] for m in top10]
            write_csv(
                exp_task_dir / f"configs_top10_task_{task}.csv",
                header=[
                    "run_name", "config_json",
                    "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                    "time_s", "log_path"
                ],
                rows=top_rows
            )
            # 控制台打印 top10 汇总（阶段B-随机）
            print(f"[HPO][B][{task}] Top10 生成（随机/回退）:")
            for m in top10:
                print(f"  - {m['run_name']}: AUPRC={m['AUPRC_mean']:.4f} AUROC={m['AUROC_mean']:.4f} F1={m['F1_mean']:.4f}")

            # 初版 best_configs_final.json（取 top3）
            best3 = top10[:3]
            best_payload = {
                "task": task,
                "top3": [{
                    "run_name": m["run_name"],
                    "config": json.loads(m["config_json"]),
                    "metrics": {
                        "AUPRC_mean": m["AUPRC_mean"], "AUROC_mean": m["AUROC_mean"], "F1_mean": m["F1_mean"]
                    },
                    "repro_cli": f"python hyperparameter-tuning/main.py --file {file_map[task]['in_file']} --neg_sample {file_map[task]['neg_file']} --validation_type 5-cv1 --task_type {task} --feature_type one_hot --similarity_threshold 0.5 --embed_dim {json.loads(m['config_json'])['dimensions']} --learning_rate {json.loads(m['config_json'])['lr']} --weight_decay {json.loads(m['config_json'])['weight_decay']} --epochs {args_cli.epochs} --alpha {json.loads(m['config_json'])['alpha']} --beta {json.loads(m['config_json'])['beta']} --gamma {json.loads(m['config_json'])['gamma']}"
                } for m in best3],
                "note": "阶段B随机搜索结果，阶段C将围绕这些配置做精调（epochs=20，启用online增强与mgraph对抗）"
            }
            (exp_task_dir / "best_configs_final.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            # 预留报告文件
            (exp_task_dir / f"summary_task_{task}.md").write_text("# 阶段B结果\n\n- 已生成 top10 与历史CSV；待阶段C与最终复现补充。\n", encoding="utf-8")

        print(f"[HPO] 阶段B完成。输出位于: {exp_root}")
        # 若选择自动模式，直接进入阶段C（当前exp_root）执行邻域小网格精调与top3多seed复验
        if args_cli.stage == "auto":
            print("[HPO][AUTO] 阶段B完成，自动进入阶段C（局部精调 + 多seed复验）")
            latest_exp = exp_root
            exp_root_c = proj_root / "experiments" / f"hpo_{now_tag()}"
            ensure_dir(exp_root_c)
            # 邻域生成器
            def _clip(v, lo, hi):
                return max(lo, min(hi, v))
            def generate_refine_neighbors(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
                neighbors: List[Dict[str, Any]] = []
                for f in [0.75, 1.0, 1.25]:
                    for dd in [-0.05, 0.0, 0.05]:
                        cfg2 = dict(base_cfg)
                        cfg2["lr"] = float(_clip(base_cfg["lr"] * f, 1e-5, 5e-2))
                        cfg2["dropout"] = float(_clip(base_cfg["dropout"] + dd, 0.0, 0.5))
                        cfg2["proj_dim"] = int(cfg2.get("proj_dim") or cfg2["hidden2"])
                        if cfg2["proj_dim"] != int(cfg2["hidden2"]):
                            neighbors.append(dict({**cfg2, "proj_dim": int(cfg2["hidden2"])}))
                        for a_scale in [0.75, 1.0, 1.25]:
                            for b_scale in [0.75, 1.0, 1.25]:
                                cfg3 = dict(cfg2)
                                cfg3["alpha"] = float(_clip(base_cfg["alpha"] * a_scale, 0.1, 3.0))
                                cfg3["beta"] = float(_clip(base_cfg["beta"] * b_scale, 0.05, 2.0))
                                cfg3["gamma"] = float(base_cfg["gamma"] if base_cfg["gamma"] > 0 else 0.1)
                                neighbors.append(cfg3)
                seen = set()
                uniq = []
                keys = ["lr", "dropout", "proj_dim", "alpha", "beta", "gamma"]
                for c in neighbors:
                    t = tuple(c[k] for k in keys)
                    if t in seen:
                        continue
                    seen.add(t)
                    c["augment_mode"] = "online"
                    c["adv_mode"] = "mgraph"
                    c["adv_on_moco"] = True
                    uniq.append(c)
                return uniq[:50]
            # 遍历任务执行C
            for task in args_cli.tasks:
                # AUTO分支同样兜底获取最近的阶段B输出目录，避免 latest_exp 异常
                try:
                    _exp_dir_root = Path(__file__).resolve().parent.parent / "experiments"
                    _hpo_dirs = sorted([p for p in _exp_dir_root.glob("hpo_*") if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                    _latest_exp_auto = latest_exp if 'latest_exp' in locals() else (_hpo_dirs[0] if _hpo_dirs else None)
                except Exception:
                    _latest_exp_auto = latest_exp if 'latest_exp' in locals() else None
                if _latest_exp_auto is None:
                    print(f"[HPO][AUTO][C][{task}] 未找到阶段B输出目录 experiments/hpo_*，跳过该任务")
                    continue
                src_task_dir = _latest_exp_auto / task
                if not src_task_dir.exists():
                    print(f"[HPO][AUTO][C][{task}] 未找到来源目录: {src_task_dir}，跳过该任务")
                    continue
                top_csv = src_task_dir / f"configs_top10_task_{task}.csv"
                if not top_csv.exists():
                    print(f"[HPO][C][{task}] 未找到 top10 CSV: {top_csv}，请确认阶段B已生成")
                    continue
                dst_task_dir = exp_root_c / task
                ensure_dir(dst_task_dir)
                lines = top_csv.read_text(encoding="utf-8").strip().splitlines()
                header = lines[0].split(",")
                try:
                    cfg_idx = header.index("config_json")
                except ValueError:
                    print(f"[HPO][C][{task}] CSV 不包含 config_json 列，跳过")
                    continue
                history_rows: List[List[Any]] = []
                trial_metrics: List[Dict[str, Any]] = []
                dataset_dir = Path(__file__).resolve().parent / "dataset1"
                in_file = str(dataset_dir / f"{task}.edgelist")
                neg_file = str(dataset_dir / f"non_{task}.edgelist")
                trial_id = 0
                for i in range(1, min(11, len(lines))):
                    row = lines[i].split(",", maxsplit=len(header)-1)
                    cfg_json = row[cfg_idx]
                    try:
                        cfg = json.loads(cfg_json)
                    except Exception as _e:
                        print(f"[HPO][C][{task}] 解析第{i}行配置失败：{_e}，跳过")
                        continue
                    neighbors = generate_refine_neighbors(cfg)
                    for cfg_refine in neighbors:
                        trial_id += 1
                        metrics, meta = run_trial_with_retry(task, trial_id, cfg_refine, args_cli.epochs_refine, dst_task_dir, in_file, neg_file)
                        history_rows.append([
                            meta.get("run_name"),
                            json.dumps(cfg_refine, ensure_ascii=False),
                            metrics["AUPRC_mean"], metrics["AUPRC_std"],
                            metrics["AUROC_mean"], metrics["AUROC_std"],
                            metrics["F1_mean"], metrics["F1_std"],
                            metrics["Loss_mean"], metrics["Loss_std"],
                            meta.get("time_s"), meta.get("log_path"), meta.get("retry"), meta.get("error")
                        ])
                        tm = dict(metrics)
                        tm["run_name"] = meta.get("run_name")
                        tm["config_json"] = json.dumps(cfg_refine, ensure_ascii=False)
                        tm["time_s"] = meta.get("time_s")
                        tm["log_path"] = meta.get("log_path")
                        trial_metrics.append(tm)
                        # 控制台 trial 摘要打印（阶段C-auto）
                        print(f"[HPO][AUTO][C][{task}][trial {trial_id}] AUPRC={metrics['AUPRC_mean']:.4f} AUROC={metrics['AUROC_mean']:.4f} F1={metrics['F1_mean']:.4f} time_s={meta.get('time_s')} retry={meta.get('retry')} error={meta.get('error')}")
                # 写历史CSV与合并B+C
                write_csv(
                    dst_task_dir / f"opt_history_refine_task_{task}.csv",
                    header=[
                        "run_name", "config_json",
                        "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                        "time_s", "log_path", "retry", "error"
                    ],
                    rows=history_rows
                )
                try:
                    hist_b = src_task_dir / f"opt_history_task_{task}.csv"
                    if hist_b.exists():
                        with hist_b.open("a", encoding="utf-8") as fa:
                            for r in history_rows:
                                fa.write(",".join(str(x) for x in r) + "\n")
                except Exception:
                    pass
                # Top3与多seed复验
                top3 = sort_and_top(trial_metrics, topn=3)
                write_csv(
                    dst_task_dir / f"configs_top3_refine_task_{task}.csv",
                    header=[
                        "run_name", "config_json",
                        "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                        "time_s", "log_path"
                    ],
                    rows=[[m["run_name"], m["config_json"], m["AUPRC_mean"], m["AUPRC_std"], m["AUROC_mean"], m["AUROC_std"], m["F1_mean"], m["F1_std"], m["Loss_mean"], m["Loss_std"], m["time_s"], m["log_path"]] for m in top3]
                )
                # 控制台打印 Top3 汇总（阶段C-auto）
                print(f"[HPO][AUTO][C][{task}] Top3 精调结果:")
                for m in top3:
                    print(f"  - {m['run_name']}: AUPRC={m['AUPRC_mean']:.4f} AUROC={m['AUROC_mean']:.4f} F1={m['F1_mean']:.4f}")
                seeds = [0, 1, 2]
                final_payload = {"task": task, "top3_final": []}
                for m in top3:
                    cfg_final = json.loads(m["config_json"])
                    run_stats = []
                    for sd in seeds:
                        trial_id += 1
                        metrics_sd, meta_sd = run_trial_with_retry(task, trial_id, cfg_final, args_cli.epochs_refine, dst_task_dir, in_file, neg_file, fixed_seed=int(sd))
                        run_stats.append({
                            "seed": int(sd),
                            "metrics": metrics_sd,
                            "run_name": meta_sd.get("run_name"),
                            "log_path": meta_sd.get("log_path")
                        })
                    def agg(key):
                        vals = [float(rs["metrics"][key]) for rs in run_stats]
                        return float(np.mean(vals)), float(np.std(vals))
                    final_payload["top3_final"].append({
                        "config": cfg_final,
                        "seeds": seeds,
                        "aggregate": {
                            "AUPRC_mean": agg("AUPRC_mean")[0], "AUPRC_std": agg("AUPRC_mean")[1],
                            "AUROC_mean": agg("AUROC_mean")[0], "AUROC_std": agg("AUROC_mean")[1],
                            "F1_mean": agg("F1_mean")[0], "F1_std": agg("F1_mean")[1]
                        },
                        "runs": run_stats
                    })
                (dst_task_dir / "best_configs_final.json").write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                (dst_task_dir / f"summary_task_{task}.md").write_text(
                    f"# 阶段C精调与复验结果（auto）\n- 基于当前阶段B的top10进行邻域小网格精调（online+mgraph，epochs={args_cli.epochs_refine}，含早停）。\n- 对精调Top3进行3个不同seed的完整5-fold复验，并聚合mean/std。\n来源：{str(src_task_dir)}",
                    encoding="utf-8"
                )
            print(f"[HPO][AUTO] 阶段C完成。输出位于: {exp_root_c}")
            return

    elif args_cli.stage == "C":
        # 自动精调：读取最近一次阶段B的 top10 配置并运行 epochs_refine，强制 online + mgraph
        proj_root = Path(__file__).resolve().parent.parent
        exp_dir_root = proj_root / "experiments"
        # 找到最近的 hpo_* 目录
        hpo_dirs = sorted([p for p in exp_dir_root.glob("hpo_*") if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        if not hpo_dirs:
            print("[HPO][C] 未发现阶段B输出目录 experiments/hpo_*，请先运行 --stage B")
            return
        latest_exp = hpo_dirs[0]
        # 选择最近一个包含 top10 CSV 的阶段B目录，避免误选到当前新建的空目录
        latest_exp = None
        for cand in hpo_dirs:
            ok = True
            for t in args_cli.tasks:
                if not (cand / t / f"configs_top10_task_{t}.csv").exists():
                    ok = False
                    break
            if ok:
                latest_exp = cand
                break
        if latest_exp is None:
            print("[HPO][C] 未找到包含 top10 CSV 的阶段B输出目录，请先运行 --stage B")
            return
        print(f"[HPO][C] 使用最近一次阶段B目录: {latest_exp}")

        # 精调输出根目录
        exp_root = proj_root / "experiments" / f"hpo_{now_tag()}"
        ensure_dir(exp_root)

        print(f"[HPO][C] 开始阶段C精调（online+mgraph），任务集合: {args_cli.tasks}")
        for task in args_cli.tasks:
            # 为当前任务选择最近一个包含该任务top10 CSV的阶段B目录
            src_task_dir = None
            for cand in hpo_dirs:
                tcsv = cand / task / f"configs_top10_task_{task}.csv"
                if tcsv.exists():
                    src_task_dir = cand / task
                    top_csv = tcsv
                    break
            if src_task_dir is None:
                print(f"[HPO][C][{task}] 未找到包含该任务top10 CSV的阶段B目录，跳过该任务")
                continue

            # 目标输出目录
            dst_task_dir = exp_root / task
            ensure_dir(dst_task_dir)

            # 读取 top10 CSV 并解析配置
            lines = top_csv.read_text(encoding="utf-8").strip().splitlines()
            header = lines[0].split(",")
            # 约定 header 中包含 "config_json"
            try:
                cfg_idx = header.index("config_json")
            except ValueError:
                print(f"[HPO][C][{task}] CSV 不包含 config_json 列，跳过")
                continue

            history_rows: List[List[Any]] = []
            trial_metrics: List[Dict[str, Any]] = []

            # 数据文件映射（与阶段B一致）
            dataset_dir = Path(__file__).resolve().parent / "dataset1"
            in_file = str(dataset_dir / f"{task}.edgelist")
            neg_file = str(dataset_dir / f"non_{task}.edgelist")

            trial_id = 0
            for i in range(1, min(11, len(lines))):  # 跳过表头，最多10条
                row = lines[i].split(",", maxsplit=len(header)-1)
                cfg_json = row[cfg_idx]
                try:
                    cfg = json.loads(cfg_json)
                except Exception as _e:
                    print(f"[HPO][C][{task}] 解析第{i}行配置失败：{_e}，跳过")
                    continue

                # 强制覆盖为精调设定：online + mgraph
                cfg_refine = dict(cfg)
                cfg_refine["augment_mode"] = "online"
                cfg_refine["adv_mode"] = "mgraph"
                cfg_refine["adv_on_moco"] = True  # 如需关闭对增强视图的对抗，可改为 False
                # 保持其余超参（含 num_views、queue_warmup_steps 等）

                trial_id += 1
                metrics, meta = run_trial_with_retry(task, trial_id, cfg_refine, args_cli.epochs_refine, dst_task_dir, in_file, neg_file)

                # 记录历史行
                history_rows.append([
                    meta.get("run_name"),
                    json.dumps(cfg_refine, ensure_ascii=False),
                    metrics["AUPRC_mean"], metrics["AUPRC_std"],
                    metrics["AUROC_mean"], metrics["AUROC_std"],
                    metrics["F1_mean"], metrics["F1_std"],
                    metrics["Loss_mean"], metrics["Loss_std"],
                    meta.get("time_s"), meta.get("log_path"), meta.get("retry"), meta.get("error")
                ])
                # trial_metrics 用于排序
                tm = dict(metrics)
                tm["run_name"] = meta.get("run_name")
                tm["config_json"] = json.dumps(cfg_refine, ensure_ascii=False)
                tm["time_s"] = meta.get("time_s")
                tm["log_path"] = meta.get("log_path")
                trial_metrics.append(tm)
                # 控制台 trial 摘要打印（阶段C-显式）
                print(f"[HPO][C][{task}][trial {trial_id}] AUPRC={metrics['AUPRC_mean']:.4f} AUROC={metrics['AUROC_mean']:.4f} F1={metrics['F1_mean']:.4f} time_s={meta.get('time_s')} retry={meta.get('retry')} error={meta.get('error')}")

            # 写历史 CSV（精调）
            write_csv(
                dst_task_dir / f"opt_history_refine_task_{task}.csv",
                header=[
                    "run_name", "config_json",
                    "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                    "time_s", "log_path", "retry", "error"
                ],
                rows=history_rows
            )

            # Top3（精调）
            top3 = sort_and_top(trial_metrics, topn=3)
            top_rows = [[
                m["run_name"],
                m["config_json"],
                m["AUPRC_mean"], m["AUPRC_std"],
                m["AUROC_mean"], m["AUROC_std"],
                m["F1_mean"], m["F1_std"],
                m["Loss_mean"], m["Loss_std"],
                m["time_s"], m["log_path"]
            ] for m in top3]
            write_csv(
                dst_task_dir / f"configs_top3_refine_task_{task}.csv",
                header=[
                    "run_name", "config_json",
                    "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                    "time_s", "log_path"
                ],
                rows=top_rows
            )
            # 控制台打印 Top3 汇总（阶段C-显式）
            print(f"[HPO][C][{task}] Top3 精调结果:")
            for m in top3:
                print(f"  - {m['run_name']}: AUPRC={m['AUPRC_mean']:.4f} AUROC={m['AUROC_mean']:.4f} F1={m['F1_mean']:.4f}")

            # 写精调摘要 JSON/MD
            best_payload = {
                "task": task,
                "top3_refine": [{
                    "run_name": m["run_name"],
                    "config": json.loads(m["config_json"]),
                    "metrics": {
                        "AUPRC_mean": m["AUPRC_mean"], "AUROC_mean": m["AUROC_mean"], "F1_mean": m["F1_mean"]
                    }
                } for m in top3],
                "note": f"阶段C自动精调（epochs={args_cli.epochs_refine}，online+mgraph），来源：{str(src_task_dir)}"
            }
            (dst_task_dir / "best_configs_refine.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            (dst_task_dir / f"summary_task_{task}.md").write_text(
                f"""# 阶段C精调结果

- 已基于最近一次阶段B的top10进行精调（online+mgraph）。
- 输出历史CSV与top3精调结果。
来源：{str(src_task_dir)}""",
                encoding="utf-8"
            )

        print(f"[HPO] 阶段C精调完成。输出位于: {exp_root}")

    elif args_cli.stage == "final":
        # 最终复现入口预置
        print("[HPO] 最终复现入口预置：按 seeds 列表对每任务 top3 配置运行完整 5-fold，写入 summary 与 artifacts。")
        print("暂未自动执行复现，以避免误覆盖实验文件。")


if __name__ == "__main__":
    print("HPO script started")
    main()