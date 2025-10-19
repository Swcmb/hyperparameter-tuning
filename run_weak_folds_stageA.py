#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段A 弱折对照实验批量运行脚本（Linux版）
功能增强：
✅ 日志保存（每次运行单独文件）
✅ 并行运行（自动轮询多GPU）
✅ 运行状态实时打印
"""

import os
import sys
import subprocess
import time
from itertools import product
from pathlib import Path
import torch

# === 基础路径与环境 ===
ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT / "hyperparameter-tuning"
PY = sys.executable or "python3"

# === 默认PGD参数 ===
EPS_LIST = [0.008, 0.012, 0.015]
STEPS_LIST = [3, 5]
ADV_MODE = "mgraph"
ADV_NORM = "linf"
ADV_RAND_INIT = "true"
ADV_PROJECT = "true"
ADV_ON_MOCO = "true"
ADV_WARMUP_END = 3

# === 基线设置（与当前项目一致） ===
BASELINE = {
    "feature_type": "one_hot",
    "batch": 64,
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "dropout": 0.05,
    "epochs": 10,
    "augment_mode": "online",
    "loss_ratio1": 0.45,
    "loss_ratio2": 0.55,
    "loss_ratio3": 0.5,
    "moco_queue": 4096,
    "moco_momentum": 0.99,
    "moco_t": 0.07,
    "proj_dim": None,
}

# === 阈值扫描与温度校准 ===
SCAN_CALIB = {
    "enable_threshold_scan": "true",
    "threshold_min": 0.35,
    "threshold_max": 0.65,
    "threshold_step": 0.01,
    "enable_temp_scaling": "true",
    "temp_grid_min": 0.5,
    "temp_grid_max": 3.0,
    "temp_grid_num": 26,
}

# === 任务列表 ===
TASKS = [
    ("LDA", "LDA_C_stageA"),
    ("LMI", "LMI_C_stageA"),
    ("MDA", "MDA_C_stageA"),
]

# === GPU设置 ===
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
MAX_PARALLEL = NUM_GPUS  # 每张GPU并行一个任务


def build_common_args(task_type, run_suffix):
    """拼接通用参数"""
    args = [
        "--task_type", task_type,
        "--run_name", run_suffix,
        "--feature_type", BASELINE["feature_type"],
        "--batch", str(BASELINE["batch"]),
        "--lr", str(BASELINE["lr"]),
        "--weight_decay", str(BASELINE["weight_decay"]),
        "--dropout", str(BASELINE["dropout"]),
        "--epochs", str(BASELINE["epochs"]),
        "--augment_mode", BASELINE["augment_mode"],
        "--moco_queue", str(BASELINE["moco_queue"]),
        "--moco_momentum", str(BASELINE["moco_momentum"]),
        "--moco_t", str(BASELINE["moco_t"]),
        "--loss_ratio1", str(BASELINE["loss_ratio1"]),
        "--loss_ratio2", str(BASELINE["loss_ratio2"]),
        "--loss_ratio3", str(BASELINE["loss_ratio3"]),
        "--enable_threshold_scan", SCAN_CALIB["enable_threshold_scan"],
        "--threshold_min", str(SCAN_CALIB["threshold_min"]),
        "--threshold_max", str(SCAN_CALIB["threshold_max"]),
        "--threshold_step", str(SCAN_CALIB["threshold_step"]),
        "--enable_temp_scaling", SCAN_CALIB["enable_temp_scaling"],
        "--temp_grid_min", str(SCAN_CALIB["temp_grid_min"]),
        "--temp_grid_max", str(SCAN_CALIB["temp_grid_max"]),
        "--temp_grid_num", str(SCAN_CALIB["temp_grid_num"]),
        "--adv_warmup_end", str(ADV_WARMUP_END),
        "--adv_mode", ADV_MODE,
        "--adv_norm", ADV_NORM,
        "--adv_rand_init", ADV_RAND_INIT,
        "--adv_project", ADV_PROJECT,
        "--adv_on_moco", ADV_ON_MOCO,
    ]
    return args


def run_once(task_type, eps, steps, gpu_id):
    """启动单个任务"""
    alpha = eps / 3.0
    run_name = f"{task_type}_stageA_eps{eps:.3f}_steps{steps}"
    log_dir = WORKDIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    cmd = [
        PY, str(WORKDIR / "main.py"),
        *build_common_args(task_type, run_name),
        "--adv_eps", str(eps),
        "--adv_alpha", str(alpha),
        "--adv_steps", str(steps),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"\n🚀 Launching {run_name} on GPU {gpu_id} (eps={eps}, steps={steps})")
    print(f"➡️ Log file: {log_path}")
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=WORKDIR, env=env, stdout=f, stderr=f)
    return proc, run_name


def main():
    active_procs = []
    all_runs = list(product(TASKS, EPS_LIST, STEPS_LIST))

    print(f"Detected {NUM_GPUS} GPU(s). Max parallel = {MAX_PARALLEL}")
    print(f"Total runs to launch: {len(all_runs)}")

    for (task_type, _), eps, steps in all_runs:
        # 若活跃进程数达到上限则等待
        while len(active_procs) >= MAX_PARALLEL:
            for p, name in active_procs[:]:
                if p.poll() is not None:
                    print(f"✅ Completed: {name}")
                    active_procs.remove((p, name))
            time.sleep(5)

        gpu_id = len(active_procs) % NUM_GPUS
        p, name = run_once(task_type, eps, steps, gpu_id)
        active_procs.append((p, name))
        time.sleep(3)  # 启动间隔，防止瞬间占满GPU

    # 等待所有剩余任务完成
    for p, name in active_procs:
        p.wait()
        print(f"✅ Completed: {name}")

    print("\n🎉 All runs completed successfully!")


if __name__ == "__main__":
    main()
