#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段A 弱折对照实验压测版
功能：
- 日志文件 + 控制台实时打印
- 单 GPU 并行 3 个任务
- GPU 轮询
- 动态 α
- 阈值扫描 + 温度校准 + PGD 后期启用
"""

import os
import sys
import subprocess
import time
from itertools import product
from pathlib import Path

# ===================== 项目路径 =====================
ROOT = Path(__file__).resolve().parent
if (ROOT / "main.py").exists():
    WORKDIR = ROOT
elif (ROOT / "hyperparameter-tuning" / "main.py").exists():
    WORKDIR = ROOT / "hyperparameter-tuning"
else:
    raise FileNotFoundError("未找到 main.py，请检查项目路径")

PY = sys.executable or "python3"

# ===================== PGD与扫描/校准 =====================
EPS_LIST: list[float] = [0.008, 0.012, 0.015]
STEPS_LIST: list[int] = [3, 5]
DYNAMIC_ALPHA = True
ADV_MODE = "mgraph"
ADV_NORM = "linf"
ADV_RAND_INIT = "true"
ADV_PROJECT = "true"
ADV_ON_MOCO = "true"
ADV_WARMUP_END = 3

# ===================== 基线参数 =====================
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
}

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

TASKS: list[tuple[str, str]] = [
    ("LDA", "LDA_C_stageA"),
    ("LMI", "LMI_C_stageA"),
    ("MDA", "MDA_C_stageA"),
]

MAX_PARALLEL_PER_GPU = 3
START_INTERVAL = 3  # 秒

# ===================== GPU检测与轮询 =====================
def detect_gpus():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        gpus = [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]
        return gpus if gpus else [0]
    except Exception:
        return [0]

GPUS = detect_gpus()
print(f"检测到 {len(GPUS)} 张 GPU: {GPUS}")

def assign_gpu(job_idx):
    return GPUS[job_idx % len(GPUS)]

# ===================== 构建通用参数 =====================
def build_common_args(task_type: str, run_suffix: str) -> list[str]:
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

# ===================== 单任务运行 =====================
def run_task(params):
    task_type, eps, steps, job_idx = params
    gpu_id = assign_gpu(job_idx)
    alpha = eps / steps if DYNAMIC_ALPHA else eps / 3.0
    run_name = f"{task_type}_stageA_eps{eps:.3f}_steps{steps}"

    log_dir = WORKDIR / "logs"
    log_dir.mkdir(exist_ok=True)
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

    print(f"\n=== Launch {run_name} on GPU {gpu_id} ===")
    print(" ".join(cmd))
    print(f"→ Log: {log_path}")

    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd, cwd=WORKDIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True
        )
        # 控制台 + 文件同步打印
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(f"[{run_name}] {line}")
            f.write(line + "\n")
            f.flush()
        proc.wait()
        print(f"[{run_name}] 任务结束 (exit={proc.returncode})")

    return (run_name, proc.returncode)

# ===================== 主入口 =====================
def main():
    jobs = []
    idx = 0
    for task_type, _ in TASKS:
        for eps, steps in product(EPS_LIST, STEPS_LIST):
            jobs.append((task_type, eps, steps, idx))
            idx += 1

    print(f"\n共 {len(jobs)} 个任务，将使用 {len(GPUS)} 张 GPU，每 GPU 同时最多 {MAX_PARALLEL_PER_GPU} 任务。\n")

    running_procs = []
    for job_idx, params in enumerate(jobs):
        # 启动间隔
        time.sleep(START_INTERVAL)
        # 启动任务
        cmd: list[str] = [
            PY,
            str(WORKDIR / "main.py"),
            *build_common_args(params[0], f"{params[0]}_stageA_eps{params[1]:.3f}_steps{params[2]}"),
            "--adv_eps", str(params[1]),
            "--adv_alpha", str(params[1] / params[2] if DYNAMIC_ALPHA else params[1] / 3.0),
            "--adv_steps", str(params[2]),
        ]
        env: dict[str, str] = {**os.environ, "CUDA_VISIBLE_DEVICES": str(assign_gpu(job_idx))}
        p = subprocess.Popen(
            cmd,
            cwd=str(WORKDIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        log_file = WORKDIR / "logs" / f"{params[0]}_eps{params[1]:.3f}_steps{params[2]}.log"
        os.makedirs(log_file.parent, exist_ok=True)
        # 打印到控制台 + 写入 log 文件
        with open(log_file, "w") as f:
            for line in p.stdout:
                line = line.rstrip("\n")
                print(f"[{params[0]}_eps{params[1]:.3f}_steps{params[2]}] {line}")
                f.write(line + "\n")
                f.flush()
        running_procs.append(p)

        # 控制单GPU并行数量
        if len(running_procs) >= MAX_PARALLEL_PER_GPU * len(GPUS):
            for pp in running_procs:
                pp.wait()
            running_procs = []

    # 等待剩余进程完成
    for pp in running_procs:
        pp.wait()

    print("\n=== 全部任务完成 ===")

if __name__ == "__main__":
    main()
