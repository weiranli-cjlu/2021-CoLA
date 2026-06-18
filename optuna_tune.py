import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import optuna


def parse_args():
    parser = argparse.ArgumentParser("Optuna tuner for 2021-CoLA")

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--script", type=str, default="run.py")
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--repeat_runs", type=int, default=5,
                        help="每组超参数重复训练次数，用不同 seed 计算均值")
    parser.add_argument("--base_seed", type=int, default=42)

    parser.add_argument("--num_epoch", type=int, default=None,
                        help="固定训练轮数；不填则使用 run.py 默认逻辑")
    parser.add_argument("--auc_test_rounds", type=int, default=256)

    parser.add_argument("--metric", type=str, default="auc",
                        choices=["auc", "auprc"],
                        help="Optuna 优化目标。当前 GitHub run.py 只输出 AUC；若本地已支持 AUPRC 可设为 auprc")

    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None,
                        help="例如 sqlite:///tune/cola_optuna.db；不填则使用内存")
    parser.add_argument("--direction", type=str, default="maximize")

    parser.add_argument("--result_dir", type=str, default="tune")
    parser.add_argument("--quiet_tqdm", action="store_true",
                        help="如果你的 run.py 支持 --quiet_tqdm，则会自动传入")

    parser.add_argument("--timeout_per_run", type=int, default=0,
                        help="单次 run.py 超时时间，单位秒；0 表示不限制")

    return parser.parse_args()


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_supported_flags(python_exec, script_path):
    """
    检测本地 run.py 是否支持 --trials、--result_csv、--quiet_tqdm 等参数。
    当前 GitHub 页面中的 run.py 不支持这些参数，但 README 中提到了这些参数。
    """
    try:
        proc = subprocess.run(
            [python_exec, script_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            cwd=str(Path(script_path).resolve().parent),
        )
        help_text = proc.stdout or ""
    except Exception:
        help_text = ""

    flags = set(re.findall(r"--[A-Za-z0-9_\-]+", help_text))
    return flags


def parse_metric_from_stdout(stdout, metric):
    """
    支持解析：
    AUC:0.9123
    AUC: 91.23
    AUROC: 91.23±0.12(91.35)
    AUPRC: 78.12±1.00(79.20)
    """
    patterns = {
        "auc": [
            r"\bAUC\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"\bAUROC\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "auprc": [
            r"\bAUPRC\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"\bAP\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        ],
    }

    for pattern in patterns[metric]:
        matches = re.findall(pattern, stdout, flags=re.IGNORECASE)
        if matches:
            value = float(matches[-1])
            # run.py 可能输出 0.9123，也可能输出 91.23
            if value > 1.5:
                value = value / 100.0
            return value

    return None


def tail_text(text, max_chars=3000):
    text = text or ""
    return text[-max_chars:]


def build_command(args, params, seed, supported_flags, inner_csv):
    cmd = [
        args.python,
        args.script,
        "--dataset", args.dataset,
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--seed", str(seed),
        "--embedding_dim", str(params["embedding_dim"]),
        "--batch_size", str(params["batch_size"]),
        "--subgraph_size", str(params["subgraph_size"]),
        "--readout", params["readout"],
        "--auc_test_rounds", str(args.auc_test_rounds),
        # 当前 GitHub run.py 的测试阶段没有正确处理 negsamp_ratio > 1，
        # 因此这里固定为 1，避免 shape mismatch。
        "--negsamp_ratio", "1",
    ]

    if args.num_epoch is not None:
        cmd += ["--num_epoch", str(args.num_epoch)]

    # 如果你的本地 run.py 已经支持这些参数，则每次外部重复只跑 1 个 trial，
    # 避免“双重 trials”导致重复次数膨胀。
    if "--trials" in supported_flags:
        cmd += ["--trials", "1"]

    if "--result_csv" in supported_flags:
        cmd += ["--result_csv", str(inner_csv)]

    if args.quiet_tqdm and "--quiet_tqdm" in supported_flags:
        cmd += ["--quiet_tqdm"]

    return cmd


def objective_factory(args, supported_flags):
    result_dir = Path(args.result_dir)
    summary_csv = result_dir / f"{args.dataset}_optuna_summary.csv"
    inner_csv = result_dir / f"{args.dataset}_run_inner.csv"

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 300]),
            "subgraph_size": trial.suggest_int("subgraph_size", 2, 4),
            "readout": trial.suggest_categorical("readout", ["avg", "max", "min", "weighted_sum"]),
        }

        metric_values = []
        auc_values = []
        auprc_values = []

        for repeat_id in range(args.repeat_runs):
            seed = args.base_seed + trial.number * 1000 + repeat_id
            cmd = build_command(args, params, seed, supported_flags, inner_csv)

            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

            timeout = None if args.timeout_per_run <= 0 else args.timeout_per_run

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(args.script).resolve().parent),
                env=env,
                timeout=timeout,
            )

            stdout = proc.stdout or ""
            auc = parse_metric_from_stdout(stdout, "auc")
            auprc = parse_metric_from_stdout(stdout, "auprc")

            if auc is not None:
                auc_values.append(auc)
            if auprc is not None:
                auprc_values.append(auprc)

            if args.metric == "auc":
                value = auc
            else:
                value = auprc

            if value is None:
                raise optuna.TrialPruned(
                    f"Cannot parse {args.metric.upper()} from run.py output. "
                    f"Current GitHub run.py only prints AUC; use --metric auc or add AUPRC printing."
                )

            metric_values.append(value)

            # 给 Optuna 一个中间结果，便于 pruner 提前停止差的参数。
            trial.report(mean(metric_values), step=repeat_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        target_mean = mean(metric_values)
        target_std = pstdev(metric_values) if len(metric_values) > 1 else 0.0

        auc_mean = mean(auc_values) if auc_values else None
        auc_std = pstdev(auc_values) if len(auc_values) > 1 else 0.0

        auprc_mean = mean(auprc_values) if auprc_values else None
        auprc_std = pstdev(auprc_values) if len(auprc_values) > 1 else 0.0

        summary_row = {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "optuna_trial": trial.number,
            "repeat_runs": args.repeat_runs,
            "target_metric": args.metric,
            "target_mean": target_mean,
            "target_std": target_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "auprc_mean": auprc_mean,
            "auprc_std": auprc_std,
            "params_json": json.dumps(params, ensure_ascii=False),
            "lr": params["lr"],
            "weight_decay": params["weight_decay"],
            "embedding_dim": params["embedding_dim"],
            "batch_size": params["batch_size"],
            "subgraph_size": params["subgraph_size"],
            "readout": params["readout"],
            "num_epoch": args.num_epoch,
            "auc_test_rounds": args.auc_test_rounds,
        }
        append_csv(summary_csv, summary_row)

        return target_mean

    return objective


def main():
    args = parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.study_name is None:
        args.study_name = f"cola_{args.dataset}_{args.metric}"

    if args.storage is None:
        args.storage = f"sqlite:///{result_dir / (args.study_name + '.db')}"

    supported_flags = get_supported_flags(args.python, args.script)
    print("Detected supported flags:", sorted(supported_flags))
    print("Study storage:", args.storage)

    sampler = optuna.samplers.TPESampler(seed=args.base_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(5, args.n_trials // 10),
        n_warmup_steps=max(1, args.repeat_runs // 2),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        objective_factory(args, supported_flags),
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print("\n===== Best Trial =====")
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_cmd = [
        "python",
        args.script,
        "--dataset", args.dataset,
        "--lr", str(study.best_params["lr"]),
        "--weight_decay", str(study.best_params["weight_decay"]),
        "--embedding_dim", str(study.best_params["embedding_dim"]),
        "--batch_size", str(study.best_params["batch_size"]),
        "--subgraph_size", str(study.best_params["subgraph_size"]),
        "--readout", str(study.best_params["readout"]),
        "--auc_test_rounds", str(args.auc_test_rounds),
        "--negsamp_ratio", "1",
    ]
    if args.num_epoch is not None:
        best_cmd += ["--num_epoch", str(args.num_epoch)]

    with open(os.path.join(args.result_dir, f"{args.study_name}.sh"), "w") as f:
        f.write(" ".join(best_cmd))
    print("\nRecommended command:")
    print(" ".join(best_cmd))


if __name__ == "__main__":
    main()