import argparse
import copy
import csv
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tqdm import tqdm, trange

from model import Model
from utils import (
    adj_to_edge_index,
    edge_index_to_csr,
    generate_rwr_subgraph_tensor,
    load_mat,
    normalize_adj,
    preprocess_features,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(description="Fast CoLA for attributed network anomaly detection")
    parser.add_argument("--dataset", type=str, default="cora")  # BlogCatalog/Flickr/ACM/cora/citeseer/pubmed
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "max", "min", "weighted_sum"])
    parser.add_argument("--auc_test_rounds", type=int, default=256)
    parser.add_argument("--negsamp_ratio", type=int, default=1)

    # 保留原始复现实验流程：多 trial、AUC/AUPRC 汇总、CSV 落盘。
    parser.add_argument("--trials", type=int, default=10, help="number of independent trials")
    parser.add_argument(
        "--result_csv",
        type=str,
        default="results/cola_results.csv",
        help="path to save the summary CSV",
    )

    # 仅新增不改变结果含义的运行选项。
    parser.add_argument("--patience", type=int, default=0, help="early stopping patience; 0 disables it")
    parser.add_argument("--quiet_tqdm", action="store_true", help="disable tqdm progress bars")
    parser.add_argument("--save_model", type=str, default="", help="optional path prefix to save best model state")
    return parser.parse_args()


def apply_default_hyperparams(args):
    if args.lr is None:
        args.lr = 1e-3
        if args.dataset == "ACM":
            args.lr = 5e-4
        elif args.dataset == "BlogCatalog":
            args.lr = 3e-3
    if args.num_epoch is None:
        args.num_epoch = 100
        if args.dataset in ["BlogCatalog", "Flickr", "ACM"]:
            args.num_epoch = 400
    return args


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 不强制 OMP_NUM_THREADS=1，避免 CPU 预处理和 numpy/scipy 被人为降速。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iter_batches(order: torch.Tensor, batch_size: int):
    for start in range(0, order.numel(), batch_size):
        yield order[start:start + batch_size]


def build_batch_fast(idx, subgraphs, adj_2d, features_2d):
    """Vectorized replacement for the original per-node Python loop.

    subgraphs: [num_nodes, subgraph_size]
    adj_2d: [num_nodes, num_nodes]
    features_2d: [num_nodes, ft_size]
    """
    sg = subgraphs.index_select(0, idx)  # [B, K]
    batch_size, subgraph_size = sg.shape

    bf_base = features_2d[sg]  # [B, K, F]
    # Dense subgraph adjacency: adj[rows, cols] -> [B, K, K]
    ba_base = adj_2d[sg.unsqueeze(2), sg.unsqueeze(1)]

    # 保持原实现的节点顺序：特征中 zero 插入到 center 前；邻接中 zero 追加到末尾。
    bf_zero = bf_base.new_zeros((batch_size, 1, bf_base.size(-1)))
    bf = torch.cat((bf_base[:, :-1, :], bf_zero, bf_base[:, -1:, :]), dim=1)

    ba = ba_base.new_zeros((batch_size, subgraph_size + 1, subgraph_size + 1))
    ba[:, :subgraph_size, :subgraph_size] = ba_base
    ba[:, -1, -1] = 1.0
    return bf, ba


def make_labels(cur_batch_size, negsamp_ratio, device):
    labels = torch.empty((cur_batch_size * (1 + negsamp_ratio), 1), device=device)
    labels[:cur_batch_size] = 1.0
    labels[cur_batch_size:] = 0.0
    return labels


def _save_trial_model(save_model: str, state_dict, trial_id: int, trials: int):
    if not save_model:
        return None
    save_path = Path(save_model)
    if trials > 1:
        suffix = save_path.suffix or ".pt"
        stem = save_path.stem if save_path.suffix else save_path.name
        save_path = save_path.with_name(f"{stem}_trial{trial_id}{suffix}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, save_path)
    return save_path


def train_one_trial(args, data, device, trial_id):
    features_2d, adj_2d, rowptr, col, ano_label, nb_nodes, ft_size = data

    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([args.negsamp_ratio], device=device),
    )

    best = float("inf")
    best_t = 0
    best_state = None
    wait = 0

    pbar = tqdm(
        range(args.num_epoch),
        desc=f"Trial {trial_id} Epoch",
        position=1,
        leave=False,
        disable=args.quiet_tqdm,
    )
    for epoch in pbar:
        model.train()
        order = torch.randperm(nb_nodes, device=device)
        # 只生成一次本 epoch 的所有节点子图，返回 tensor，后续 batch 直接 index_select。
        subgraphs = generate_rwr_subgraph_tensor(rowptr, col, nb_nodes, args.subgraph_size)

        total_loss = 0.0
        seen = 0
        for idx in iter_batches(order, args.batch_size):
            cur_batch_size = idx.numel()
            if cur_batch_size == 0:
                continue
            lbl = make_labels(cur_batch_size, args.negsamp_ratio, device)
            bf, ba = build_batch_fast(idx, subgraphs, adj_2d, features_2d)

            optimiser.zero_grad(set_to_none=True)
            logits = model(bf, ba)
            loss = b_xent(logits, lbl).mean()
            loss.backward()
            optimiser.step()

            total_loss += loss.detach().item() * cur_batch_size
            seen += cur_batch_size

        mean_loss = total_loss / max(seen, 1)
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            # 原代码每次 loss 下降就写 best_model.pkl；这里改为内存保存，避免频繁磁盘 IO。
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        pbar.set_postfix(loss=f"{mean_loss:.5f}", best_epoch=best_t)
        if args.patience > 0 and wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        _save_trial_model(args.save_model, best_state, trial_id, args.trials)

    auc, auprc = test_one_trial(args, model, data, device)
    return auc, auprc, best_t, best


@torch.inference_mode()
def test_one_trial(args, model, data, device):
    features_2d, adj_2d, rowptr, col, ano_label, nb_nodes, ft_size = data
    model.eval()
    # 原代码保存 [auc_test_rounds, num_nodes] 矩阵；这里改为累加，结果等价但更省内存。
    score_sum = np.zeros(nb_nodes, dtype=np.float64)

    pbar = tqdm(
        range(args.auc_test_rounds),
        desc="Testing",
        position=1,
        leave=False,
        disable=args.quiet_tqdm,
    )
    for _ in pbar:
        order = torch.randperm(nb_nodes, device=device)
        subgraphs = generate_rwr_subgraph_tensor(rowptr, col, nb_nodes, args.subgraph_size)
        for idx in iter_batches(order, args.batch_size):
            cur_batch_size = idx.numel()
            if cur_batch_size == 0:
                continue
            bf, ba = build_batch_fast(idx, subgraphs, adj_2d, features_2d)
            logits = torch.sigmoid(model(bf, ba)).view(-1)
            pos_score = logits[:cur_batch_size]
            neg_score = logits[cur_batch_size:].view(args.negsamp_ratio, cur_batch_size).mean(dim=0)
            ano_score = -(pos_score - neg_score).detach().cpu().numpy()
            score_sum[idx.detach().cpu().numpy()] += ano_score

    ano_score_final = score_sum / max(args.auc_test_rounds, 1)
    auc = roc_auc_score(ano_label, ano_score_final)
    precision, recall, _ = precision_recall_curve(ano_label, ano_score_final)
    auprc = sk_auc(recall, precision)
    return auc, auprc


def format_metric(values):
    values = np.asarray(values, dtype=np.float64) * 100.0
    return f"{values.mean():.2f}±{values.std():.2f}({values.max():.2f})"


def format_trial_values(values):
    values = np.asarray(values, dtype=np.float64) * 100.0
    return ";".join(f"{v:.2f}" for v in values)


def append_result_csv(args, auc_list, auprc_list, best_epoch_list, best_loss_list):
    result_csv = Path(args.result_csv)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "trial": args.trials,
        "completed_trials": len(auc_list),
        "auc": format_metric(auc_list),
        "auprc": format_metric(auprc_list),
        "seed": args.seed,
        "训练轮次": args.num_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "batch_size": args.batch_size,
        "subgraph_size": args.subgraph_size,
        "readout": args.readout,
        "auc_test_rounds": args.auc_test_rounds,
        "negsamp_ratio": args.negsamp_ratio,
        "auc_each_trial": format_trial_values(auc_list),
        "auprc_each_trial": format_trial_values(auprc_list),
        "best_epoch_each_trial": ";".join(str(int(x)) for x in best_epoch_list),
        "best_loss_each_trial": ";".join(f"{float(x):.6f}" for x in best_loss_list),
    }
    write_header = (not result_csv.exists()) or result_csv.stat().st_size == 0
    with result_csv.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return result_csv, row


def main():
    args = apply_default_hyperparams(parse_args())

    print("Dataset:", args.dataset)
    print("Trials:", args.trials)
    print("Epochs:", args.num_epoch)
    print("Test rounds:", args.auc_test_rounds)

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using CUDA")

    # 数据只读取和预处理一次；各 trial 只重置随机种子和模型。
    adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(
        args.dataset, data_dir=args.data_dir
    )
    features, _ = preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    # 子图采样用稀疏 edge_index；邻接归一化后保留 dense 矩阵用于向量化子图抽取。
    edge_index = adj_to_edge_index(adj, device=device)
    rowptr, col = edge_index_to_csr(edge_index, nb_nodes)

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features_2d = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    adj_2d = torch.as_tensor(np.asarray(adj), dtype=torch.float32, device=device)

    data = (features_2d, adj_2d, rowptr, col, ano_label, nb_nodes, ft_size)

    auc_list = []
    auprc_list = []
    best_epoch_list = []
    best_loss_list = []

    trial_pbar = trange(args.trials, desc="Trial", position=0, leave=True, disable=args.quiet_tqdm)
    for trial_id in trial_pbar:
        set_random_seed(args.seed + trial_id)
        trial_auc, trial_auprc, best_epoch, best_loss = train_one_trial(args, data, device, trial_id)
        auc_list.append(trial_auc)
        auprc_list.append(trial_auprc)
        best_epoch_list.append(best_epoch)
        best_loss_list.append(best_loss)
        trial_pbar.set_postfix(auc=f"{trial_auc * 100:.2f}", auprc=f"{trial_auprc * 100:.2f}")

    result_csv, row = append_result_csv(args, auc_list, auprc_list, best_epoch_list, best_loss_list)

    print("\n===== Summary =====")
    print(f"AUC: {row['auc']}")
    print(f"AUPRC: {row['auprc']}")
    print(f"Saved to: {result_csv}")


if __name__ == "__main__":
    main()
