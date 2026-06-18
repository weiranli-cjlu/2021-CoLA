import argparse
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
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
best_model_path = "best_model.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="CoLA: Self-Supervised Contrastive Learning for Anomaly Detection"
    )
    parser.add_argument("--dataset", type=str, default="cora")  # BlogCatalog/Flickr/ACM/cora/citeseer/pubmed
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--readout", type=str, default="avg")  # max/min/avg/weighted_sum
    parser.add_argument("--auc_test_rounds", type=int, default=256)
    parser.add_argument("--negsamp_ratio", type=int, default=1)

    # Multi-trial and result saving.
    parser.add_argument("--trials", type=int, default=10, help="number of independent trials")
    parser.add_argument(
        "--result_csv",
        type=str,
        default="results/cola_results.csv",
        help="path to save the summary CSV",
    )
    parser.add_argument(
        "--quiet_tqdm",
        action="store_true",
        help="disable tqdm progress bars when running many trials",
    )
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
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_batch(idx, subgraphs, adj, features, subgraph_size, ft_size, device):
    cur_batch_size = len(idx)
    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size), device=device)
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1), device=device)
    added_adj_zero_col[:, -1, :] = 1.0
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size), device=device)

    ba = []
    bf = []
    for i in idx:
        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
        cur_feat = features[:, subgraphs[i], :]
        ba.append(cur_adj)
        bf.append(cur_feat)

    ba = torch.cat(ba)
    ba = torch.cat((ba, added_adj_zero_row), dim=1)
    ba = torch.cat((ba, added_adj_zero_col), dim=2)

    bf = torch.cat(bf)
    bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
    return bf, ba


def train_one_trial(args, data, device, trial_id):
    (
        features,
        adj,
        edge_index,
        ano_label,
        nb_nodes,
        ft_size,
        batch_num,
        batch_size,
        subgraph_size,
    ) = data

    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([args.negsamp_ratio], device=device),
    )

    best = 1e9
    best_t = 0

    train_pbar = tqdm(
        range(args.num_epoch),
        desc=f"Epoch",
        position=1,
        leave=False,
        disable=args.quiet_tqdm,
    )
    for epoch in train_pbar:
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs = generate_rwr_subgraph(edge_index, nb_nodes, subgraph_size)

        total_loss = 0.0
        seen_nodes = 0
        for batch_idx in range(batch_num):
            is_final_batch = batch_idx == (batch_num - 1)
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)
            if cur_batch_size == 0:
                continue

            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))),
                1,
            ).to(device)

            optimiser.zero_grad()
            bf, ba = build_batch(idx, subgraphs, adj, features, subgraph_size, ft_size, device)
            logits = model(bf, ba)
            loss_all = b_xent(logits, lbl)
            loss = torch.mean(loss_all)
            loss.backward()
            optimiser.step()

            loss_value = loss.detach().cpu().item()
            total_loss += loss_value * cur_batch_size
            seen_nodes += cur_batch_size

        mean_loss = total_loss / max(seen_nodes, 1)
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            torch.save(model.state_dict(), best_model_path)

        train_pbar.set_postfix(loss=mean_loss, best_epoch=best_t)

    auc, auprc = test_one_trial(
        args=args,
        model=model,
        data=data,
        device=device,
        checkpoint_path=best_model_path,
    )
    return auc, auprc, best_t


def test_one_trial(args, model, data, device, checkpoint_path):
    (
        features,
        adj,
        edge_index,
        ano_label,
        nb_nodes,
        ft_size,
        batch_num,
        batch_size,
        subgraph_size,
    ) = data

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
    test_pbar = tqdm(
        range(args.auc_test_rounds),
        desc="Testing",
        disable=args.quiet_tqdm,
    )
    for round_id in test_pbar:
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs = generate_rwr_subgraph(edge_index, nb_nodes, subgraph_size)

        for batch_idx in range(batch_num):
            is_final_batch = batch_idx == (batch_num - 1)
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)
            if cur_batch_size == 0:
                continue

            bf, ba = build_batch(idx, subgraphs, adj, features, subgraph_size, ft_size, device)
            with torch.no_grad():
                logits = torch.sigmoid(model(bf, ba)).view(-1)
                pos_score = logits[:cur_batch_size]
                neg_score = logits[cur_batch_size:].view(args.negsamp_ratio, cur_batch_size).mean(dim=0)
                ano_score = -(pos_score - neg_score).detach().cpu().numpy()

            multi_round_ano_score[round_id, idx] = ano_score

    ano_score_final = np.mean(multi_round_ano_score, axis=0)
    auc = roc_auc_score(ano_label, ano_score_final)

    # AUPRC required by the user: sklearn.metrics.auc + precision_recall_curve.
    precision, recall, _ = precision_recall_curve(ano_label, ano_score_final)
    auprc = sk_auc(recall, precision)
    return auc, auprc


def format_metric(values):
    values = np.asarray(values, dtype=np.float64) * 100.0
    return f"{values.mean():.2f}±{values.std():.2f}({values.max():.2f})"


def format_trial_values(values):
    values = np.asarray(values, dtype=np.float64) * 100.0
    return ";".join(f"{v:.2f}" for v in values)


def append_result_csv(args, auc_list, auprc_list, best_epoch_list):
    result_csv = Path(args.result_csv)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "trial": args.trials,
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
        "auc_each_trial": format_trial_values(auc_list),
        "auprc_each_trial": format_trial_values(auprc_list),
        "best_epoch_each_trial": ";".join(str(int(x)) for x in best_epoch_list),
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

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using CUDA")

    # Load and preprocess data once. The training/testing randomness is controlled per trial.
    adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
    features, _ = preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    # PyG graph representation for random-walk subgraph generation.
    # Keep this before adjacency normalization because adj will later become dense.
    edge_index = adj_to_edge_index(adj, device=device)
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    batch_num = nb_nodes // batch_size + 1

    data = (
        features,
        adj,
        edge_index,
        ano_label,
        nb_nodes,
        ft_size,
        batch_num,
        batch_size,
        subgraph_size,
    )

    auc_list = []
    auprc_list = []
    best_epoch_list = []

    trial_pbar = trange(args.trials, desc=f"Trial", position=0, leave=True, disable=args.quiet_tqdm)

    for trial_id in trial_pbar:
        set_random_seed(args.seed + trial_id)
        trial_auc, trial_auprc, best_epoch = train_one_trial(args, data, device, trial_id)
        auc_list.append(trial_auc)
        auprc_list.append(trial_auprc)
        best_epoch_list.append(best_epoch)

    result_csv, row = append_result_csv(args, auc_list, auprc_list, best_epoch_list)
    print("\n===== Summary =====")
    print(f"AUC:   {row['auc']}")
    print(f"AUPRC: {row['auprc']}")
    print(f"Saved to: {result_csv}")


if __name__ == "__main__":
    main()
