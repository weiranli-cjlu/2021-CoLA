import argparse
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model import Model
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


parser = argparse.ArgumentParser(description="CoLA: Self-Supervised Contrastive Learning for Anomaly Detection")
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
args = parser.parse_args()

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

batch_size = args.batch_size
subgraph_size = args.subgraph_size
print("Dataset: ", args.dataset)

# Set random seed. DGL seed has been removed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
features, _ = preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

# PyG graph representation for random-walk subgraph generation.
# Keep this before adjacency normalization because adj will later become dense.
edge_index = adj_to_edge_index(adj, device=device)

adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis]).to(device)
adj = torch.FloatTensor(adj[np.newaxis]).to(device)
labels = torch.FloatTensor(labels[np.newaxis]).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)

# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print("Using CUDA")

b_xent = nn.BCEWithLogitsLoss(
    reduction="none",
    pos_weight=torch.tensor([args.negsamp_ratio], device=device),
)
xent = nn.CrossEntropyLoss()

cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description("Training")
    for epoch in range(args.num_epoch):
        loss_full_batch = torch.zeros((nb_nodes, 1), device=device)
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.0

        subgraphs = generate_rwr_subgraph(edge_index, nb_nodes, subgraph_size)

        for batch_idx in range(batch_num):
            optimiser.zero_grad()
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

            logits = model(bf, ba)
            loss_all = b_xent(logits, lbl)
            loss = torch.mean(loss_all)
            loss.backward()
            optimiser.step()

            loss_value = loss.detach().cpu().item()
            loss_full_batch[idx] = loss_all[:cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss_value

        mean_loss = (total_loss * batch_size + loss_value * cur_batch_size) / nb_nodes
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), "best_model.pkl")
        else:
            cnt_wait += 1

        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)

# Test model
print("Loading {}th epoch".format(best_t))
model.load_state_dict(torch.load("best_model.pkl", map_location=device))
model.eval()

multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description("Testing")
    for round_id in range(args.auc_test_rounds):
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

            with torch.no_grad():
                logits = torch.squeeze(model(bf, ba))
                logits = torch.sigmoid(logits)
                ano_score = -(logits[:cur_batch_size] - logits[cur_batch_size:]).detach().cpu().numpy()

            multi_round_ano_score[round_id, idx] = ano_score

        pbar_test.update(1)

ano_score_final = np.mean(multi_round_ano_score, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)
print("AUC:{:.4f}".format(auc))
