import os
import random
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch import Tensor
try:
    from torch_geometric.utils import sort_edge_index
except ImportError:
    sort_edge_index = None


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            shape = mx.shape
        values = mx.data
        return coords, values, shape

    if isinstance(sparse_mx, list):
        return [to_tuple(mx) for mx in sparse_mx]
    return to_tuple(sparse_mx)


def preprocess_features(features):
    """Row-normalize feature matrix."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1, data_dir="~/datasets/GAD/mat"):
    """Load .mat dataset."""
    data = sio.loadmat(f"{os.path.expanduser(data_dir)}/{dataset}")
    label = data["Label"] if "Label" in data else data["gnd"]
    attr = data["Attributes"] if "Attributes" in data else data["X"]
    network = data["Network"] if "Network" in data else data["A"]

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    if "Class" in data:
        labels = np.squeeze(np.array(data["Class"], dtype=np.int64) - 1)
        num_classes = np.max(labels) + 1
        labels = dense_to_one_hot(labels, num_classes)
    else:
        labels = np.zeros((adj.shape[0], 1), dtype=np.float32)

    ano_labels = np.squeeze(np.array(label))
    if "str_anomaly_label" in data:
        str_ano_labels = np.squeeze(np.array(data["str_anomaly_label"]))
        attr_ano_labels = np.squeeze(np.array(data["attr_anomaly_label"]))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_edge_index(adj: sp.spmatrix, device: Optional[torch.device] = None) -> Tensor:
    """Convert a scipy adjacency matrix to a PyG edge_index tensor sorted by source node."""
    coo = sp.coo_matrix(adj)
    row = torch.from_numpy(coo.row).long()
    col = torch.from_numpy(coo.col).long()
    edge_index = torch.stack([row, col], dim=0)
    if sort_edge_index is not None:
        out = sort_edge_index(edge_index, num_nodes=coo.shape[0], sort_by_row=True)
        edge_index = out[0] if isinstance(out, tuple) else out
    else:
        # Fallback when torch_geometric is unavailable: sort by source row, then destination col.
        order = edge_index[0] * coo.shape[0] + edge_index[1]
        perm = torch.argsort(order)
        edge_index = edge_index[:, perm]
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def edge_index_to_csr(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """Build CSR row pointer and destination tensors once, then reuse them."""
    row, col = edge_index[0], edge_index[1]
    deg = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device)
    rowptr[1:] = torch.cumsum(deg, dim=0)
    return rowptr, col


def _sample_next(rowptr: Tensor, col: Tensor, current: Tensor) -> Tensor:
    """Uniformly sample one outgoing neighbor for each node in current."""
    deg = rowptr[current + 1] - rowptr[current]
    has_neighbor = deg > 0
    nxt = current.clone()
    if has_neighbor.any():
        valid_current = current[has_neighbor]
        valid_deg = deg[has_neighbor]
        offset = torch.floor(torch.rand(valid_deg.numel(), device=current.device) * valid_deg.float()).long()
        nxt[has_neighbor] = col[rowptr[valid_current] + offset]
    return nxt


@torch.no_grad()
def generate_rwr_subgraph_tensor(
    rowptr: Tensor,
    col: Tensor,
    num_nodes: int,
    subgraph_size: int,
    sample_multiplier: int = 4,
) -> Tensor:
    """Fast CoLA-style local subgraph sampler.

    The original code calls generate_rwr_subgraph every epoch/testing round. Its first pass uses
    restart_prob=1.0, which is equivalent to repeatedly sampling direct neighbours of each center.
    This implementation exploits that equivalence, builds CSR only once outside this function, and
    returns a tensor [num_nodes, subgraph_size] directly on the target device.

    Each row is [context nodes..., center]. When a node has too few unique neighbours, contexts are
    padded by repeated sampled nodes or by the center itself, matching the fixed-size requirement.
    """
    device = rowptr.device
    reduced_size = subgraph_size - 1
    starts = torch.arange(num_nodes, dtype=torch.long, device=device)

    # 采样候选数略多于所需 context，减少低度节点重复导致的不足。
    candidate_len = max(subgraph_size * sample_multiplier, reduced_size)
    current = starts.repeat_interleave(candidate_len)
    candidates = _sample_next(rowptr, col, current).view(num_nodes, candidate_len)

    subgraphs = torch.empty((num_nodes, subgraph_size), dtype=torch.long, device=device)
    # context 去重仍需逐节点处理，但只处理一个小矩阵，避免了原始实现中的多次 RWR 重试。
    cand_cpu = candidates.detach().cpu().numpy()
    out_cpu = np.empty((num_nodes, subgraph_size), dtype=np.int64)
    for i in range(num_nodes):
        seen = set()
        context = []
        for node in cand_cpu[i]:
            node = int(node)
            if node == i or node in seen:
                continue
            seen.add(node)
            context.append(node)
            if len(context) == reduced_size:
                break
        if len(context) < reduced_size:
            if not context:
                context = [i] * reduced_size
            else:
                repeat = (reduced_size + len(context) - 1) // len(context)
                context = (context * repeat)[:reduced_size]
        out_cpu[i, :-1] = np.asarray(context[:reduced_size], dtype=np.int64)
        out_cpu[i, -1] = i
    subgraphs.copy_(torch.from_numpy(out_cpu).to(device))
    return subgraphs
