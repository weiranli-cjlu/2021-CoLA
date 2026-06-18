import os
import random
from typing import List, Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.utils import sort_edge_index


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
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
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
    label = data["Label"] if ("Label" in data) else data["gnd"]
    attr = data["Attributes"] if ("Attributes" in data) else data["X"]
    network = data["Network"] if ("Network" in data) else data["A"]

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    labels = np.squeeze(np.array(data["Class"], dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)
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
    """Convert a scipy adjacency matrix to a PyG edge_index tensor.

    This replaces the old DGLGraph conversion.  The returned tensor has shape
    [2, num_edges] and is sorted by source node, which is required by the
    CSR-style random-walk sampler below.
    """
    coo = sp.coo_matrix(adj)
    row = torch.from_numpy(coo.row).long()
    col = torch.from_numpy(coo.col).long()
    edge_index = torch.stack([row, col], dim=0)

    out = sort_edge_index(edge_index, num_nodes=coo.shape[0], sort_by_row=True)
    edge_index = out[0] if isinstance(out, tuple) else out

    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def _edge_index_to_csr(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """Build CSR row pointer and destination tensors from sorted edge_index."""
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


def _pyg_rwr_trace(
    rowptr: Tensor,
    col: Tensor,
    starts: Tensor,
    walk_length: int,
    restart_prob: float = 0.9,
) -> Tensor:
    """Generate random-walk-with-restart traces using PyTorch/PyG tensors.

    Unlike DGL's deprecated contrib API, this implementation has no DGL or
    torch-cluster dependency.  At every step, the walker first restarts to the
    seed node with probability restart_prob, then samples a uniform outgoing
    neighbor.  Isolated nodes stay at themselves.
    """
    starts = starts.long()
    current = starts.clone()
    traces = torch.empty(
        starts.numel(), walk_length + 1,
        dtype=torch.long,
        device=starts.device,
    )
    traces[:, 0] = starts

    for step in range(1, walk_length + 1):
        if restart_prob > 0:
            mask = torch.rand(starts.numel(), device=starts.device) < restart_prob
            current = torch.where(mask, starts, current)
        current = _sample_next(rowptr, col, current)
        traces[:, step] = current

    return traces


def _unique_context(trace: Tensor, center: int, reduced_size: int) -> List[int]:
    """Keep unique nodes in sampled order and put the center node outside context."""
    context: List[int] = []
    seen = set()
    for node in trace.detach().cpu().tolist():
        node = int(node)
        if node == center or node < 0 or node in seen:
            continue
        seen.add(node)
        context.append(node)
        if len(context) >= reduced_size:
            break
    return context


def generate_rwr_subgraph(edge_index: Tensor, num_nodes: int, subgraph_size: int) -> List[List[int]]:
    """Generate CoLA subgraphs with a PyG/PyTorch RWR sampler.

    Return format is exactly what the original run.py expects:
    each item has length subgraph_size, and the last element is the center node.
    """
    reduced_size = subgraph_size - 1
    rowptr, col = _edge_index_to_csr(edge_index, num_nodes)
    starts = torch.arange(num_nodes, device=edge_index.device)

    # First pass: high restart probability emphasizes the local neighborhood,
    # matching the intent of CoLA's DGL RWR-based local subgraph sampling.
    traces = _pyg_rwr_trace(
        rowptr=rowptr,
        col=col,
        starts=starts,
        walk_length=max(subgraph_size * 3, reduced_size),
        restart_prob=1.0,
    )

    subgraphs: List[List[int]] = []
    for i in range(num_nodes):
        context = _unique_context(traces[i], i, reduced_size)
        retry_time = 0

        while len(context) < reduced_size and retry_time < 10:
            retry_trace = _pyg_rwr_trace(
                rowptr=rowptr,
                col=col,
                starts=torch.tensor([i], dtype=torch.long, device=edge_index.device),
                walk_length=max(subgraph_size * 5, reduced_size),
                restart_prob=0.9,
            )[0]
            context = _unique_context(retry_trace, i, reduced_size)
            retry_time += 1

        if len(context) < reduced_size:
            if len(context) == 0:
                context = [i] * reduced_size
            else:
                repeat = (reduced_size + len(context) - 1) // len(context)
                context = (context * repeat)[:reduced_size]

        subgraphs.append(context[:reduced_size] + [i])

    return subgraphs
