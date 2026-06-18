# CoLA PyG Patch: remove DGL random walk dependency

This patch replaces the deprecated DGL 0.4.0 dependency in TrustAGI-Lab/CoLA with a PyG/PyTorch random-walk-with-restart sampler.

## Files to replace

Copy these files into the original CoLA repository root:

```bash
cp utils.py /path/to/CoLA/utils.py
cp run.py /path/to/CoLA/run.py
```

## Main changes

1. Removed `import dgl` and `dgl.random.seed` from `run.py`.
2. Replaced `adj_to_dgl_graph(adj)` with `adj_to_edge_index(adj, device=device)`.
3. Replaced `generate_rwr_subgraph(dgl_graph, subgraph_size)` with `generate_rwr_subgraph(edge_index, nb_nodes, subgraph_size)`.
4. Removed `networkx` and `dgl` imports from `utils.py`.
5. Added a PyG/PyTorch implementation of random-walk-with-restart.

## Suggested RTX 50-series environment

Use a PyTorch build that supports your RTX 50-series GPU, then install PyG matching that PyTorch version.

Example. Choose the CUDA index from the official PyTorch install selector. For many RTX 50-series environments CUDA 12.8+ is required; use cu128 or newer if available for your platform.

```bash
pip uninstall -y dgl dgl-cu* torch torchvision torchaudio torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
# Example for CUDA 12.8 wheels; replace cu128 with the official selector output when needed.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric
```

This patch does not require `torch-cluster` or DGL.
