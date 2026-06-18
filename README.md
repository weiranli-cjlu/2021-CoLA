# Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning

This is the fork code of TNNLS paper [Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113) (CoLA). 

## Setup
```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn --torch-backend=cu128
```

dataset_path: ~/datasets/GAD/mat

示例：

```bash
python run.py --dataset cora --trials 10 --num_epoch 100 --result_csv results/cola_results.csv
```

新增参数：

- `--trials`：独立训练次数，默认 10。
- `--result_csv`：汇总结果保存路径，默认 `results/cola_results.csv`。
- `--model_dir`：每个 trial 的最佳模型保存目录，默认 `checkpoints`。
- `--quiet_tqdm`：关闭进度条，适合批量跑实验。

CSV 中 `auc`、`auprc` 的格式为 `mean±std(max)`，数值已经乘以 100，例如 `90.21±2.33(91.00)`。
AUPRC 按要求使用：

```python
precision, recall, _ = precision_recall_curve(ano_label, ano_score_final)
auprc = sklearn.metrics.auc(recall, precision)
```

## Cite

If you compare with, build on, or use aspects of CoLA framework, please cite the following:
```
@article{liu2021anomaly,
  title={Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning},
  author={Liu, Yixin and Li, Zhao and Pan, Shirui and Gong, Chen and Zhou, Chuan and Karypis, George},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```
