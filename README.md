# Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning

This is the fork code of TNNLS paper [Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113) (CoLA). 

## Setup
```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn --torch-backend=cu128
```

dataset_path: ~/datasets/GAD/mat

## Running the experiments
### Step 2: Anomaly Detection
This step is to run the CoLA framework to detect anomalies in the network datasets. Take Cora dataset as an example:
```
python run.py --dataset cora
```
The hyperparameters are set to be the values reported in [our paper](https://arxiv.org/abs/2103.00113). 
You can change the default values of other parameters to simulate different conditions. 

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
