# TemporalOT

## Environment Setup

We use Python3.9 and Cuda12.

```shell
conda create --name TemporalOT python==3.9
conda activate TemporalOT
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Run Experiments

We need to first generate the expert demo data using `collect_expert_traj.py`.

We can then run the TemporalOT agent using `python main.py`.

## Cite

Please cite our work if you find it useful:

```
@InProceedings{fu2024robot,
  title={Robot Policy Learning with Temporal Optimal Transport Reward},
  author = {Yuwei Fu and Haichao Zhang and Di Wu and Wei Xu and Benoit Boulet},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year = {2024}
}
```