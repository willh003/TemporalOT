# Cover Your Bases

## Environment Setup

Install the requirements, including the metaworld submodule. This should link to the temporalot branch of the portal-cornell metaworld repo.

```shell
conda create --name coverage python==3.9
conda activate coverage
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
cd Metaworld
pip install -e .
```

## Run Experiments

We need to first generate the expert demo data using `demo/collect_expert_traj.py`. We can optionally configure the environment, number of demos (default 2, as they do in TemporalOT), and camera angle (for default, use "d"). Environment defaults, like cameras and episode lengths, are specified in `demo/constants.py`

We can then run the TemporalOT agent using `python main.py`. To change the reward function, simply specify reward_fn in the config or command line. These functions are defined in seq_matching/load_matching_fn.

Training configs are stored in configs/ . You can modify them as command line arguments or directly in the config.


```shell
python -m demo.collect_expert_traj -e "door-close-v2" -n 2
python main.py reward_fn="coverage" env_name="door-close-v2" num_demos=2
```

Scripts for a single train run or a batched train run over different parameters are available in single.sh and multi.sh.
