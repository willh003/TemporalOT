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

### Collect Expert Data

We need to first generate the expert demo data using `demo/collect_expert_traj.py`. We can optionally configure the environment, number of demos (default 2, as they do in TemporalOT), and camera angle (for default, use "d"). Environment defaults, like cameras and episode lengths, are specified in `demo/constants.py`. By default, the expert runs are truncated to 

### Subsample Expert Data

To obtain expert runs with mismatched execution, use `subsample_gifs_and_states.py`. You can specify indices of the gif with `mismatched_subsample_gifs_and_states`, or uniformly take N frames from all the frames up to last_frame with `evenly_subsample_gif_and_states`. These runs will be saved under the task and camera with the "mismatched" tag (so if you have multiple mismatched versions for the same task, they will be overwritten).

### Training

After collecting expert trajectories, you can run the TemporalOT agent using `python main.py`. To change the reward function, simply specify reward_fn in the config or command line. These functions are defined in seq_matching/load_matching_fn.

The expert trajectory(s) to use is specified by the environment, camera, and whether or not it is mismatched. If you choose to train with multiple expert trajectories, make sure that you have also generated multiple.

Training configs are stored in configs/ . You can modify them as command line arguments or directly in the config.


```shell
python -m demo.collect_expert_traj -e "door-close-v2" -n 2
python main.py reward_fn="coverage" env_name="door-close-v2" num_demos=2
```

`single.sh` runs a single train run, and `multi.sh` runs a slurm batch job, allowing you to run with different configurations (especially useful for running baselines and multiple seeds). By default, `multi.sh` will submit jobs to g2, not portal-cornell.

## Eval

States from each eval run are saved in train_logs/{run_name}/eval (I'm only saving the first 10 eval rollouts per eval step, but we are also only doing 10 eval rollouts per step now so this should be fine). Performance metrics over the course of training, such as final success rate (% eval rollouts with success on final frame) and total success rate (# successful frames per eval rollout) are stored in train_logs/{run_name}/logs/performance.csv after training is complete.