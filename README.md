# ORCA: ORdered Coverage Alignment

## Environment Setup

Install the requirements, including the metaworld submodule.

```shell
conda create --name coverage python==3.9
conda activate coverage
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
cd Metaworld
pip install -e .
```

## Run Experiments

### Collect Expert Demos

We need to first generate the expert demo data using `demo/collect_expert_traj.py`. We can optionally configure the environment/task and camera angle (for default, use "d"). The cameras angles and episode lengths that we use for each task are specified in `demo/constants.py`. Demos are saved in `create_demo/{environment}_demo/{task}/...`. To change the demo save path, change BASE_DEMO_DIR in `demo/constants.py`


Collect temporally aligned demo for door-close-v2:
```shell
python -m demo.collect_expert_traj -e "door-close-v2"
```
Collect temporally misaligned demo (hand selected misalignment):
```shell
python -m demo.create_mismatch_traj -e "door-close-v2"
```
Collect randomly temporally misaligned demo:
```shell
python -m demo.create_random_mismatch_traj -e "door-close-v2" -s "fast"
```

### Training

After collecting expert trajectories, you can run the TemporalOT agent using `python main.py`. To change the reward function, simply specify reward_fn in the config or command line. These functions are defined in seq_matching/load_matching_fn. **You must collect expert demos of the correct type (matched, mismatched, random mismatched) before training**

Train ORCA with temporally aligned demo
```shell
python main.py reward_fn="coverage" env_name="door-close-v2" mismatched=false random_mismatched=false
```
Train ORCA with temporally misaligned demo. Make sure to create the aligned demo first, then run `demo.create_mismatch_traj`
```shell
python main.py reward_fn="coverage" env_name="door-close-v2" mismatched=true random_mismatched=false
```
Train ORCA with randomly temporally misaligned, faster demo. To run slower demo, use speed_type="slow". Make sure to create the aligned demo first, then run `demo.create_random_mismatch_traj`
```shell
python main.py reward_fn="coverage" env_name="door-close-v2" speed_type="fast" mismatched=false random_mismatched=true
```

Training configs are stored in configs/ . You can modify them as command line arguments or directly in the config. The expert trajectory that is used is specified by the environment, camera, and whether it is {matched, mismatched, random mismatched}. 


`single.sh` runs a single train run, and contains all of the parameters of interest.

## Eval

States from each eval run are saved in train_logs/{run_name}/eval. Performance metrics over the course of training, such as final success rate (% eval rollouts with success on final frame) and total success rate (# successful frames per eval rollout) are stored in train_logs/{run_name}/logs/performance.csv after training is complete.