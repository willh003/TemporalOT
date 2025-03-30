
import torch
from models import LIVRewarder, TASK_DESCRIPTIONS
import numpy as np
import time
from utils import cosine_distance
from seq_matching import load_matching_fn
from models import ResNet
import csv

# get the custom reward function
matching_fn_cfg = {
    "tau": 1,
    "ent_reg": .01,
    "mask_k": 10,
    "sdtw_smoothing": 5,
    "track_progress": False,
    "threshold": 0.9
}

def write_method_compute_times_to_csv(method_compute_times, filename="method_compute_times.csv"):
    """
    Write the method compute times dictionary to a CSV file.
    
    Parameters:
    -----------
    method_compute_times : dict
        Dictionary mapping method names to their average compute times
    filename : str, optional
        Name of the CSV file to write to (default: "method_compute_times.csv")
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['method', 'compute_time_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for method, compute_time in method_compute_times.items():
            writer.writerow({'method': method, 'compute_time_ms': compute_time})
    
    print(f"Results written to {filename}")

def warmup_gpu():
    """
    Perform a quick GPU warm-up to ensure measurements aren't affected by initial GPU spin-up.
    """
    if torch.cuda.is_available():
        print("Warming up GPU...")
        # Create a random tensor and perform some operations
        dummy_tensor = torch.randn(1000, 1000, device='cuda')
        for _ in range(100):
            dummy_tensor = torch.matmul(dummy_tensor, dummy_tensor)
            torch.cuda.synchronize()  # Ensure GPU operations are completed
        # Clear cache
        torch.cuda.empty_cache()
        print("GPU warm-up complete")
    else:
        print("No GPU available, skipping warm-up")

class DistanceMatrixRewarder:
    def __init__(self, demo, reward_fn, cost_encoder, device, context_num):
        self.reward_fn = reward_fn
        self.device = device
        self.context_num = context_num
        self.cost_encoder = cost_encoder
        with torch.no_grad():
            demo_emb = self.cost_encoder(torch.as_tensor(demo).to(self.device))
        self.demo = self.get_context_observations(demo_emb)

    def get_context_observations(self, observations):
        L = len(observations)
        idx0 = np.arange(L)
        context_observations = [observations[idx0]]
        for i in range(1, self.context_num):
            idx_i = (idx0 + i).clip(0, L-1)
            context_observations.append(observations[idx_i])
        context_observations = torch.stack(context_observations)
        return context_observations

    def __call__(self, observations):
        obs = torch.as_tensor(observations).to(self.device)

        with torch.no_grad():
            obs = self.cost_encoder(obs)
        obs = self.get_context_observations(obs)

        d_times = []
        
        # context cost matrix
        distance_matrix = 0

        for i in range(self.context_num):
            distance_matrix += cosine_distance(obs[i], self.demo[i])
        distance_matrix /= self.context_num
        
        rewards, info = self.reward_fn(distance_matrix.cpu().numpy())
        return rewards

def main():
    demo_length = 100
    n_rollouts = 100
    rollout_len = 100
    device = 'cuda'
    methods = ["threshold", "liv_text",  "ot", "temporal_ot", "dtw", "coverage"]
    demo = np.random.randint(0, 255, size=(demo_length, 3, 224, 224))
    warmup_gpu()

    method_compute_times = {}
    for method in methods:
        if "liv" in method:
            image_goals = None
            text_goals = None
            if "text" in method:
                text_goals = [TASK_DESCRIPTIONS["button-press-v2"]]

            rewarder = LIVRewarder(text_goals=text_goals, image_goals=image_goals, device=device)

        else:
            cost_encoder = ResNet().to(device)
            cost_encoder.eval()
            reward_fn = load_matching_fn(method, matching_fn_cfg)
            rewarder = DistanceMatrixRewarder(demo, reward_fn, cost_encoder, device, context_num=3)

        times = []
        for i in range(n_rollouts):
            rollout = np.random.randint(0, 255, size=(rollout_len, 3, 224, 224))  
                      
            time_start = time.time()
            _ = rewarder(rollout)
            compute_time = time.time() - time_start
            times.append(compute_time)

        avg_latency = np.mean(times) * 1000 # store ms
        method_compute_times[method] = avg_latency
        print(f"Average latency for {method}: {avg_latency} ms")

    write_method_compute_times_to_csv(method_compute_times)

if __name__=="__main__":
    main()