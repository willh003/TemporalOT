import liv
import clip
import torch
from torchvision import transforms
import numpy as np
import time

def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C

TASK_DESCRIPTIONS = {
    "button-press-v2": "Press a button from the side",
    "door-close-v2": "Close a door with a revolving joint",
    "door-open-v2": "Open a door with a revolving joint",
    "window-open-v2": "Push and open a window.",
    "lever-pull-v2": "Pull a lever up",
    "hand-insert-v2": "Insert a block into an opening on the floor",
    "push-v2": "Push the puck to a goal",
    "basketball-v2": "Place a basketball in a hoop",
    "stick-push-v2": "Use a stick to push a cylinder",
    "door-lock-v2": "Pull down a lever to lock a door"
}

class LIVCostEncoder:
    def __init__(self):
        self.model = liv.load_liv()
    
    def __call__(self, obs):
        obs = obs[:, -3:] / 255.0 
        return self.model(obs, modality="vision")

class DINOCostEncoder:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.normalizer = transforms.Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]),
                                    std=torch.FloatTensor([0.229, 0.224, 0.225]))
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, obs):
        
        obs = obs[:, -3:] / 255.0 
        h = self.normalizer(obs)
        return self.model(h)

class LIVRewarder:
    def __init__(self, text_goals: list = None, image_goals: list = None, device='cuda'):
        """
        text goals: a list of text goals, strings
        images: a list of image goals, represented as torch tensors of shape (3, H, W) 
        """
        self.model = liv.load_liv()
        self.device = device

        assert not (text_goals is None and image_goals is None), "Error: must specify at least one type of goal"

        if text_goals is not None:
            text_goal_tokens = clip.tokenize(text_goals).to(self.device)
            text_embs = self.model(text_goal_tokens, modality="text")
        
        if image_goals is not None:
            image_embs = self.model(torch.stack(image_goals).to(self.device), modality="vision")
            
        if text_goals is None:
            all_embs = image_embs
        elif image_goals is None:
            all_embs = text_embs
        else:
            all_embs = torch.cat((text_embs, image_embs), dim=0)
        
        all_embs = all_embs.detach()
        
        # goal is the mean of text and image goals, shape (1, d)
        self.goal = all_embs.mean(dim=0)[None] 

    def __call__(self, observations):
        """
        Observations may be of shape B, C*N, H, W, where N is the number of images (stack on each other)
        C must be a multiple of 3
        Calculates a simple cosine similarity reward, as performed in FuRL and VLMRMs
        """
        start = time.time()
        observations = torch.from_numpy(observations).to(self.device)

        B, C, H, W = observations.shape
        assert C % 3 == 0
        observations = observations.view(B * (C // 3), 3, H, W)

        obs_embs = self.model(observations, modality="vision") # (T, d)
        
        obs_goal_distance = cosine_distance(obs_embs, self.goal)[:, 0]
        obs_goal_distance = obs_goal_distance.view(B, C // 3)
        mean_obs_goal_distance = obs_goal_distance.mean(dim=1) # mean distance from goal for all observations (smoothens if frame stacking)

        final_rewards = mean_obs_goal_distance.detach().cpu().numpy()

        total_time = time.time() - start
        info = {"cost_matrix": None,
                "assignment": None,
                "reward_calculation_time": total_time,
                "matching_calculation_time": 0}

        return final_rewards, info

    def liv_potential_reward(self, observations):
        """
        Observations may be of shape B, C*N, H, W, where N is the number of images (stack on each other)
        C must be a multiple of 3
        Calculates the potential reward as described in LIV: 
        r_{t+1} = <phi(o_{t+1}), phi(g)> - <phi(o_{t}), phi(g)>
        """    
        observations = torch.from_numpy(observations).to(self.device)

        B, C, H, W = observations.shape
        assert C % 3 == 0
        observations = observations.view(B * (C // 3), 3, H, W)

        obs_embs = self.model(observations, modality="vision") # (T, d)
        
        obs_goal_distance = cosine_distance(obs_embs, self.goal)[:, 0]
        obs_goal_distance = obs_goal_distance.view(B, C // 3)
        mean_obs_goal_distance = obs_goal_distance.mean(dim=1)
        

        potential_reward = mean_obs_goal_distance[1:] - mean_obs_goal_distance[:-1]
        final_rewards = potential_reward.detach().cpu().numpy()

        final_rewards = np.concatenate(([0], final_rewards))

        info = {"cost_matrix": None,
                "assignment": None}

        return final_rewards, info

if __name__ == "__main__":
    text_goals = ["close the door"]
    rewarder = LIVRewarder(text_goals=text_goals)

    obs = torch.rand(10, 3, 224, 224).cuda()

    rewards = rewarder(obs)    
