import liv
import clip
import torch
from torchvision import transforms
import numpy as np

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
    "button-press-v2": "press the button",
    "door-close-v2": "close the door",
    "door-open-v2": "open the door",
    "window-open-v2": "open the window",
    "lever-pull-v2": "pull the lever",
    "hand-insert-v2": "insert hand into opening",
    "push-v2": "push the object forward",
    "basketball-v2": "shoot the basketball",
    "stick-push-v2": "push with the stick",
    "door-lock-v2": "lock the door"
}

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
