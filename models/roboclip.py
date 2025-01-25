"""
Roboclip original code:
https://github.com/sumedh7/RoboCLIP/blob/main/metaworld_envs.py
"""

import torch
from .s3dg import S3D
import numpy as np

RB_PATH = "/home/aw588/git_annshin/roboclip_local/"


_roboclip_encoder = None

def get_roboclip_encoder():
    """Singleton pattern to maintain one encoder instance"""
    global _roboclip_encoder
    if _roboclip_encoder is None:
        _roboclip_encoder = RoboClipEncoder()
    return _roboclip_encoder

class RoboClipEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.net = S3D(f'{RB_PATH}/s3d_dict.npy', 512) # TODO@Anne
        self.net.load_state_dict(torch.load(f'{RB_PATH}/s3d_howto100m.pth')) # TODO@Anne
        self.net = self.net.to(device).eval()
        
    def __call__(self, frames):
        """Match ResNet interface"""
        return self.encode_video(frames)
        
    def encode_video(self, frames):
        """Encode video frames to embedding"""
        frames = self.preprocess_frames(frames)
    
        with torch.no_grad():
            video_output = self.net(frames.float())
            return video_output['video_embedding']

    def preprocess_frames(self, frames, shorten=True):
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy() # (63, 3, 224, 224)

        # Center crop
        # center = 240, 320
        # h, w = (250, 250)

        # For 224 x 224 source images:
        center = (112, 112)   # row=112, col=112
        h, w = (224, 224)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        # print(f"{frames[0].shape}") # torch.Size([18, 3, 224, 224])
        frames = np.array([frame[y:y+h, x:x+w] for frame in frames])

        frames = frames[None, :,:,:,:] # (1, 18, 3, 224, 224)
        # frames = frames.transpose(0, 4, 1, 2, 3)
        frames = frames.transpose(0, 2, 1, 3, 4) # (1, 3, 18, 224, 224), batch, channels, time, height, width
        if shorten:
            n_frames = frames.shape[2]
        
            # If fewer than 32 frames, keep all
            if n_frames > 32:
                # Evenly sample 32 frames across the entire sequence
                indices = np.linspace(0, n_frames - 1, 32, dtype=int)
                frames = frames[:, :, indices, :, :]
        #     frames = frames[:, :,::4,:,:]
        return torch.from_numpy(frames).to(self.device)
