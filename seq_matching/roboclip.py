import torch
import numpy as np
from .s3dg import S3D

def compute_roboclip_reward(distance_matrix, current_frames=None, demo_embedding=None):
    """
    Compute rewards using S3D embeddings similarity
    Args:
        distance_matrix: Ignored (kept for API compatibility)
        current_frames: Current episode frames
        demo_frames: Expert demonstration frames
    Returns:
        rewards: numpy array of rewards
        info: dict with debug information
    """
    if current_frames is None or demo_embedding is None:
        raise ValueError("Both current_frames and demo_frames must be provided")

    # Get the global RoboClip encoder instance
    from models.roboclip import get_roboclip_encoder
    encoder = get_roboclip_encoder()
    
    # Get embeddings
    current_embedding = encoder.encode_video(current_frames)
    # demo_embedding = encoder.encode_video(demo_frames)
    
    # Compute similarity
    similarity_matrix = torch.matmul(demo_embedding, current_embedding.t())
    reward = similarity_matrix.detach().cpu().numpy()[0][0]
    
    # Return reward and empty info dict to match API
    return reward, {"assignment": None}
