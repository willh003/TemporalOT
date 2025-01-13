from .constants import get_demo_gif_path
from utils import load_gif_frames, plot_train_heatmap, cosine_distance
import torch
from models import ResNet

def compare(env, task, camera):
    # default gif path for demos
    cost_encoder = ResNet().to('cuda')
    _ = cost_encoder.eval()

    demo_path = get_demo_gif_path(env, task, camera, demo_num=0, num_frames="d") 
    demo = load_gif_frames(demo_path, "torch")
    with torch.no_grad():
        demo_emb = cost_encoder(demo.to('cuda'))
    
    
    cost_matrix = cosine_distance(demo_emb, demo_emb).cpu().numpy()

    img = plot_train_heatmap(cost_matrix, "Cost")
    img.save(f"create_demo/comparisons/compare_{task}_{camera}.png")

    

if __name__=="__main__":
    env_name = "metaworld"
    task_name = "stick-push-v2"
    camera_name = "d"

    compare(env_name, task_name, camera_name)
    
