import os
import numpy as np
from PIL import Image
from .constants import get_demo_dir, get_demo_gif_path

def load_frames_and_states(input_gif_path, input_states_path=None):

    if input_states_path is None: # infer from gif path
        input_states_path =  os.path.splitext(input_gif_path)[0] + "_states.npy"

    assert input_gif_path.endswith(".gif"), "error: reference seq not a gif"
    
    # Load GIF and states
    gif = Image.open(input_gif_path)
    states = np.load(input_states_path)

    # Verify that the number of frames matches the states
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # End of GIF frames

    return frames, states

def save_frames_and_states(frames, gif_path, states, states_path):
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        loop=0
    )
    np.save(states_path, states)

def evenly_subsample_gif_and_states(input_gif_path, output_dir, N, last_frame=None, input_states_path=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    frames, states = load_frames_and_states(input_gif_path, input_states_path)

    if last_frame is not None:
        frames = frames[:last_frame]
        states = states[:last_frame]

    num_frames = len(frames)
    if num_frames != states.shape[0]:
        raise ValueError(f"Mismatch between GIF frames ({num_frames}) and states ({states.shape[0]}) in {base_name}")

    # Subsample evenly
    step = num_frames // N
    selected_indices = list(range(0, num_frames, step))[:N]
    subsampled_frames = [frames[i] for i in selected_indices]
    subsampled_states = states[selected_indices]

    # Save subsampled frames as a new GIF
    subsampled_gif_path = os.path.join(output_dir, f"{base_name}_subsampled_{N}.gif")
    subsampled_states_path = os.path.join(output_dir, f"{base_name}_subsampled_{N}_states.npy")

    save_frames_and_states(subsampled_frames, subsampled_gif_path, subsampled_states, subsampled_states_path)
    print(f"Processed {input_gif_path}: saved subsampled GIF and states to {output_dir}")

# Example usage

def mismatched_subsample_gifs_and_states(input_gif_path, output_dir, frame_indices, input_states_path=None):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    
    frames, states = load_frames_and_states(input_gif_path, input_states_path)
    subsampled_frames = [frames[idx] for idx in frame_indices]
    subsampled_states = [states[idx] for idx in frame_indices]

    subsampled_gif_path = os.path.join(output_dir, f"{base_name}.gif")
    subsampled_states_path = os.path.join(output_dir, f"{base_name}_states.npy")

    save_frames_and_states(subsampled_frames, subsampled_gif_path, subsampled_states, subsampled_states_path)

if __name__=="__main__":

    env_name = "metaworld"
    task_name = "door-open-v2"
    camera_name = "d"
    mismatched = True
    
    # default gif path for demos
    input_gif_path = get_demo_gif_path(env_name, task_name, camera_name, demo_num=0, num_frames="d") 
    # new gif path
    output_dir = get_demo_dir(env_name, task_name, camera_name, mismatched=mismatched) 
    
    #frame_indices = list(range(15)) + [31, 43, 44] # for button-press, door-close, window-open
    frame_indices = list(range(15)) + [31, 32, 55, 56]
    mismatched_subsample_gifs_and_states(input_gif_path, output_dir, frame_indices=frame_indices)

    #last_frame = 80
    #num_frames = 120
    #output_dir = get_demo_dir(env_name, task_name, camera_name, num_frames=num_frames, mismatched=False) # new gif path
    #evenly_subsample_gif_and_states(input_gif_path, output_dir, num_frames, last_frame=last_frame)
