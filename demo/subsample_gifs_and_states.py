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

def evenly_subsample_gif_and_states(input_gif_path, output_dir, N, last_frame=None, cut_after_N_consecutive_success=None, input_states_path=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    frames, states = load_frames_and_states(input_gif_path, input_states_path)

    if last_frame is not None:
        frames = frames[:last_frame]
        states = states[:last_frame]
    elif cut_after_N_consecutive_success is not None:
        # Load the success vector
        success = np.load(os.path.splitext(input_gif_path)[0] + "_success.npy")
        print(f"Loaded success vector of length {len(success)}")
        # Find the beginning of the first N consecutive successes
        start_idx = 0
        found_N_consecutive = False
        while start_idx < len(success) - N:
            if np.all(success[start_idx:start_idx + N]):
                found_N_consecutive = True
                break
            start_idx += 1

        if not found_N_consecutive:
            raise ValueError(f"Could not find {N} consecutive successes in {input_gif_path}")
        
        frames = frames[:start_idx + N]
        states = states[:start_idx + N]

    num_frames = len(frames)
    if num_frames != states.shape[0]:
        raise ValueError(f"Mismatch between GIF frames ({num_frames}) and states ({states.shape[0]}) in {base_name}")

    # Subsample evenly
    step = num_frames // N
    selected_indices = list(range(0, num_frames, step))[:N]
    subsampled_frames = [frames[i] for i in selected_indices]
    subsampled_states = states[selected_indices]

    # Save subsampled frames as a new GIF
    file_base_name = f"{base_name}_subsampled_{N}"
    if cut_after_N_consecutive_success:
        file_base_name += f"_cut-after-{cut_after_N_consecutive_success}-success"

    subsampled_gif_path = os.path.join(output_dir, f"{file_base_name}.gif")
    subsampled_states_path = os.path.join(output_dir, f"{file_base_name}_states.npy")

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

# sparse frame indices (5 each)
sparse_frame_indices = {
    "window-open-v2": [5, 7, 21, 25, 27, 31, 38, 56],
    "button-press-v2": [6, 14, 22, 24, 26, 32, 38, 56],
    "door-close-v2": [6, 13, 15, 21, 22, 30, 36, 48],
    "door-open-v2": [6, 10, 16, 20, 22, 24, 31, 62],
    "stick-push-v2": [10, 12, 14, 16, 18, 20, 30, 50],
    "door-lock-v2": [6, 12, 20, 24, 26, 28, 30, 32]
}

first_frame_indices = {
    "window-open-v2": list(range(5)) + list(range(26, 38)) + [49, 50, 51], # window-open try2
    "button-press-v2": list(range(15)) + [31, 43, 44],
    "door-close-v2": list(range(15)) + [31, 43, 44],
    "door-open-v2":  list(range(15)) + [31, 32, 55, 56],
    "stick-push-v2": list(range(10, 25)) + [49, 50, 51],
    "push-v2": list(range(10, 25)) + [49, 50, 51],
    "door-lock-v2": list(range(20)) + [28,29,30] + [61, 62] 
}

if __name__=="__main__":

    env_name = "metaworld"
    task_name = "stick-push-v2"
    camera_name = "d"
    task_name = "door-open-v2"
    camera_name = "corner3"
    mismatched = True

    # denser frame indices (around 20 each)
    #frame_indices = list(range(15)) + [31, 43, 44] # for button-press, door-close
    #frame_indices = list(range(15)) + [31, 32, 55, 56] # for door-open
    #frame_indices = list(range(5)) + list(range(26, 38)) + [49, 50, 51] # window-open try2
    #frame_indices = list(range(20)) + [28,29,30] + [61, 62] # door-lock
    #frame_indices = list(range(10, 25)) + [49, 50, 51] # stick push
    #frame_indices = list(range(10, 25)) + [49, 50, 51] # push

    for task_name in sparse_frame_indices.keys():

        # default gif path for demos
        input_gif_path = get_demo_gif_path(env_name, task_name, camera_name, demo_num=0, num_frames="d") 
        # new gif path
        output_dir = get_demo_dir(env_name, task_name, camera_name, mismatched=mismatched) 

        mismatched_subsample_gifs_and_states(input_gif_path, output_dir, frame_indices=sparse_frame_indices[task_name])
    
    # default gif path for demos
    input_gif_path = get_demo_gif_path(env_name, task_name, camera_name, demo_num=0, num_frames="d") 
    # new gif path
    output_dir = get_demo_dir(env_name, task_name, camera_name, mismatched=mismatched) 
    
    #frame_indices = list(range(15)) + [31, 43, 44] # for button-press, door-close, window-open
    # frame_indices = list(range(5, 18)) + [31, 32, 55, 56, 60, 61] # for lever-pull
    # frame_indices = list(range(5, 18)) + [21, 22, 31, 32, 55, 56] # for hand-insert
    # frame_indices = list(range(7, 20)) + [25, 26, 31, 32, 55, 56] # for basketball
    # frame_indices = list(range(15)) + [31, 32, 55, 56]
    # mismatched_subsample_gifs_and_states(input_gif_path, output_dir, frame_indices=frame_indices)

    #last_frame = 80
    #num_frames = 120
    #output_dir = get_demo_dir(env_name, task_name, camera_name, num_frames=num_frames, mismatched=False) # new gif path
    #evenly_subsample_gif_and_states(input_gif_path, output_dir, num_frames, last_frame=last_frame)

    ###### For subsampling after N consecutive successes
    evenly_subsample_gif_and_states(input_gif_path, output_dir, N=15, last_frame=None, cut_after_N_consecutive_success=5)




