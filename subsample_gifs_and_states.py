import os
import numpy as np
from PIL import Image

def subsample_gif_and_states(input_gif_path, output_dir, N, last_frame=None, input_states_path=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if input_states_path is None: # infer from gif path
        input_states_path =  os.path.splitext(input_gif_path)[0] + "_states.npy"

    assert input_gif_path.endswith(".gif"), "error: reference seq not a gif"
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    
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
    subsampled_frames[0].save(
        subsampled_gif_path,
        save_all=True,
        append_images=subsampled_frames[1:],
        loop=0
    )

    # Save subsampled states
    subsampled_states_path = os.path.join(output_dir, f"{base_name}_subsampled_{N}_states.npy")
    np.save(subsampled_states_path, subsampled_states)

    print(f"Processed {input_gif_path}: saved subsampled GIF and states to {output_dir}")

# Example usage

if __name__=="__main__":
    # input_gif_path = "/share/portal/hw575/CrossQ/train_logs/2024-11-25-124432_sb3_sac_envt=door-close-v2-goal-hidden_rm=hand_engineered_nt=ep-len=200_sparse/eval/1000000_rollouts.gif"
    # output_dir = "ref_seqs/door_close"

    input_gif_path="/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hammer-v2/hammer-v2_corner3_0.gif"
    output_dir = "../../ref_seqs/door_close_v2"
    
    N = 20
    last_frame = 80

    subsample_gif_and_states(input_gif_path, output_dir, N, last_frame=last_frame)