import os

def get_demo_gif_path(env_name, task_name, camera_name, demo_num, num_frames='d', mismatched=False):
    if camera_name=="d":
        camera_name = CAMERA[task_name]

    dir_path = get_demo_dir(env_name, task_name, camera_name, num_frames, mismatched)
    return os.path.join(dir_path, f"{task_name}_{camera_name}_{demo_num}.gif")

def get_demo_dir(env_name, task_name, camera_name, num_frames='d', mismatched=False):
    if camera_name=="d":
        camera_name = CAMERA[task_name]

    if mismatched:
        return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/mismatched")
    elif num_frames == "d":
        return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/default")
    else:
        return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/frames_{num_frames}")


BASE_DEMO_DIR = '/share/portal/wph52/TemporalOT/create_demo'

CAMERA = {
    'button-press-v2': 'corner',
    'door-close-v2': 'corner',
    'hammer-v2': 'corner3',
    'drawer-close-v2': 'corner',
    'drawer-open-v2': 'corner',
    'door-open-v2': 'corner3',
    'bin-picking-v2': 'corner',
    'button-press-topdown-v2': 'corner',
    'door-unlock-v2': 'corner',
    'basketball-v3': 'corner',
    'plate-slide-v2': 'corner',
    'hand-insert-v2': 'corner',
    'peg-insert-side-v2': 'corner3',
    'assembly-v3': 'corner',
    'assembly-v2': 'corner',
    'push-wall-v2': 'corner',
    'soccer-v2': 'corner',
    'disassemble-v2': 'corner',
    'pick-place-wall-v3': 'corner3',
    'pick-place-v2': 'corner3',
    'push-v2': 'corner3',
    'lever-pull-v2': 'corner3', # corner4 in temporalOT paper
    'stick-pull-v2': 'corner3',
    'shelf-place-v2': 'corner',
    'window-close-v2': 'corner3',
    'window-open-v2': 'corner3',
    'reach-v2': 'corner3',
    'button-press-wall-v2': 'corner',
    'box-close-v2': 'corner3',
    'stick-push-v2': 'corner',
    'handle-pull-v2': 'corner3',
    'door-lock-v2': 'corner',
}

MAX_PATH_LENGTH = {
    'button-press-v2': 125,
    'door-close-v2': 125,
    'hammer-v2': 125,
    'drawer-close-v2': 125,
    'drawer-open-v2': 125,
    'door-open-v2': 125,
    'bin-picking-v2': 175,
    'button-press-topdown-v2': 125,
    'door-unlock-v2': 125,
    'basketball-v3': 175,
    'plate-slide-v2': 125,
    'hand-insert-v2': 125,
    'peg-insert-side-v2': 150,
    'assembly-v3': 175,
    'assembly-v2': 175,
    'push-wall-v2': 175,
    'soccer-v2': 125,
    'disassemble-v2': 125,
    'pick-place-wall-v3': 175,
    'pick-place-v2': 125,
    'push-v2': 125,
    'lever-pull-v2': 175,
    'stick-pull-v2': 175,
    'shelf-place-v2': 175,
    'window-close-v2': 125,
    'window-open-v2': 125,
    'reach-v2': 125,
    'button-press-wall-v2': 125,
    'box-close-v2': 175,
    'stick-push-v2': 125,
    'handle-pull-v2': 175,
    'door-lock-v2': 125,
}
