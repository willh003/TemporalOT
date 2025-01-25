from .coverage import compute_coverage_reward, compute_log_coverage_reward
from .dtw import compute_dtw_reward, dtw_progress_tracker
from .soft_dtw import compute_soft_dtw_reward
from .even_distribution import compute_even_distribution_reward
from .final_frame import compute_final_frame_reward
from .optimal_transport import compute_ot_reward
from .prob import compute_log_probability_reward, compute_probability_reward
from .temporalot import compute_temporal_ot_reward
from .threshold import compute_tracking_with_threshold_reward
from .roboclip import compute_roboclip_reward

def load_matching_fn(fn_name, fn_config):
    fn = None 

    if fn_name == "coverage":
        fn = lambda cost_matrix: compute_coverage_reward(cost_matrix, tau=float(fn_config.get("tau", 1)))
    elif fn_name == "log_coverage":
        fn = lambda cost_matrix: compute_log_coverage_reward(cost_matrix, tau=float(fn_config.get("tau", 1)))
    elif fn_name == "prob":
        fn = lambda cost_matrix: compute_probability_reward(cost_matrix, tau=float(fn_config.get("tau", 1)))
    elif fn_name == "log_prob":
        fn = lambda cost_matrix: compute_log_probability_reward(cost_matrix, tau=float(fn_config.get("tau", 1)))
    elif fn_name == "ot":
        fn = lambda cost_matrix: compute_ot_reward(cost_matrix, ent_reg=float(fn_config.get("ent_reg", .01)))
    elif fn_name == "temporal_ot":
        fn = lambda cost_matrix: compute_temporal_ot_reward(cost_matrix, mask_k=float(fn_config.get("mask_k",10)), ent_reg=float(fn_config.get("ent_reg", 0.01)))
    elif fn_name == "soft_dtw":
        fn = lambda cost_matrix : compute_soft_dtw_reward(cost_matrix, smoothing=float(fn_config.get("sdtw_smoothing", 5)))
    elif fn_name == "dtw":
        fn = compute_dtw_reward
    elif fn_name == "even":
        fn = lambda cost_matrix: compute_even_distribution_reward(cost_matrix, mask_k=float(fn_config.get("mask_k",0)))
    elif fn_name == "final_frame":
        fn = compute_final_frame_reward
    elif fn_name == "threshold":
        fn = lambda cost_matrix: compute_tracking_with_threshold_reward(cost_matrix, threshold=float(fn_config.get("threshold", 0.75)))
    elif fn_name == "roboclip":
        fn = compute_roboclip_reward
    else:
        raise Exception(f"Invalid fn {fn_name}")

    # Note@Roboclip: No need to modify since track_progress=False in the config    
    if fn_config.get('track_progress', False):

        def tracking_fn(cost_matrix, lookahead=10):
            final_reward, info = fn(cost_matrix)
            progress = dtw_progress_tracker(cost_matrix) + 1
            info["progress"] = progress
            return final_reward, info
        
        return tracking_fn

    return fn