from .coverage import compute_coverage_reward, compute_log_coverage_reward
from .dtw import compute_dtw_reward
from .soft_dtw import compute_soft_dtw_reward
from .even_distribution import compute_even_distribution_reward
from .final_frame import compute_final_frame_reward
from .optimal_transport import compute_ot_reward
from .prob import compute_log_probability_reward, compute_probability_reward
from .temporalot import compute_temporal_ot_reward

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
        fn = lambda cost_matrix: compute_temporal_ot_reward(cost_matrix, mask_k=float(fn_config.get("mask_k",2)), ent_reg=float(fn_config.get("ent_reg", 0.01)))
    elif fn_name == "soft_dtw":
        fn = lambda cost_matrix : compute_soft_dtw_reward(cost_matrix, smoothing=float(fn_config.get("sdtw_smoothing", 0.01)))
    elif fn_name == "dtw":
        fn = compute_dtw_reward
    elif fn_name == "even":
        fn = compute_even_distribution_reward
    elif fn_name == "final_frame":
        fn = compute_final_frame_reward
    else:
        raise Exception(f"Invalid fn {fn_name}")
    
    return fn