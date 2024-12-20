from .load_matching_fn import load_matching_fn
import numpy as np

def test():
    names = ["coverage", "log_coverage", "prob", "log_prob", "ot", "temporal_ot", "soft_dtw", "dtw", "even", "final_frame"]

    matching_fn_cfg = {
        "tau": 1,
        "ent_reg": .01,
        "mask_k": 3,
        "sdtw_smoothing": 5
    }

    for name in names:
        print(f"Testing {name}")
        fn = load_matching_fn(name, matching_fn_cfg)   
        cost_matrix = np.random.rand(125, 125)
        rew = fn(cost_matrix)
        assert rew.shape[0] == cost_matrix.shape[0]
        cost_matrix = np.random.rand(125, 20)
        rew = fn(cost_matrix)
        assert rew.shape[0] == cost_matrix.shape[0]

if __name__=="__main__":
    test()