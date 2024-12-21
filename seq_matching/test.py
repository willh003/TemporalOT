from .load_matching_fn import load_matching_fn
from .temporalot import bordered_identity_like
import numpy as np

def test_bordered_identity():

    env_horizon = 120
    mask_k = int(np.random.rand(1) * 5 + 1)
    target_mask = np.triu(np.tril(np.ones((env_horizon, env_horizon)),
                            k=mask_k), k=-mask_k)
    source_mask = bordered_identity_like(env_horizon, env_horizon, mask_k)

    assert (target_mask == source_mask).all
    

def test_all_matching():
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
    test_bordered_identity()
    test_all_matching()