import numpy as np

def compute_log_probability_reward(cost_matrix, tau=1):
    """
    see overleaf
    hypothesis: tau should be ref seq length * max distance between two frames
    """
    
    # max_probs[i, j] represents the max probability that reference j was reached at any timestep before i
    # this is a lower bound on the total probability that reference j was reached by timestep i
    min_cost = np.zeros_like(cost_matrix)
    min_cost[0, :] = cost_matrix[0, :]
    min_cost[:, -1] = cost_matrix[:, -1]
    for i in range(1, min_cost.shape[0]):
        for j in range(min_cost.shape[1] - 1): # we want current probability of being in the last reference, not max so far (discourage moving out of final state)
            min_cost[i, j] = min(min_cost[i-1, j], cost_matrix[i, j]) # monotonically increasing probability matrix
                
    cumulative_cost = np.zeros_like(min_cost)
    cumulative_cost[:, 0] = min_cost[:,0]
    for i in range(min_cost.shape[0]):
        for j in range(1, min_cost.shape[1]):
            cumulative_cost[i, j] = min_cost[i,j] + cumulative_cost[i, j-1] 

    # cumulative_cost bounded above by tau (at most d for each reference subgoal, tau = d*len(ref))
    final_reward = 1 - (1/tau) * cumulative_cost[:,  -1] 
    
    return final_reward, cumulative_cost

def compute_probability_reward(cost_matrix, tau=1):
    probability_matrix = np.exp(-cost_matrix/tau)
    
    # max_probs[i, j] represents the max probability that reference j was reached at any timestep before i
    # this is a lower bound on the total probability that reference j was reached by timestep i
    max_probs = np.zeros_like(probability_matrix)
    max_probs[0, :] = probability_matrix[0, :]
    max_probs[:, -1] = probability_matrix[:, -1]
    for i in range(1, max_probs.shape[0]):
        for j in range(max_probs.shape[1] - 1): # we want current probability of being in the last reference, not max so far (discourage moving out of final state)
            max_probs[i, j] = max(max_probs[i-1, j], probability_matrix[i, j]) # monotonically increasing probability matrix
                
    # cumulative_probs[i, j] represents a lower bound on the probability that reference j and all previous references were reached by timestep i
    cumulative_probs = np.zeros_like(probability_matrix)
    cumulative_probs[:, 0] = max_probs[:,0]
    #cumulative_probs[:, 0] = np.log(max_probs[:,0])
    for i in range(max_probs.shape[0]):
        for j in range(1, max_probs.shape[1]):
            # TODO: this can be converted to a sum of log probs if numerical instability occurs
            # especially likely to happen with a large number of reference states
            cumulative_probs[i, j] = max_probs[i,j] * cumulative_probs[i, j-1] 
            #cumulative_probs[i, j] = np.log(max_probs[i,j]) + cumulative_probs[i, j-1] # sum log probs to prevent numerical instability

    final_reward = cumulative_probs[:,  -1] # only because we are doing np.log( ... np.exp())
    
    return final_reward, cumulative_probs