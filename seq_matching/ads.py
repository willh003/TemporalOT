import torch
import numpy as np
import math
from utils import cosine_distance
import matplotlib.pyplot as plt


class AutomaticDiscountScheduling:
    def __init__(self, horizon, alpha, threshold, progress_start, max_progress_delta, ref_score_percentile, agent_score_percentile, device):
        self.horizon = horizon
        self.alpha = alpha
        self.threshold = threshold
        self.progress_start = progress_start
        self.max_progress_delta = max_progress_delta
        self.ref_score_percentile = ref_score_percentile
        self.agent_score_percentile = agent_score_percentile
        self.device = device

        self.progress = int(progress_start * horizon)

    def init_demos(self, demo_embeddings):
        scores = []
        for i in range(len(demo_embeddings)):
            for j in range(len(demo_embeddings)):
                # commented so it will always work, even with just 1 demo
                # if j != i:
                demo1 = demo_embeddings[i]
                demo2 = demo_embeddings[j]

                cost_matrix = cosine_distance(demo1, demo2)

                score = torch.zeros(cost_matrix.shape[0])
                for k in range(cost_matrix.shape[0]):
                    pos = cost_matrix[:k + 1, :k + 1].min(1)[1]
                    score[k] = longest_increasing_subsequence(pos)
                scores.append(score)

        scores = np.stack(scores, axis=0)
        self.ref_score = np.percentile(scores, self.ref_score_percentile, axis=0)

        plt.clf()
        plt.bar(range(self.ref_score.shape[0]), self.ref_score)
        plt.savefig(f'ref_score')
        return

    def get_discount(self):
        discount = math.exp(math.log(self.alpha) / self.progress)
        return discount

    def update(self, cost_matrixs):
        for i in range(self.max_progress_delta):
            match_scores = []
            for cost in cost_matrixs:
                pos = cost[:self.progress, :self.progress].argmin(1)
                match_scores.append(longest_increasing_subsequence(pos))
            match_score = np.percentile(match_scores, self.agent_score_percentile)
            ref_score = self.ref_score[self.progress - 1]
            if self.progress < self.horizon:
                if match_score >= int(self.threshold * ref_score):
                    self.progress += 1
            else:
                break
        discount = self.get_discount()
        metrics = {
            'discount': discount,
            'discount_log': math.log(discount),
            'match_score': match_score,
            'ref_score': ref_score,
            'progress': self.progress
        }
        return discount, metrics


def longest_increasing_subsequence(a):
    dp = np.ones(a.shape[0])
    answer = 1
    for i in range(1, a.shape[0]):
        for j in range(i):
            if a[j] < a[i]:
                dp[i] = max(dp[i], dp[j] + 1)
        answer = max(answer, dp[i])
    return answer