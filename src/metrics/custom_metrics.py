from __future__ import division

import torch

from torch.nn.functional import cosine_similarity
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
# from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class CosineSimilarity(Metric):
    """
    Calculates the cosine similarity.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    """
    # @reinit__is_reduced
    def reset(self):
        self._sum_of_cosine_similarities = 0.0
        self._num_examples = 0

    # @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        cos_sim = cosine_similarity(y_pred, y, dim=1, eps=1e-8)
        self._sum_of_cosine_similarities += torch.sum(cos_sim).item()
        self._num_examples += y.shape[0]

    # @sync_all_reduce("_sum_of_cosine_similarities", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'CosineSimilarity must have at least one example'
                'before it can be computed.'
            )
        return self._sum_of_cosine_similarities / self._num_examples
