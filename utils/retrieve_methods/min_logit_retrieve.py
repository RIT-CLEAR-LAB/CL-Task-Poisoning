import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.retrieve_utils import random_retrieve, get_grad_vector, maybe_cuda
import copy
import numpy as np
# from utils.co2l_loss import SupConLoss


class Min_logit_retrieve(object):
    def __init__(self, params, size, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = 1000
        self.num_retrieve = size
        self.return_logits = False

    def retrieve(self, buffer, **kwargs):

        cur_size = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        if self.num_retrieve > cur_size:
            self.num_retrieve = cur_size

        # Subsample candidates for efficiency
        choice = np.random.choice(np.arange(cur_size), size=min(self.subsample, cur_size), replace=False)
        sub_x, sub_y = buffer.examples[choice], buffer.labels[choice]

        if sub_x.size(0) > 0:
            with torch.no_grad():
                # Get feature representations
                features = buffer.model.forward(sub_x, returnt='features')
                # Get weights for the last fully connected layer
                weights = buffer.model.classifier.weight  # Assuming fc is the final layer
                weights_y = weights[sub_y]  # Select weights corresponding to the true labels
                
                # Compute distances to the decision boundary
                distances = torch.einsum('ij,ij->i', features, weights_y)  # φ(x)ᵀw^y
                
                # Invert distances for sampling (closer to the boundary -> higher priority)
                distances = 1.0 / (distances + 1e-6)  # Avoid division by zero
                
                # Retrieve the top-k samples with the smallest distances
                local_small_ind = distances.sort(descending=True)[1][:self.num_retrieve]
                local_small_ind = local_small_ind.cpu() if local_small_ind.is_cuda else local_small_ind
                big_ind = choice[local_small_ind]  # Map back to the global indices

            return big_ind
        else:
            return choice[:self.num_retrieve]

    