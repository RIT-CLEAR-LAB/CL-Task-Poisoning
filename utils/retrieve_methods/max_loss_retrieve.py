import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.retrieve_utils import random_retrieve, get_grad_vector, maybe_cuda
import copy
import numpy as np
# from utils.co2l_loss import SupConLoss


class Max_loss_retrieve(object):
    def __init__(self, params, size, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = 1000
        self.num_retrieve = size
        self.return_logits = False

    def retrieve(self, buffer, **kwargs):
        if self.num_retrieve > min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size):
            self.num_retrieve = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        cur_size = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        choice = np.random.choice(np.arange(cur_size), size=min(self.subsample, cur_size), replace=False)
        sub_x, sub_y = buffer.examples[choice], buffer.labels[choice]

        # print(sub_x.shape)
        # print
        if sub_x.size(0) > 0:
            with torch.no_grad():
                logits_pre = buffer.model.forward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1] 

                pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                
                local_big_ind = pre_loss.sort(descending=True)[1][:self.num_retrieve]
                local_big_ind = local_big_ind.cpu() if local_big_ind.is_cuda else local_big_ind
                big_ind = choice[local_big_ind]  
            return big_ind
        else:
            return choice[:self.num_retrieve]






