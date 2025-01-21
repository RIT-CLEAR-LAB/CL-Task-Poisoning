import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.retrieve_utils import random_retrieve, get_grad_vector, maybe_cuda
import copy
import numpy as np
# from utils.co2l_loss import SupConLoss


class MIR_retrieve(object):
    def __init__(self, params, size, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = 1000
        self.num_retrieve = size
        self.return_logits = False

    def retrieve(self, buffer, **kwargs):
        # if self.return_logits:
        #     sub_x, sub_y, sub_logits = random_retrieve(buffer, self.subsample, return_logits=True)
        # else:
        if self.num_retrieve > min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size):
            self.num_retrieve = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        cur_size = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        choice = np.random.choice(np.arange(cur_size), size=min(self.subsample, cur_size), replace=False)
        sub_x, sub_y = buffer.examples[choice], buffer.labels[choice]
        grad_dims = []

        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        # print(sub_x.shape)
        # print
        
        
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                logits_pre = buffer.model.forward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1] 
                logits_post = model_temp.forward(sub_x)
                if isinstance(logits_post, tuple):
                    logits_post = logits_post[1]

                pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                
                scores = post_loss - pre_loss
                local_big_ind = scores.sort(descending=True)[1][:self.num_retrieve]
                local_big_ind = local_big_ind.cpu() if local_big_ind.is_cuda else local_big_ind
                big_ind = choice[local_big_ind]  
            return big_ind
        else:
            return choice[:self.num_retrieve]

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        # try:
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.named_parameters(), grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.lr * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for name, param in pp:
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            # print(beg, en) # 0, 540 for PCR
            # print(new_grad[beg: en].shape) # torch.Size([640000]) for ca # torch.Size([540]) for PCR
            # print(param.data.size()) # torch.Size([1000, 640]) for ca # torch.Size([20, 3, 3, 3]) for PCR
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1





class Hardness_retrieve(object):
    def __init__(self, params,  size, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = 200
        self.num_retrieve = size
        self.return_logits = False

    def retrieve(self, buffer, **kwargs):
        # if self.return_logits:
        #     sub_x, sub_y, sub_logits = random_retrieve(buffer, self.subsample, return_logits=True)
        # else:
        if self.num_retrieve > min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size):
            self.num_retrieve = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        cur_size = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        choice = np.random.choice(np.arange(cur_size), size=self.num_retrieve, replace=False)
        sub_x, sub_y = buffer.examples[choice], buffer.labels[choice]
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        # print(sub_x.shape)
        # print
        
        
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            # print(grad_vector)
            # print(grad_vector.shape)
            # exit(0)
            with torch.no_grad():
                logits_pre = buffer.model.forward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1] 
                logits_post = model_temp.forward(sub_x)
                if isinstance(logits_post, tuple):
                    logits_post = logits_post[1]
                # exit(0)
                # pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                # post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                # scores = post_loss - pre_loss
                flip_abs = torch.abs(logits_pre - logits_post)
                flip_sum = torch.sum(flip_abs, dim=1)
                local_big_ind = flip_sum.sort(descending=True)[1][:self.num_retrieve]
                local_big_ind = local_big_ind.cpu() if local_big_ind.is_cuda else local_big_ind
                big_ind = choice[local_big_ind]

            return big_ind
        else:
            return choice

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """

        new_model = copy.deepcopy(model)
        # except RuntimeError as e:
        #     new_model = model.custom_deepcopy()
        # self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        # print(new_model)
        # for name, param in new_model.named_parameters():
        #     print(f"Parameter name: {name}")
        # exit(0)
        self.overwrite_grad(new_model.named_parameters(), grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.lr * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for name, param in pp:
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1



