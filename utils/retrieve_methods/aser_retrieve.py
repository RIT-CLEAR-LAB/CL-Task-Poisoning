import torch
from utils.retrieve_utils import random_retrieve, ClassBalancedRandomSampling, compute_knn_sv, maybe_cuda
import numpy as np



class ASER_retrieve(object):
    def __init__(self, params, size, n_smp_cls, mem_size, **kwargs):
        """
        Initialize ASER retrieval.

        Args:
        - params: Parameters for ASER.
        """
        super().__init__()
        self.num_retrieve = size  # Number of samples to retrieve
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = params.k  # k-NN for Shapley Value calculation
        self.mem_size = mem_size  # Buffer size
        self.aser_type = params.aser_type  # Type of ASER (e.g., "neg_sv", "asv")
        self.n_smp_cls = int(n_smp_cls)  # Number of samples per class
        self.out_dim = 100  # Number of classes
        self.is_aser_upt = False
        ClassBalancedRandomSampling.class_index_cache = None

    def retrieve(self, buffer, **kwargs):
        """
        Retrieve samples using ASER or random retrieval if the buffer is not full.

        Args:
        - buffer: The memory buffer.

        Returns:
        - ret_x (tensor): Retrieved input samples.
        - ret_y (tensor): Retrieved labels.
        """
        model = buffer.model

        # Use random retrieval if the buffer is not full
        
        if buffer.num_seen_examples <= self.mem_size:
            if self.num_retrieve > min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size):
                self.num_retrieve = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
            cur_size = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
            choice = np.random.choice(np.arange(cur_size), size=self.num_retrieve, replace=False)
        else:
            # Use ASER retrieval
            cur_x, cur_y = kwargs['x'], kwargs['y']
            buffer_x, buffer_y = buffer.examples, buffer.labels
            ret_x, ret_y, choice = self._retrieve_by_knn_sv(
                model, buffer_x, buffer_y, cur_x, cur_y, self.num_retrieve
            )
        return choice

    def _retrieve_by_knn_sv(self, model, buffer_x, buffer_y, cur_x, cur_y, num_retrieve):
        """
        Retrieve data instances with top-N Shapley Values from candidate set.

        Args:
        - model (object): Neural network.
        - buffer_x (tensor): Data buffer.
        - buffer_y (tensor): Label buffer.
        - cur_x (tensor): Current input data tensor.
        - cur_y (tensor): Current input label tensor.
        - num_retrieve (int): Number of data instances to retrieve.

        Returns:
        - ret_x (tensor): Retrieved data tensor.
        - ret_y (tensor): Retrieved label tensor.
        """
        cur_x, cur_y = maybe_cuda(cur_x), maybe_cuda(cur_y)

        # Reset ClassBalancedRandomSampling cache if ASER update is not enabled
        if not self.is_aser_upt:
            ClassBalancedRandomSampling.update_cache(buffer_y, self.out_dim)

        # Candidate samples (class balanced subsamples from buffer)
        cand_x, cand_y, cand_ind = ClassBalancedRandomSampling.sample(
            buffer_x, buffer_y, self.n_smp_cls, device=self.device
        )

        # Adversarial SV: Evaluate current input with candidate samples
        sv_matrix_adv = compute_knn_sv(
            model, cur_x, cur_y, cand_x, cand_y, self.k, device=self.device
        )

        # Cooperative SV for ASER types other than "neg_sv"
        if self.aser_type != "neg_sv":
            # Exclude candidate indices from evaluation
            excl_indices = set(cand_ind.tolist())
            eval_coop_x, eval_coop_y, _= ClassBalancedRandomSampling.sample(
                buffer_x, buffer_y, self.n_smp_cls, excl_indices=excl_indices, device=self.device
            )

            # Compute cooperative Shapley values
            sv_matrix_coop = compute_knn_sv(
                model, eval_coop_x, eval_coop_y, cand_x, cand_y, self.k, device=self.device
            )

            # Calculate Shapley values based on ASER type
            if self.aser_type == "asv":
                sv = sv_matrix_coop.max(0).values - sv_matrix_adv.min(0).values
            else:  # Default to mean variation (e.g., "asvm")
                sv = sv_matrix_coop.mean(0) - sv_matrix_adv.mean(0)
        else:
            # For "neg_sv", use only adversarial Shapley values
            sv = sv_matrix_adv.sum(0) * -1

        # Select top-N candidates based on Shapley values
        ret_ind = sv.argsort(descending=True)
        local_ret_ind = sv.argsort(descending=True)[:self.num_retrieve]
        ret_x = cand_x[ret_ind][:num_retrieve]
        ret_y = cand_y[ret_ind][:num_retrieve]
        ret_ind_local = cand_ind[local_ret_ind]

        return ret_x, ret_y, ret_ind_local
