import torch
import torch.nn.functional as F
import numpy as np


class Min_confidence_retrieve(object):
    def __init__(self, params, size, **kwargs):
        """
        Initialize Min Confidence Retrieve.

        Args:
        - params: Parameters for the method.
        - size: Number of samples to retrieve.
        """
        super().__init__()
        self.params = params
        self.subsample = 1000  # Number of candidates to consider for retrieval
        self.num_retrieve = size  # Number of samples to retrieve
        self.return_logits = False  # Option to return logits (if needed)

    def retrieve(self, buffer, **kwargs):
        """
        Retrieve samples with the lowest confidence scores.

        Args:
        - buffer: The buffer storing examples, labels, and the model.

        Returns:
        - big_ind: Indices of the retrieved samples.
        """
        # Determine current buffer size
        cur_size = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        if self.num_retrieve > cur_size:
            self.num_retrieve = cur_size

        # Subsample candidates for efficiency
        choice = np.random.choice(np.arange(cur_size), size=min(self.subsample, cur_size), replace=False)
        sub_x, sub_y = buffer.examples[choice], buffer.labels[choice]

        if sub_x.size(0) > 0:
            with torch.no_grad():
                # Get model predictions (logits)
                logits_pre = buffer.model.forward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1]

                # Calculate confidence scores (max softmax value for each sample)
                softmax_scores = F.softmax(logits_pre, dim=1)
                confidences, _ = torch.max(softmax_scores, dim=1)

                # Sort indices based on ascending confidence scores (lower confidence prioritized)
                local_low_conf_ind = confidences.sort()[1][:self.num_retrieve]
                local_low_conf_ind = local_low_conf_ind.cpu() if local_low_conf_ind.is_cuda else local_low_conf_ind
                big_ind = choice[local_low_conf_ind]  # Map back to global indices

            return big_ind
        else:
            return choice[:self.num_retrieve]
