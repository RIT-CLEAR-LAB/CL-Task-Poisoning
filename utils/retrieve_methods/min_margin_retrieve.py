import torch
import numpy as np
import torch.nn.functional as F

class Min_margin_retrieve(object):
    def __init__(self, params, size, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = 1000
        self.num_retrieve = size  # Number of samples to retrieve
        self.return_logits = False

    def retrieve(self, buffer, **kwargs):
        # Ensure the number to retrieve doesn't exceed buffer size
        if self.num_retrieve > min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size):
            self.num_retrieve = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)

        cur_size = min(buffer.num_seen_examples, buffer.examples.shape[0], buffer.current_size)
        choice = np.random.choice(np.arange(cur_size), size=self.num_retrieve, replace=False)
        sub_x, sub_y = buffer.examples[choice], buffer.labels[choice]

        if sub_x.size(0) > 0:
            with torch.no_grad():
                logits_pre = buffer.model.forward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1]  # Handle case where logits is a tuple

                # Step 1: Retrieve the correct class probabilities
                correct_probs = logits_pre[torch.arange(logits_pre.size(0)), sub_y]

                # Step 2: Mask correct class probabilities and find max incorrect probabilities
                logits_pre_clone = logits_pre.clone()
                logits_pre_clone.scatter_(1, sub_y.unsqueeze(1), -float('inf'))  # Mask correct probabilities
                max_incorrect_probs, _ = logits_pre_clone.max(dim=1)

                # Step 3: Calculate the difference
                diff = max_incorrect_probs - correct_probs

                # Step 4: Filter samples where max_incorrect_probs > correct_probs
                positive_diff_mask = diff > 0  # Select only positive differences
                if positive_diff_mask.any():
                    valid_diff = diff[positive_diff_mask]
                    valid_indices = torch.arange(diff.size(0))[positive_diff_mask]

                    # Step 5: Sort valid differences in ascending order and retrieve top indices
                    sorted_indices = valid_diff.sort()[1][:self.num_retrieve]  # Sort by smallest differences
                    local_big_ind = valid_indices[sorted_indices]

                    # Map back to the original indices in the buffer
                    big_ind = choice[local_big_ind]
                    return big_ind
                else:
                    # If no positive differences exist, return a random choice
                    return choice[:self.num_retrieve]
        else:
            return choice[:self.num_retrieve]
