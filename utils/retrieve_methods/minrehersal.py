import torch

class MinRehearsalWithReservoir(object):
    def __init__(self, params, buffer_size, **kwargs):
        """
        Initialize the MinRehearsalWithReservoir class.

        :param params: Object containing necessary parameters.
        """
        self.params = params
        self.rehearsal_counts = torch.zeros(buffer_size, dtype=torch.long)

    def update(self, buffer, x, y, logits=None, **kwargs):
        """
        Update the buffer using reservoir sampling. Tracks minimal rehearsal counts.
        :param buffer: Buffer object to store data.
        :param x: Input data.
        :param y: Input labels.
        :param logits: (Optional) Logits associated with the inputs.
        """
        batch_size = x.size(0)

        # Add whatever still fits in the buffer
        place_left = max(0, buffer.examples.size(0) - buffer.current_index)
        if place_left:
            offset = min(place_left, batch_size)
            buffer.examples[buffer.current_index:buffer.current_index + offset].data.copy_(x[:offset])
            buffer.labels[buffer.current_index:buffer.current_index + offset].data.copy_(y[:offset])
            if logits is not None:
                buffer.buffer_logits[buffer.current_index:buffer.current_index + offset].data.copy_(logits[:offset])

            # Reset rehearsal counts for new samples
            self.rehearsal_counts[buffer.current_index:buffer.current_index + offset] = 0

            buffer.current_index += offset
            buffer.n_seen_so_far += offset

            # If all new samples were added, return the indices of the added samples
            if offset == batch_size:
                return list(range(buffer.current_index - offset, buffer.current_index))

        # Remove samples that are already in the buffer
        x, y = x[place_left:], y[place_left:]
        logits = logits[place_left:] if logits is not None else None

        # Reservoir sampling for the remaining samples
        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, buffer.n_seen_so_far).long()
        valid_indices = (indices < buffer.examples.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        buffer.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return []

        # Map new data to buffer indices
        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        # Replace samples in the buffer
        buffer.examples[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.labels[list(idx_map.keys())] = y[list(idx_map.values())].long()
        if logits is not None:
            buffer.buffer_logits[list(idx_map.keys())] = logits[list(idx_map.values())]

        # Reset rehearsal counts for replaced samples
        self.rehearsal_counts[list(idx_map.keys())] = 0

        return list(idx_map.keys())

    def retrieve(self, size, **kwargs):
        """
        Retrieve samples with the least rehearsal counts.
        :param buffer: Buffer object to retrieve data from.
        :return: Retrieved images, labels, and optionally logits.
        """
        indices = torch.argsort(self.rehearsal_counts)[:size]
        # retrieved_imgs = buffer.examples[indices]
        # retrieved_labels = buffer.labels[indices]
        # retrieved_logits = (
        #     buffer.buffer_logits[indices] if hasattr(buffer, "buffer_logits") else None
        # )

        # Increment rehearsal counts for retrieved samples
        self.rehearsal_counts[indices] += 1

        return indices
