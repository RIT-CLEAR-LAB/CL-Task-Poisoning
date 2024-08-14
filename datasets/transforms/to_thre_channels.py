import torch


class ToThreeChannels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            x = torch.repeat_interleave(x, 3, dim=0)
        return x
