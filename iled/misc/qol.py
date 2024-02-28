import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def to_tensor(x):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.tensor(x).to(dev, torch.get_default_dtype())
