import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def to_tensor(x):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not isinstance(x,torch.Tensor):
        x = torch.tensor(x) 
    return x.to(dev, torch.get_default_dtype())
