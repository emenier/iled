import torch.nn as nn


class TanhPlus(nn.Module):

    def __init__(self):
        super().__init__()

        self.tanh = nn.Tanh()

    def forward(self, x):

        return 0.5 + self.tanh(x) * 0.5
