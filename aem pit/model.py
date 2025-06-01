import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.Sigmoid(),
            nn.Linear(256, 112),
            nn.Sigmoid(),
            nn.Linear(112, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)
