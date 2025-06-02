import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 112),
            nn.BatchNorm1d(112),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(112, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)
