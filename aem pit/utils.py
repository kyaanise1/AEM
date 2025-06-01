import torch
import torch.nn.functional as F

def to_one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()
