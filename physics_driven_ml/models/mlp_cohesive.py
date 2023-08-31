import os
import torch
import torch.nn.functional as F

from torch import Tensor
import torch.nn as nn
from physics_driven_ml.utils import ModelConfig

class MLP_cohesive_model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MLP_cohesive_model, self).__init__()

        # Define your model architecture
        self.fc1 = nn.Linear(3, 8)  # Input size is 6
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)  # Output size is 4

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def predict(self, x):
        x = self.forward(x)
        return x

