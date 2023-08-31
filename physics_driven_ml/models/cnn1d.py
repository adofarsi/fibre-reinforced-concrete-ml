import torch.nn as nn
from physics_driven_ml.utils import ModelConfig

import torch.nn as nn

class CNN1D_model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(CNN1D_model, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, 3)  # Assuming input is reshaped to (batch_size, 1, 5)
        self.fc1 = nn.Linear(3*16, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape to (batch_size, 1, 5)
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, x):
        return self.forward(x)

