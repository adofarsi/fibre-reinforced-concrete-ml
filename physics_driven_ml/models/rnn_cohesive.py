import torch
import torch.nn as nn
from physics_driven_ml.utils import ModelConfig


class RNN_model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(RNN_model, self).__init__()

        self.rnn = nn.RNN(input_size=3, hidden_size=8, batch_first=True)    # Assuming input is reshaped to (batch_size, 5, 1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x, _ = self.rnn(x.unsqueeze(1))    # Reshape
        x = self.fc(x[:, -1, :])           # Use the last RNN output
        return x

    def predict(self, x):
        return self.forward(x)

