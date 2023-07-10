import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from firedrake_adjoint import *
from firedrake import *
import sklearn.metrics as metrics

# Define your machine learning model
class MLP_model(nn.Module):
    def __init__(self):
        super(MLP_model, self).__init__()
        # Define your model architecture
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    # Generate a dataset
    df = pd.read_csv('/Users/mh522/Documents/new/graduation design/6.28code/three_point_bending/data.csv')
    X = df['w_max'].values.reshape(-1, 1)
    y = df['force'].values.reshape(-1, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create data loaders for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the model
    model = MLP_model()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        for batch_X, batch_y in train_dataloader:
            # Forward pass
            outputs = model(batch_X)

            # Compute loss
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")

        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0

        predictions = []
        with torch.no_grad():
            for batch_X, batch_y in test_dataloader:
                outputs = model(batch_X)

                predictions.append(outputs.item())
                test_loss += criterion(outputs, batch_y).item()
        
        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}") 
        
        score = metrics.r2_score(y_test_tensor, predictions)
        print("MLP evaluation (score):", score)
