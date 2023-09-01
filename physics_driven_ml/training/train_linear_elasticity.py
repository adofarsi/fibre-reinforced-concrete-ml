from sklearn.model_selection import KFold
import numpy as np
import torch.optim as optim
import os
import torch
import torch.nn as nn

from physics_driven_ml.models.mlp import MLP_model
from physics_driven_ml.models.rnn import RNN_model
from physics_driven_ml.models.cnn1d import CNN1D_model
from physics_driven_ml.utils import ModelConfig
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

WEIGHT_DECAY = 0.001  # Regularization strength


def train_cross_validation(model, dataset, k_folds, epochs, batch_size, lr, criterion, save_folder):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_losses = []
    val_losses = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{k_folds}")

        # Define data subsets for training and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        # Re-initialize model for each fold
        model_fold = model

        optimizer_fold = optim.Adam(model_fold.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        # Training loop for each fold
        for epoch in tqdm(range(epochs)):
            model_fold.train()
            current_train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_fold.zero_grad()
                outputs = model_fold(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_fold.step()
                current_train_loss += loss.item()

            current_val_loss = 0.0
            model_fold.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model_fold(inputs)
                    loss = criterion(outputs, targets)
                    current_val_loss += loss.item()

            train_losses.append(current_train_loss/len(train_loader))
            val_losses.append(current_val_loss/len(val_loader))

            if epoch == epochs - 1:
                print(f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
            # Save model if the validation loss has decreased
            best_val_loss = np.Inf
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model_fold.state_dict(), os.path.join(config.save_folder, config.best_model_name))
    # Plotting after all folds
    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # load data
    load_path = "../../data/datasets/linear_data"
    X_train = np.load(f"{load_path}/X_train.npy", allow_pickle=True)
    X_test = np.load(f"{load_path}/X_test.npy", allow_pickle=True)
    y_train = np.load(f"{load_path}/y_train.npy", allow_pickle=True)
    y_test = np.load(f"{load_path}/y_test.npy", allow_pickle=True)
    # Standardize the data
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    # Compute the mean and standard deviation for the training data
    X_mean = torch.mean(X_train_tensor, dim=0)
    X_std = torch.std(X_train_tensor, dim=0)
    y_mean = torch.mean(y_train_tensor, dim=0)
    y_std = torch.std(y_train_tensor, dim=0)
    # Standardize the training data
    X_train_standardized = (X_train_tensor - X_mean) / X_std
    y_train_standardized = (y_train_tensor - y_mean) / y_std
    # Standardize the test data using the training data's statistics
    X_test_standardized = (X_test_tensor - X_mean) / X_std
    y_test_standardized = (y_test_tensor - y_mean) / y_std

    # Initialize the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model, criterion and optimizer
    config = ModelConfig()
    model = MLP_model(config).to(device)
    criterion = nn.MSELoss()
    # Hyperparameters
    EPOCHS = 300
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dataset loading utilities
    train_dataset = torch.utils.data.TensorDataset(X_train_standardized, y_train_standardized)
    test_dataset = torch.utils.data.TensorDataset(X_test_standardized, y_test_standardized)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    config.save_folder = '../../saved_models'
    config.best_model_name = 'mlp_model.pt'

    # Using the function:
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_cross_validation(model, combined_dataset, k_folds=2, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, criterion=criterion, save_folder=config.save_folder)