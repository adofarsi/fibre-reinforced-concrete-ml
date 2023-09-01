from sklearn.model_selection import KFold
import numpy as np
import torch.optim as optim
import os
import torch
import torch.nn as nn
import pandas as pd

from physics_driven_ml.models.mlp_cohesive import MLP_cohesive_model
from physics_driven_ml.models.rnn_cohesive import RNN_model
from physics_driven_ml.models.cnn1d_cohesive import CNN1D_cohesive_model
from physics_driven_ml.utils import ModelConfig
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

WEIGHT_DECAY = 0.001  # Regularization strength


def train_cross_validation(model, config, dataset, k_folds, epochs, batch_size, lr, criterion):
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
        model_fold = model(config)

        optimizer_fold = optim.Adam(model_fold.parameters(), lr=config.learning_rate, weight_decay=config.WEIGHT_DECAY)

        # model.to(config.device)

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
    train_data = pd.read_csv('../../data/datasets/cohesive_crack_data/cohesive_crack.csv', header=1, names=["sigx", "RF", "CMOD", "disp", "steps"])
    test_data = pd.read_csv('../../data/datasets/cohesive_crack_data/cohesive_crack_test.csv', header=1, names=['sigx', 'RF', 'CMOD', 'disp', 'steps'])
    train_data['E'] = 20e3
    train_data['nu'] = 0.2
    test_data['E'] = 20e3
    test_data['nu'] = 0.2

    noise_level = 0.001  # adjust this to change the noise level
    train_data['E'] = train_data['E'] * (1 + noise_level * np.random.randn(train_data.shape[0]))
    train_data['nu'] = train_data['nu'] * (1 + noise_level * np.random.randn(train_data.shape[0]))
    test_data['E'] = test_data['E'] * (1 + noise_level * np.random.randn(test_data.shape[0]))
    test_data['nu'] = test_data['nu'] * (1 + noise_level * np.random.randn(test_data.shape[0]))

    X_train_tensor = torch.tensor(train_data[['steps', 'E', 'nu']].values, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_data[['CMOD', 'sigx']].values, dtype=torch.float32)
    X_test_tensor = torch.tensor(test_data[['steps', 'E', 'nu']].values, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_data[['CMOD', 'sigx']].values, dtype=torch.float32)

    # Compute the mean and standard deviation for the training data
    X_mean = torch.mean(X_train_tensor, dim=0)
    X_std = torch.std(X_train_tensor, dim=0)
    y_mean = torch.mean(y_train_tensor, dim=0)
    y_std = torch.std(y_train_tensor, dim=0)
    X_train_standardized = (X_train_tensor - X_mean) / X_std
    y_train_standardized = (y_train_tensor - y_mean) / y_std
    X_test_standardized = (X_test_tensor - X_mean) / X_std
    y_test_standardized = (y_test_tensor - y_mean) / y_std

    # Initialize the device for training
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model, criterion and optimizer
    model = MLP_cohesive_model(config).to(device)
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
    config.best_model_name = 'cohesive_model.pt'
    # Using the function:
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_cross_validation(MLP_cohesive_model, config, combined_dataset, k_folds=2, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, criterion=criterion)
