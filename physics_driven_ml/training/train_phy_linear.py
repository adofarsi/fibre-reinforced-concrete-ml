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
from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import torch_operator

from physics_driven_ml.evaluation import evaluate

WEIGHT_DECAY = 0.001  # Regularization strength

def solve_pde(input_data):
    #print(input_data)
    input_data = input_data * X_std + X_mean
    E = input_data[0][0]
    nu = input_data[0][1]
    exx, eyy, exy = input_data[0][2], input_data[0][3], input_data[0][4]
    strain_tensor = np.array([[exx, exy], [exy, eyy]])
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    # lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v), 0.1 is the trace of eps(v)
    s = lmbda*np.trace(strain_tensor)*np.eye(2) + 2*mu*strain_tensor
    output = torch.tensor([s[0,0], s[1,1], s[0,1]], dtype=torch.float32)
    stress = (output - y_mean) / y_std
    return stress


def train_phy(model, config: ModelConfig, dataset, solver):
    """Train the model on a given dataset."""
    criterion = nn.MSELoss()
    
    best_error = 0.
    kfold = KFold(n_splits=config.K_fold, shuffle=True)
    train_losses = []
    val_losses = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{config.K_fold}")
        
        # Define data subsets for training and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=val_subsampler)
        
        # Re-initialize model for each fold
        # model = model()
        model.to(config.device)
        optimizer_fold = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.WEIGHT_DECAY)
        
        # training loop
        loss_phy_total = 0.0
        loss_ml_total = 0.0
        for epoch in tqdm(range(config.epochs)):
            model.train()
            current_train_loss = 0.0
            for inputs, targets in train_loader:
                # input=E,nu,strain[0,1,2], target=stress[0,1,2]
                inputs, targets = inputs, targets
                optimizer_fold.zero_grad()
                outputs = model(inputs)                
                
                # calculate machine learning loss
                loss_ml = criterion(outputs, targets)
                loss_ml_total += loss_ml
                # calculate physics loss
                output_phy = solver(inputs)
                loss_phi = criterion(output_phy, targets)  
                loss_phy_total += loss_phi                
                
                # back propagation
                loss = 0.1 * loss_phi + loss_ml
                loss.backward()
                
                optimizer_fold.step()
                current_train_loss += loss.item()
                
            # validation loop
            model.eval()
            current_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs, targets
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    current_val_loss += loss.item()
            
            # print current losses
            train_losses.append(current_train_loss/len(train_loader))
            val_losses.append(current_val_loss/len(val_loader))
            if epoch == config.epochs - 1:
                print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {current_train_loss/len(train_loader)} | Validation Loss: {current_val_loss/len(val_loader)}")

            # Save best-performing model
            if current_val_loss < best_error or epoch == 0:
                best_error = current_val_loss
                torch.save(model.state_dict(), os.path.join(config.save_folder, config.best_model_name))
    # plot losses
    plt.figure(figsize=(12,4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()    

    return model

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



    # args = parser.parse_args()
    config = ModelConfig()
    config.batch_size = 16
    train_dataset = torch.utils.data.TensorDataset(X_train_standardized, y_train_standardized)
    test_dataset = torch.utils.data.TensorDataset(X_test_standardized, y_test_standardized)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)
    #save_folder = 'saved_models'

    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.save_folder = '../../saved_models'
    config.best_model_name = 'physics_RNN_model.pt'

    # Using the function:
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_model = RNN_model(config).to(config.device)
    train_phy(train_model, config, combined_dataset, solver=solve_pde)