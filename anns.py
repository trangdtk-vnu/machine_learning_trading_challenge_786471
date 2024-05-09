# -*- coding: utf-8 -*-
"""anns.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eVsQjJZys64an2cgbzUmvSkwX9RSotIF

## Libraries
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
from scipy.stats import truncnorm
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import TensorDataset
from torch.nn.functional import mse_loss as MSELoss

"""# Custom Loss Function"""

"""# custom_loss function"""
# class EnhancedSignAgreementLoss extends nn.Module
#, which is the base class for all neural network modules in PyTorch
class EnhancedSignAgreementLoss(nn.Module):
#
    def __init__(self, loss_penalty):
        super(EnhancedSignAgreementLoss, self).__init__()
        self.loss_penalty = loss_penalty

    def forward(self, y_true, y_pred):
        # Check if signs of y_true and y_pred are the same (including zero)
        same_sign = torch.eq(torch.sign(y_true), torch.sign(y_pred))

        # Calculate the residual (difference between y_true and y_pred)
        residual = y_true - y_pred

        # Compute the loss based on the condition
        loss = torch.where(same_sign,
                           torch.square(residual),
                           torch.square(residual) + self.loss_penalty)

        # Return the mean loss
        return torch.mean(loss)

"""# RMSE"""

"""# RMSE"""

# Define RMSE loss function
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))

"""# RNN"""

# Create sequences for RNNs
def create_sequences_rnns(X, y, time_steps):
    #create an empty list to store inputs and outputs
    Xs, ys = [], []
    # loop through the array to the end minus the number of time steps
    # - time_steps in the loop is to ensure that each input sequence created has
    # a corresponding future data point
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

"""## RNNs architecture"""

# Define a simple RNN class and initialize it with __init__ with needed parameters
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        # batch_first = True means that the input tensor will have batch size as first dimension

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # define a fully connected layer to transform the output from the RNN
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # Outputs from all timesteps
        out = self.fc(out[:, -1, :])  # Last timestep output to fully connected layer
        return out.squeeze()  # Remove extra dimension to match target shape

def rnns(model, train_loader, val_loader, epochs, optimizer, loss_function):
    best_val_loss = float('inf') # initialize best validation loss to infity for comparison
    best_model_state = None #variable to store the best model
    # loop for n training epochs
    for epoch in range(epochs):
        model.train()
        #iterate over each batch in training dataset
        for inputs, labels in train_loader:
            optimizer.zero_grad() # clear gradients before computing them
            outputs = model(inputs) #generate predictions
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            #Calculate the loss, apply backpropagation to compute gradients and uodate weights
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval() #evaluate the model (disables dropout etc.)
        total_val_loss = 0
        #do not compute gradients
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                val_loss = loss_function(outputs, labels)
                total_val_loss += val_loss

        avg_val_loss = total_val_loss / len(val_loader) #calculate the average loss for each epoch
        #check if avg loss is the best and update it if needed
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save the best model state

        if epoch % 10 == 0:
            #print progress every 10 epochs
            print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Val Loss: {avg_val_loss}')

    return best_model_state, best_val_loss

"""# ANFIS RMSE"""

## ANFIS RMSE


class ANFIS_RMSE(nn.Module):
#initialization of class with parameters function and input dimention
    def __init__(self, functions, input_dim):
        super(ANFIS_RMSE, self).__init__()
        self.functions = functions
        self.input_dim = input_dim
        # mu and sigma are learnable parameters which shape the membership function
        self.mu = nn.Parameter(torch.randn(input_dim, functions) * 0.1)
        self.sigma = nn.Parameter(torch.randn(input_dim, functions) * 0.1)
        # linear layer that maps the output of fuzzy fuction into one single output
        self.linear = nn.Linear(functions, 1)
        self.best_model_state = None  # To save the best model state

    def forward(self, x):
        batch_size = x.size(0) # batch size = number of observations
        # adapt mu and sigma to the batch dimension
        mu = self.mu.expand(batch_size, -1, -1)
        sigma = torch.exp(self.sigma).expand(batch_size, -1, -1)
        # we add an extra dimension to x: x(batch_size, input_dim) => x(batch_size, input_dim, number of fuctions)
        x = x.unsqueeze(2).expand(-1, -1, self.functions)
        # Create Gaussians membership values for each input (observation)
        gaussians = torch.exp(-torch.pow(x - mu, 2) / (2 * sigma.pow(2)))
        # t_norm defines how strongly each observation meets the criteria of each set of rule
        # by taking the product of the membership values across the features involves in that set of rule
        t_norm = torch.prod(gaussians, dim=1)
        # Applying linear transformation to t_norm into one single output
        y = self.linear(t_norm)
        # one output for each example of batch size
        return y.squeeze(1)

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, lr_step=10, lr_gamma=0.95):
      # Fit method trains the model using the provided training data and validate
      # it using the validation data, running for n set epochs with the lr.
      # adjust lr each 10 epochs and for a rate of 95% of the previous lr
        train_dataset = TensorDataset(X_train, y_train)
        # shuffle train data to ensure that model does not learn any pattern in which
        # the examples presented
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataset = TensorDataset(X_val, y_val)
        # not shuffle val data ensure the model is evaluated in a consistent way each time
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        # the optimizer here is Adam
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
        loss_function = nn.MSELoss()
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(x_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * x_batch.size(0)

            scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # Validation phase
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    outputs = self.forward(x_batch)
                    val_loss = loss_function(outputs, y_batch)
                    total_val_loss += val_loss.item() * x_batch.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)

            # Update best model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_model_state = self.state_dict()

            # Print losses
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        print(f"Best Validation Loss: {best_val_loss:.4f}")

    def predict(self, X):
        self.load_state_dict(self.best_model_state)
        self.eval()
        with torch.no_grad():
            return self.forward(X)

"""# ANFIS custom loss function"""

"""## ANFIS custom loss function"""

class ANFIS_CustomLoss(nn.Module):
    def __init__(self, functions, input_dim, loss_penalty):
        super(ANFIS_CustomLoss, self).__init__()
        self.functions = functions
        self.input_dim = input_dim
        self.loss_penalty = loss_penalty
        self.mu = nn.Parameter(torch.randn(input_dim, functions) * 0.1)  # Improved initialization
        self.sigma = nn.Parameter(torch.randn(input_dim, functions) * 0.1)  # Improved initialization
        self.linear = nn.Linear(functions, 1)

    def forward(self, x):
        batch_size = x.size(0)
        mu = self.mu.expand(batch_size, -1, -1)
        sigma = torch.exp(self.sigma).expand(batch_size, -1, -1)
        x = x.unsqueeze(2).expand(-1, -1, self.functions)
        gaussians = torch.exp(-torch.pow(x - mu, 2) / (2 * sigma.pow(2)))
        t_norm = torch.prod(gaussians, dim=1)
        y = self.linear(t_norm)
        return y.squeeze(1)

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, lr_step=10, lr_gamma=0.95):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
        loss_function = EnhancedSignAgreementLoss(self.loss_penalty)
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(x_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * x_batch.size(0)

            scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader.dataset)

            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    outputs = self(x_batch)
                    val_loss = loss_function(outputs, y_batch)
                    total_val_loss += val_loss.item() * x_batch.size(0)
            avg_val_loss = total_val_loss / len(val_loader.dataset)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.state_dict()

            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if best_model_state:
            self.load_state_dict(best_model_state)
            print(f'Best Validation Loss: {best_val_loss:.4f}')

        return best_val_loss  # Make sure to return the best validation loss

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)