import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import mean_squared_error

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

class MLP0(nn.Module):
    def __init__(self, in_size, h1, h2):
        super(MLP0, self).__init__()
        self.layer1 = nn.Linear(in_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, 1)

        
    def forward(self, X):
        
        h = self.layer1(X)
        h = torch.tanh(h)
        h = self.layer2(h)
        h = torch.tanh(h)
        h = self.output(h)
        
        
        return h

class MLP1(nn.Module):
    def __init__(self, in_size, h1, h2, h3):
        super(MLP1, self).__init__()
        self.layer1 = nn.Linear(in_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, h3)
        self.output = nn.Linear(h3, 1)
   
        
    def forward(self, X):
        
        h = self.layer1(X)
        h = torch.tanh(h)
        h = self.layer2(h)
        h = torch.tanh(h)
        h = self.layer3(h)
        h = torch.tanh(h)
        h = self.output(h)
        
        
        return h

class CNN(nn.Module):
    def __init__(self, in_size, n_conv, kernel, h1, h2):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=kernel)
        self.layer1 = nn.Linear((in_size - 2)*n_conv, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, 1)

        
    def forward(self, X):
        # To tensor
        X = X.unsqueeze(1)
        
        # Conv layer
        h = self.conv(X)        
        h = h.view(-1, h.size(1)*h.size(2))

        # FC
        h = self.layer1(h)
        h = torch.tanh(h)
        h = self.layer2(h)
        h = torch.tanh(h)
        h = self.output(h)        
        
        return h

class UsefullModel:
    def __init__(self,
                 model
                 ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        print(f'Device --> {self.device}')

    def to_tensor(self, array):
        return torch.Tensor(array).to(self.device)

    def detach_tensor_gpu(self, tensor):
        return tensor.cpu().detach().numpy()

    def fit(self, name, X_train, y_train, X_val, y_val, epochs, optim, loss_function):
        writer = SummaryWriter(f"{name}")
        #writer.add_text('lstm', 'This is an lstm', 0)
        #writer.add_graph(self.model, X_train)
        self.model.train()
        optimizer = optim(self.model.parameters(), lr=1e-3)
        loss_function = loss_function.to(self.device)

        y_train = self.to_tensor(y_train)
        y_val = self.to_tensor(y_val)
        X_train, X_val = self.to_tensor(X_train), self.to_tensor(X_val)

        for e in tqdm(range(epochs)):
            y_pred = self.model(X_train)
            y_prev_val = self.model(X_val)
            
            loss = loss_function(y_pred, y_train.reshape(-1,1))
            val_loss = loss_function(y_prev_val, y_val.reshape(-1,1))
            writer.add_scalar("Loss/train", loss, e) #tensorboard
            writer.add_scalar("Loss/Val", val_loss, e) #tensorboard
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X = self.to_tensor(X)
        y = self.model(X)

        if self.device == torch.device('cuda'):
            y = self.detach_tensor_gpu(y)
        else:
            y = y.numpy()

        return y

    def evaluate(self, 
                 scaler_y,
                 X_train,
                 y_train,
                 X_val = None,
                 y_val = None,
                 X_test = None,                                 
                 y_test = None):

        # train
        X_train = self.to_tensor(X_train)
        y_train_pred = self.model(X_train)
        y_train_pred = self.detach_tensor_gpu(y_train_pred)
        y_train_pred = scaler_y.inverse_transform(y_train_pred)
        y_train = scaler_y.inverse_transform(y_train)
        print(f'Train ---> RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}')

        # val
        if X_val is not None:
            X_val = self.to_tensor(X_val)
            y_val_pred = self.model(X_val)
            y_val_pred = self.detach_tensor_gpu(y_val_pred)
            y_val_pred = scaler_y.inverse_transform(y_val_pred)
            y_val = scaler_y.inverse_transform(y_val)
            print(f'Validation ---> RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred))}')

        # test
        if X_test is not None:
            X_test = self.to_tensor(X_test)
            y_test_pred = self.model(X_test)
            y_test_pred = self.detach_tensor_gpu(y_test_pred)
            y_test_pred = scaler_y.inverse_transform(y_test_pred)
            y_test = scaler_y.inverse_transform(y_test)
            print(f'Test ---> RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}')

    def save(self, name):
        torch.save(self.model.state_dict(), name)
        print(f'Saved as {name}!')

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        print(f'{name} loaded!')

    def plot_example(self, X, y, start, end):
        X = self.to_tensor(X)
        pred = self.model(X[start:end])
        pred = self.detach_tensor_gpu(pred)
        plt.figure(figsize=(12,8))
        plt.plot(pred)
        plt.plot(y[start:end])
        plt.title('Prediction vs Real')
        plt.ylabel('y')
        plt.xlabel('index')
        plt.legend(['previsto', 'real'], loc='upper left')

        plt.show()

    