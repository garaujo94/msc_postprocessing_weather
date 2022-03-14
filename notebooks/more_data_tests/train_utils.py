import torch
import numpy as np
from sklearn.metrics import mean_squared_error

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    devide = torch.device('cpu')


def detach_tensor(tensor):
    return tensor.cpu().detach().numpy()


def to_tensor(array):
    return torch.Tensor(array).to(device)


def full_report(model, scaler_y, X_train, X_val, X_test, y_train, y_val, y_test):
    y_train_pred = model(X_train)
    y_val_pred = model(X_val)
    y_test_pred = model(X_test)

    y_train_pred = detach_tensor(y_train_pred)
    y_val_pred = detach_tensor(y_val_pred)
    y_test_pred = detach_tensor(y_test_pred)

    # train
    y_train_pred = scaler_y.inverse_transform(y_train_pred)
    y_train = scaler_y.inverse_transform(y_train)
    print(f'Train ---> RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}')

    # val
    y_val_pred = scaler_y.inverse_transform(y_val_pred)
    y_val = scaler_y.inverse_transform(y_val)
    print(f'Train ---> RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred))}')

    # test
    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    y_test = scaler_y.inverse_transform(y_test)
    print(f'Train ---> RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}')

