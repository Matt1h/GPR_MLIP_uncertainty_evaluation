import numpy as np
import torch


def mae(y_test, y_pred):
    if isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        loss = torch.abs(torch.sub(y_pred, y_test)).mean()
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, np.ndarray):
        y_test = y_test.detach().numpy()
        loss = np.mean(np.abs(y_pred - y_test))
    elif isinstance(y_test, np.ndarray) and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().numpy()
        loss = np.mean(np.abs(y_pred - y_test))
    else:
        loss = np.mean(np.abs(y_pred - y_test))
    return loss.detach()
