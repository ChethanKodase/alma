import torch

import torch.nn as nn

import numpy as np






def get_condition_weights(model):
    cond_nums = []
    for layer in model.encoder:
        if isinstance(layer, nn.Conv2d):  
            wt_tensor = layer.weight.detach()
            W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
            U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
            condition_number = S.max() / S.min()
            cond_nums.append(condition_number.item())
        else:
            condition_number = 1.0 # technically it is not zero. But since I want to convert these numbers to weights  I am putting it as zero so that more weight doesnt go to activations llayer losses
            cond_nums.append(condition_number)
    wt_tensor = model.fc1.weight.detach()
    W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
    condition_number = S.max() / S.min()
    cond_nums.append(condition_number.item())

    wt_tensor = model.fc2.weight.detach()
    W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
    condition_number = S.max() / S.min()
    cond_nums.append(condition_number.item())

    wt_tensor = model.fc3.weight.detach()
    W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
    condition_number = S.max() / S.min()
    cond_nums.append(condition_number.item())

    for layer in model.decoder:
        if isinstance(layer, nn.ConvTranspose2d):  
            wt_tensor = layer.weight.detach()
            W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
            U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
            condition_number = S.max() / S.min()
            cond_nums.append(condition_number.item())
        else:
            condition_number = 1.0
            cond_nums.append(condition_number)
    cond_nums_array = np.array(cond_nums)

    cond_cmpli = np.sum(cond_nums_array) - cond_nums_array
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)

    return cond_nums, cond_nums_normalized