import torch

import torch.nn as nn

#from nvae.utils import add_sn
#from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.jit import script
import pandas as pd
#from nvae.utils import reparameterize
import matplotlib.ticker as ticker


'''

######################################################################################################################################################

conda deactivate
conda deactivate
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/betaVAE_tcVAE_conditioning_analysis.py  --which_gpu 0 --beta_value 5.0 --which_model VAE --checkpoint_storage /home/luser/autoencoder_attacks/saved_celebA/checkpoints


#########################################################################################################################


conda deactivate
conda deactivate
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/betaVAE_tcVAE_conditioning_analysis.py  --which_gpu 1 --beta_value 5.0 --which_model TCVAE --checkpoint_storage /home/luser/autoencoder_attacks/saved_celebA/checkpoints

######################################################################################################################################################


'''


#device = "cuda:1" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
#parser.add_argument('--segment', type=int, default=3, help='Segment index')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--attck_type', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--beta_value', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--which_model', type=str, default=5, help='model to attack')
parser.add_argument('--desired_norm_l_inf', type=float, default=5, help='perturbation norm bounding')
parser.add_argument('--checkpoint_storage', type=str, default="/home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints", help='run time plots directory')


args = parser.parse_args()

#segment = args.segment
which_gpu = args.which_gpu
attck_type = args.attck_type
beta_value = args.beta_value
which_model = args.which_model
desired_norm_l_inf = args.desired_norm_l_inf
device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training
checkpoint_storage = args.checkpoint_storage

from vae import VAE_big, VAE_big_b

#model = VAE_big(device, image_channels=3).to(device)
model = VAE_big(device, image_channels=3).to(device)
train_data_size = 162079
epochs = 199

model_type = which_model


model.load_state_dict(torch.load(''+checkpoint_storage+'/celebA_CNN_'+model_type+''+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

model.eval()

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

    #source_flow = model.fc3(z2)

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
    cond_nums_normalized = cond_nums_array / np.sum(cond_nums_array)
    return cond_nums, cond_nums_normalized


cond_nums, cond_nums_normalized = get_condition_weights(model)

sum = sum(cond_nums_normalized)


filtered_g_cond_nums = [value for value in cond_nums if value != 1.0]

#filtered_g_cond_nums = cond_nums


plt.figure(figsize=(6, 8))  # Set figure size
plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)

plt.ylabel("Block index", fontsize=24)
plt.xlabel("$\kappa$", fontsize=24)

plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))  # Two decimal places


plt.xticks(fontsize=24, rotation=45)
plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)
#plt.xticks(fontsize=14)



plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("./conditioning_analysis/"+which_model+"_conditioning_chart_.png")
plt.show()