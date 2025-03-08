'''
conda deactivate
conda deactivate
cd autoencoder_attacks/
cd train_aautoencoders/
conda activate inn
python tcvae_celebA_three_non_reduction_layers_THIS_ONE_New.py



conda deactivate
cd alma/beta_tc_vaes/
conda activate /home/luser/anaconda3/envs/inn
python TC_vae_celebA_training.py --which_gpu 0 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-6 --run_time_plot_dir /home/luser/autoencoder_attacks/a_training_runtime --checkpoint_storage /home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints


'''

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import zipfile

import shutil
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import VAE, VAE_big, VAE_big_b
import torch.nn.functional as F
from torch.distributions.normal import Normal



import argparse

parser = argparse.ArgumentParser(description='VAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--beta_value', type=float, default=5.0, help='Beta VAE beta value')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--epochs', type=int, default=200, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-6, help='Beta VAE beta value')
parser.add_argument('--run_time_plot_dir', type=str, default="/home/luser/autoencoder_attacks/a_training_runtime", help='run time plots directory')
parser.add_argument('--checkpoint_storage', type=str, default="/home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints", help='run time plots directory')



args = parser.parse_args()

which_gpu = args.which_gpu
beta_value = args.beta_value
data_directory = args.data_directory
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
run_time_plot_dir = args.run_time_plot_dir
checkpoint_storage = args.checkpoint_storage


device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training



def compute_marginal_entropy(z):
    batch_size, z_dim = z.size()
    z = z.view(-1, z_dim)
    return torch.mean(z, dim=0)

def compute_joint_entropy(mu, logvar):
    batch_size, z_dim = mu.size()
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    log_qz = Normal(mu, torch.exp(0.5 * logvar)).log_prob(z)
    log_qz_sum = torch.sum(log_qz, dim=1)
    return log_qz_sum

def loss_fn(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total Correlation
    batch_size = x.size(0)
    marginal_entropy = compute_marginal_entropy(mu + torch.exp(0.5 * logvar) * torch.randn_like(mu))
    joint_entropy = compute_joint_entropy(mu, logvar)
    TC = torch.mean(joint_entropy) - torch.sum(marginal_entropy)

    # Total loss
    beta = 6.0  # Hyperparameter for balancing TC
    loss = BCE + beta * (KLD + TC)

    return loss, BCE, KLD, TC


#########################################################################



data_directory1 = ''+data_directory+'/smile/'
data_directory2 = ''+data_directory+'/no_smile/'
img_list = os.listdir(data_directory1)
img_list.extend(os.listdir(data_directory2))
transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
celeba_data = datasets.ImageFolder(data_directory, transform=transform)
train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * 0.8), len(img_list) - int(len(img_list) * 0.8)])
train_data_size = len(train_set)
test_data_size = len(test_set)
print('train_data_size', train_data_size)
print('test_data_size', test_data_size)
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)




model = VAE_big_b(device, image_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 


train_loss = []

for epoch in range(epochs):
   
    total_train_loss = 0
    # training our model
    for idx, (image, label) in enumerate(trainLoader):
        images, label = image.to(device), label.to(device)

        recon_images, mu, logvar = model(images.to(device))
        loss, bce, kld, tc = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))

        #loss, bce, kld, tc = loss_fn(recon_images, images, mu, logvar)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
    
    print('loss', loss)
    print("Epoch : ", epoch)
    with torch.no_grad():
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(images[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        ax[1].imshow(recon_images[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Reconstructed Image')
        ax[1].axis('off')

        plt.show()
        plt.savefig(""+run_time_plot_dir+"/tcVAE_epoch_"+str(epoch)+"_.png")

    print('loss', loss)
    print("Epoch : ", epoch)

    torch.save(model.state_dict(), ''+checkpoint_storage+'/celebA_CNN_TCVAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epoch)+'.torch')

