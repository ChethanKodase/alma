import matplotlib.pyplot as plt
import numpy as np
import random

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import zipfile

import shutil
import os
import pandas as pd


'''


conda deactivate
cd alma/beta_tc_vaes/
conda activate /home/luser/anaconda3/envs/inn
python vae_celebA_training.py --which_gpu 0 --beta_value 1.0


conda deactivate
conda deactivate
cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate /home/luser/anaconda3/envs/inn
python vae_celebA_training.py --which_gpu 0 --beta_value 5.0


conda deactivate
conda deactivate
cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate /home/luser/anaconda3/envs/inn
python vae_celebA_training.py --which_gpu 0 --beta_value 10.0

'''


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import argparse

parser = argparse.ArgumentParser(description='VAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--beta_value', type=float, default=1.0, help='Beta VAE beta value')


args = parser.parse_args()

which_gpu = args.which_gpu
beta_value = args.beta_value

device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training



data_directory = '/home/luser/autoencoder_attacks/train_aautoencoders/data_cel1'
data_directory1 = ''+data_directory+'/smile/'
data_directory2 = ''+data_directory+'/no_smile/'

#data_directory2 = '/home/luser/autoencoder_attacks/train_aautoencoders/data_cel1/no_smile/'


img_list = os.listdir(data_directory1)
img_list.extend(os.listdir(data_directory2))


transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])

batch_size = 64
celeba_data = datasets.ImageFolder(data_directory, transform=transform)



train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * 0.8), len(img_list) - int(len(img_list) * 0.8)])
train_data_size = len(train_set)
test_data_size = len(test_set)

print('train_data_size', train_data_size)
print('test_data_size', test_data_size)

trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import VAE, VAE_big, VAE_big_b



image_channels = 3

print('image_channels', image_channels)

#model = VAE(image_channels=image_channels).to(device)

model = VAE_big_b(device, image_channels=image_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) 

#beta_value = 1.0

def loss_fn(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #MSE = F.mse_loss(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta_value *  KLD, BCE, KLD
    #return MSE + beta_value *  KLD, MSE, KLD


epochs = 200

#########################
# training our model
for idx, (image, label) in enumerate(testLoader):
    images, label = image.to(device), label.to(device)
    recon_images, mu, logvar = model(images.to(device))
    break

print("image.shape", image.shape)

print("recon_images.shape", recon_images.shape)

'''for i in range(100):
    with torch.no_grad():
        plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
        plt.show()
        plt.savefig("/home/luser/autoencoder_attacks/test_sets/test_set_analysis/celebA_test_"+str(i)+".png")
        plt.close()'''



#########################

train_loss = []

for epoch in range(epochs):
   
    total_train_loss = 0
    # training our model
    for idx, (image, label) in enumerate(trainLoader):
        images, label = image.to(device), label.to(device)

        recon_images, mu, logvar = model(images.to(device))
        loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
        optimizer.zero_grad()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(images[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        ax[1].imshow(recon_images[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Reconstructed Image')
        ax[1].axis('off')

        plt.show()
        plt.savefig("/home/luser/autoencoder_attacks/a_training_runtime/betaVAE_epoch_"+str(epoch)+"_.png")

    print('loss', loss)
    print("Epoch : ", epoch)


    #torch.save(model.state_dict(), '/home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints/celebA_seeded_clipped_CNN_VAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epoch)+'.torch')