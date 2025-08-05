import torch

import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.jit import script
import pandas as pd
from vae import VAE_big, VAE_big_b
import torch.optim as optim

from conditioning import get_condition_weights
from attacks import get_latent_space_l2_loss, get_latent_space_wasserstein_loss, get_latent_space_SKL_loss, get_latent_space_cosine_loss, output_attack_l2, output_attack_wasserstein, output_attack_SKL, output_attack_cosine, get_weighted_combinations_k_eq_latent_l2, get_weighted_combinations_k_eq_latent_wasserstein, get_weighted_combinations_k_eq_latent_SKL, get_weighted_combinations_k_eq_latent_cos, get_weighted_combinations_l2_aclmd_l2_cond, get_weighted_combinations_aclmd_wasserstein_cond, get_weighted_combinations_aclmd_SKL_cond, get_weighted_combinations_aclmd_cos_cond
'''

######################################################################################################################################################

conda deactivate
conda deactivate
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_wasserstein --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_SKL --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_cosine --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_wasserstein --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_SKL --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_cosine --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_wass --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_skl --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_cos --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type grill_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type grill_wass --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type grill_skl --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type grill_cos --which_model TCVAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage

TCVAE


'''

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
#parser.add_argument('--segment', type=int, default=3, help='Segment index')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--attck_type', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--beta_value', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--which_model', type=str, default=5, help='model to attack')
parser.add_argument('--desired_norm_l_inf', type=float, default=5, help='perturbation norm bounding')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--model_location', type=str, default=0, help='model directory')
parser.add_argument('--num_steps', type=int, default=300, help='Index of the GPU to use (0-N)')
parser.add_argument('--runtime_plots_location', type=str, default=0, help='model directory')
parser.add_argument('--attack_store_location', type=str, default=0, help='model directory')



args = parser.parse_args()

which_gpu = args.which_gpu
attck_type = args.attck_type
beta_value = args.beta_value
which_model = args.which_model
desired_norm_l_inf = args.desired_norm_l_inf
data_directory = args.data_directory
model_location = args.model_location
num_steps = args.num_steps
runtime_plots_location = args.runtime_plots_location
attack_store_location = args.attack_store_location

device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training


model = VAE_big(device, image_channels=3).to(device)
train_data_size = 162079
epochs = 199
model_type = which_model
model.load_state_dict(torch.load(''+model_location+'/celebA_CNN_'+model_type+''+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
model.eval()


noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0
optimizer = optim.Adam([noise_addition], lr=0.0001)
noise_addition.requires_grad = True


with torch.no_grad():
    cond_nums, cond_normal = get_condition_weights(model)

criterion = nn.MSELoss()


def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, outputChange: {instability}, deviation: {deviation}")
    print()
    print("attack type", attck_type)    
    adv_div_list.append(deviation.item())
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(normalized_attacked[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(scaled_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig(""+runtime_plots_location+"/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, ""+attack_store_location+"/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".pt")
    #np.save("/home/luser/autoencoder_attacks/robustness_eval_saves_univ/adversarial_div_convergence/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".npy", adv_div_list)
    plt.close()

    plt.plot(adv_div_list)
    plt.savefig(''+runtime_plots_location+'/deviation_attack_type_'+attck_type+'_desired_norm_l_inf_'+str(desired_norm_l_inf)+'_.png')
    plt.close()


batch_size = 50


import random
from torchvision import datasets, transforms
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


if(attck_type == "latent_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
            break
        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_cosine"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize = get_latent_space_cosine_loss(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_latent_space_wasserstein_loss(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_latent_space_SKL_loss(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "lma_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_l2(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "lma_wass"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_wasserstein(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "lma_skl"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "lma_cos"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_cos(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "output_attack_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_l2(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "output_attack_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_wasserstein(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "output_attack_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_SKL(normalized_attacked, source_im, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "output_attack_cosine"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_cosine(normalized_attacked, source_im, model)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "grill_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im, cond_normal, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "grill_wass"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_aclmd_wasserstein_cond(normalized_attacked, source_im, cond_normal, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "grill_skl"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_aclmd_SKL_cond(normalized_attacked, source_im, cond_normal, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "grill_cos"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_aclmd_cos_cond(normalized_attacked, source_im, cond_normal, model, device)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)

