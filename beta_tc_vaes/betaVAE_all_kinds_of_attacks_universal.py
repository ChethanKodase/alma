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
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_wasserstein --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_SKL --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_cosine --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_wasserstein --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_SKL --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_cosine --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_wass --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_skl --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type lma_cos --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type alma_ls --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type alma_wass --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type alma_skl --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type alma_cos --which_model TCVAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300

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



args = parser.parse_args()

which_gpu = args.which_gpu
attck_type = args.attck_type
beta_value = args.beta_value
which_model = args.which_model
desired_norm_l_inf = args.desired_norm_l_inf
data_directory = args.data_directory
model_location = args.model_location
num_steps = args.num_steps

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


def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

def wasserstein_distance(tensor_a, tensor_b):

    tensor_a_flat = torch.flatten(tensor_a)
    tensor_b_flat = torch.flatten(tensor_b)
    tensor_a_sorted, _ = torch.sort(tensor_a_flat)
    tensor_b_sorted, _ = torch.sort(tensor_b_flat)    
    wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    
    return wasserstein_dist

def compute_mean_and_variance(tensor):
    flattened_tensor = torch.flatten(tensor)  # Flatten the tensor
    mean = torch.mean(flattened_tensor)  # Compute mean
    variance = torch.var(flattened_tensor, unbiased=False)  # Compute variance (unbiased=False for population variance)
    return mean, variance

def get_symmetric_KLDivergence(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl



def get_weighted_combinations_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    lipschitzt_loss_encoder = criterion(z1, z2) 
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  (lipschitzt_loss_encoder + encoder_lip_sum  +  lipschitzt_loss_decoder + decoder_lip_sum ) * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_enco(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  encoder_lip_sum * criterion(z1, z2)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_cosine_enco(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = (cos(attack_out, source_out)-1.0)**2 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    return encoder_lip_sum * (cos(z1, z2)-1)**2


'''def get_weighted_combinations_k_eq_latent_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  criterion(z1, z2)   * criterion(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon'''


'''def output_attack_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon'''

'''def output_attack_wasserstein(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = wasserstein_distance(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon'''


'''def output_attack_SKL(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon'''


'''def output_attack_cosine(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
     
    loss_to_maximize = (cos(normalized_attacked, adv_gen)-1.0)**2

    return loss_to_maximize, adv_gen, source_recon'''


'''def get_weighted_combinations_k_eq_latent_wasserstein(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  wasserstein_distance(z1, z2)   * wasserstein_distance(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon'''

'''def get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2)   * get_symmetric_KLDivergence(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon'''


'''def get_weighted_combinations_k_eq_latent_cos(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  ((cos(z1, z2)-1.0)**2)   * (cos(source_recon, adv_gen)-1.0)**2

    return loss_to_maximize, adv_gen, source_recon'''


def get_layer_prod_loss(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum *= layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum *= layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    lipschitzt_loss_encoder = criterion(z1, z2) 
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  lipschitzt_loss_encoder * encoder_lip_sum * decoder_lip_sum 

    return loss_to_maximize, adv_gen, source_recon




def get_l2_combinations_k_equals_latent_loss(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    lipschitzt_loss_encoder = criterion(z1, z2) 
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  lipschitzt_loss_encoder  *  lipschitzt_loss_decoder 

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_aclmd_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (criterion(z1, z2) + decoder_lip_sum) * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



'''def get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = criterion(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = criterion(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)

    return loss_to_maximize, attack_flow, source_flow'''




'''def get_weighted_combinations_aclmd_wasserstein_cond(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += wasserstein_distance(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = wasserstein_distance(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = wasserstein_distance(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = wasserstein_distance(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += wasserstein_distance(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * wasserstein_distance(source_flow, attack_flow)

    return loss_to_maximize, attack_flow, source_flow'''





'''def get_weighted_combinations_aclmd_SKL_cond(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += get_symmetric_KLDivergence(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = get_symmetric_KLDivergence(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = get_symmetric_KLDivergence(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = get_symmetric_KLDivergence(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += get_symmetric_KLDivergence(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * get_symmetric_KLDivergence(source_flow, attack_flow)

    return loss_to_maximize, attack_flow, source_flow'''




'''def get_weighted_combinations_aclmd_cos_cond(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += (cos(attack_out, source_out)-1.0)**2  *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = (cos(mu1, mu2)-1.0)**2  *cond_normal[l_ct]
    l_ct += 1

    rep_loss = (cos(std1 * esp1, std2 * esp2)-1.0)**2 *cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = (cos(attack_flow, source_flow)-1.0)**2  *cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += (cos(attack_out, source_out)-1.0)**2 *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * (cos(source_flow, attack_flow)-1.0)**2

    return loss_to_maximize, attack_flow, source_flow'''





def get_weighted_combinations_l2_aclmd_wasserstein(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = wasserstein_distance(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = wasserstein_distance(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (wasserstein_distance(z1, z2) + decoder_lip_sum) * wasserstein_distance(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_aclmd_SKL(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = get_symmetric_KLDivergence(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = get_symmetric_KLDivergence(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    loss_to_maximize =  (get_symmetric_KLDivergence(z1, z2) + decoder_lip_sum) * get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_aclmd_cos(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = (cos(attack_out, source_out)-1.0)**2 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = (cos(attack_out, source_out)-1.0)**2 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #loss_to_maximize =  (get_symmetric_KLDivergence(z1, z2) + decoder_lip_sum) * get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    loss_to_maximize =  ((cos(z1, z2)-1.0)**2 + decoder_lip_sum ) * (cos(normalized_attacked, adv_gen)-1.0)**2


    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



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
        plt.savefig("/home/luser/autoencoder_attacks/robustness_eval_saves_univ/optimization_time_plots/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "/home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".pt")
    np.save("/home/luser/autoencoder_attacks/robustness_eval_saves_univ/adversarial_div_convergence/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".npy", adv_div_list)
    plt.close()

    plt.plot(adv_div_list)
    plt.savefig('/home/luser/autoencoder_attacks/robustness_eval_saves_univ/run_time_div/deviation_attack_type_'+attck_type+'_desired_norm_l_inf_'+str(desired_norm_l_inf)+'_.png')
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



if(attck_type == "aclmd_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2(normalized_attacked, source_im)

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



if(attck_type == "alma_ls"):
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


if(attck_type == "alma_wass"):
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


if(attck_type == "alma_skl"):
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


if(attck_type == "alma_cos"):
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




if(attck_type == "aclmd_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_wasserstein(normalized_attacked, source_im)

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




if(attck_type == "aclmd_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_SKL(normalized_attacked, source_im)

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


if(attck_type == "aclmd_cosine"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_cos(normalized_attacked, source_im)

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
