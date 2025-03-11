import torch

import torch.nn as nn

#from nvae.utils import add_sn
#from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.jit import script
import random

#from nvae.utils import reparameterize


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''


######################################################################################################################################################

conda deactivate
conda deactivate
cd /home/luser/autoencoder_attacks/train_aautoencoders/
conda activate /home/luser/anaconda3/envs/inn
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type latent_cosine
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_l2
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_SKL
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_l2
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_SKL
python betaVAE_all_kinds_of_attacks.py --segment 53 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_cosine


python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type latent_cosine
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_l2
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_SKL
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_l2
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_SKL
python betaVAE_all_kinds_of_attacks.py --segment 54 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_cosine



python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type latent_cosine
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_l2
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_SKL
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_l2
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_SKL
python betaVAE_all_kinds_of_attacks.py --segment 55 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_cosine


python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type latent_cosine
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_l2
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type output_attack_SKL
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_l2
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type weighted_combi_k_eq_latent_SKL
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_l2
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_wasserstein
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_SKL
python betaVAE_all_kinds_of_attacks.py --segment 56 --which_gpu 1 --beta_value 5.0 --attck_type aclmd_cosine


'''


#device = "cuda:1" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--segment', type=int, default=3, help='Segment index')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--attck_type', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--beta_value', type=str, default=5, help='Index of the feature to attack (0-5)')


args = parser.parse_args()

segment = args.segment
which_gpu = args.which_gpu
attck_type = args.attck_type
beta_value = args.beta_value
device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

from vae import VAE_big, VAE_big_b

#model = VAE_big(device, image_channels=3).to(device)
model = VAE_big(device, image_channels=3).to(device)
train_data_size = 162079
epochs = 199


sensitivities = np.load("/home/luser/autoencoder_attacks/a_sensitivity_results/layer_sensitivities_array/all_layer_sensitivities.npy")

#numbers = np.load("/home/luser/autoencoder_attacks/a_sensitivity_results/layer_sensitivities_array/all_layer_sensitivities.npy")

sens_array = np.array(sensitivities)
layer_wise_weights = sens_array/np.sum(sens_array)
print("layer_wise_weights", layer_wise_weights)

if attck_type == "acmld_sens_l2_deco":
    sensitivities[:15]=0.0
    sens_array = np.array(sensitivities)
    layer_wise_weights = sens_array/np.sum(sens_array)
    print("layer_wise_weights", layer_wise_weights)
else:
    sens_array = np.array(sensitivities)
    layer_wise_weights = sens_array/np.sum(sens_array)
    print("layer_wise_weights", layer_wise_weights)



#model_type = "TCVAE"
model_type = "VAE"
#beta_value = 10.0

#below is for beta = 1.0
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_VAE_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

#below is for beta = 5 or 10
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_seeded_CNN_VAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints/celebA_seeded_CNN_VAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_'+model_type+''+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))



model.eval()

roll_no = 1

#segment = 20


all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men", "women", "young", "old", "youngmen", "oldmen", "youngwomen", "oldwomen", "oldblackmen", "oldblackwomen", "oldwhitemen", "oldwhitewomen", "youndblackmen", "youndblackwomen", "youngwhitemen", "youngwhitewomen" ]

populations_all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men : 84434", "women : 118165", "young : 156734", "old : 45865", "youngmen : 53448 ", "oldmen : 7003", "youngwomen : 103287", "oldwomen : 1116" ]



#source_im = torch.load("/home/luser/autoencoder_attacks/train_aautoencoders/fairness_trials/attack_saves/"+str(select_feature)+"_d/images.pt")[segment].unsqueeze(0).to(device) 

source_im = torch.load("/home/luser/autoencoder_attacks/test_sets/celebA_test_set.pt")[segment].unsqueeze(0).to(device) 

model.eval()


if attck_type == "weighted_combi_l2":
    noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0
    layer_weights_un = nn.Parameter(torch.rand(1, 29), requires_grad=True).to(device)
if attck_type == "weighted_combi_l2_tw":
    noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0
    layer_weights_un = nn.Parameter(torch.tensor([ 0.0, 0.0, 0.0, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23,1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23,1/23, 1/23, 1/23, 1/23, 1/23, 0.0, 0.0, 0.0]), requires_grad=True).to(device)
else:
    noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0

#noise_addition = 0.08 * (2 * noise_addition - 1)


#desired_norm_l_inf = 0.094  # Worked very well
#desired_norm_l_inf = 0.094  # Worked very well
desired_norm_l_inf = 0.05  # Worked very well 0.15 is goog



import torch.optim as optim
if(attck_type == "weighted_combi_l2"):
    optimizer = optim.Adam([noise_addition], lr=0.0001)
    noise_addition.requires_grad = True
else:
    optimizer = optim.Adam([noise_addition], lr=0.0001)
    noise_addition.requires_grad = True


adv_alpha = 0.5

criterion = nn.MSELoss()

#num_steps = 1000000
#num_steps = 600000    # this is the accurate number of steps when you want to plot optimization plots

num_steps = 60010 # this is early stopping to get more and more resulst


prev_loss = 0.0

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

# Function to compute symmetric KL divergence between two tensors
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

    #attack_flow = model.fc3(z1)
    #source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    '''for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out'''


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  encoder_lip_sum * criterion(z1, z2)

    return loss_to_maximize, adv_gen, source_recon




def get_weighted_combinations_sensitivities_l2(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    block_count = 0

    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * layer_wise_weights[block_count]
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out
        block_count+=1  

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

    block_count+=2  

    lat_diff = criterion(z1, z2)* (layer_wise_weights[block_count] + layer_wise_weights[block_count-1])
    block_count+=1  

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)

    fc3_diff = criterion(attack_flow, source_flow) * layer_wise_weights[block_count]
    block_count+=1

    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * layer_wise_weights[block_count]
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out
        block_count+=1

    loss_to_maximize =  (encoder_lip_sum + lat_diff + fc3_diff + decoder_lip_sum)  * criterion(attack_flow, source_flow)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_sensitivities_l2_corr(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    block_count = 0

    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * layer_wise_weights[block_count]
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out
        block_count+=1  

    '''print("after encoder block_count", block_count)

    print("after encoder layer_wise_weights[block_count-1]", layer_wise_weights[block_count-1])
    print("after encoder layer_wise_weights[block_count]", layer_wise_weights[block_count])'''


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

    block_count+=2  

    lat_diff = criterion(z1, z2)* (layer_wise_weights[block_count-1] + layer_wise_weights[block_count-2])

    '''print("after latent layer_wise_weights[block_count-2]", layer_wise_weights[block_count-2])
    print("after latent layer_wise_weights[block_count-1]", layer_wise_weights[block_count-1])
    print("after latent layer_wise_weights[block_count]", layer_wise_weights[block_count])'''

    #block_count+=1  

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)


    fc3_diff = criterion(attack_flow, source_flow) * layer_wise_weights[block_count]
    block_count+=1


    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * layer_wise_weights[block_count]
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out
        block_count+=1

    #loss_to_maximize =  (lat_diff + decoder_lip_sum)  * criterion(attack_flow, source_flow)
    loss_to_maximize =  (encoder_lip_sum + lat_diff + fc3_diff + decoder_lip_sum)  * criterion(attack_flow, source_flow)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_sensitivities_l2_deco(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    block_count = 0

    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * layer_wise_weights[block_count]
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out
        block_count+=1  

    '''print("after encoder block_count", block_count)

    print("after encoder layer_wise_weights[block_count-1]", layer_wise_weights[block_count-1])
    print("after encoder layer_wise_weights[block_count]", layer_wise_weights[block_count])'''


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

    block_count+=2  

    lat_diff = criterion(z1, z2)* (layer_wise_weights[block_count-1] + layer_wise_weights[block_count-2])

    '''print("after latent layer_wise_weights[block_count-2]", layer_wise_weights[block_count-2])
    print("after latent layer_wise_weights[block_count-1]", layer_wise_weights[block_count-1])
    print("after latent layer_wise_weights[block_count]", layer_wise_weights[block_count])'''

    #block_count+=1  

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)


    fc3_diff = criterion(attack_flow, source_flow) * layer_wise_weights[block_count]
    block_count+=1


    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * layer_wise_weights[block_count]
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out
        block_count+=1

    #loss_to_maximize =  (lat_diff + decoder_lip_sum)  * criterion(attack_flow, source_flow)
    loss_to_maximize =  (lat_diff + decoder_lip_sum)  * criterion(attack_flow, source_flow)

    return loss_to_maximize, adv_gen, source_recon




def get_weighted_combinations_l2_full(normalized_attacked, source_im):
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


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  (encoder_lip_sum + criterion(z1, z2) + decoder_lip_sum) * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_aclm_l2(normalized_attacked, source_im):
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


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  (criterion(z1, z2) + encoder_lip_sum) * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_l2_aclmr_l2(normalized_attacked, source_im):
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

    #attack_flow = model.fc3(z1)
    #source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  (criterion(z1, z2) + encoder_lip_sum) * criterion(normalized_attacked, adv_gen)

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


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    loss_to_maximize =  (criterion(z1, z2) + decoder_lip_sum) * criterion(normalized_attacked, adv_gen)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_l2_aclmd_wasserstein(normalized_attacked, source_im):
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


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    loss_to_maximize =  (wasserstein_distance(z1, z2) + decoder_lip_sum) * wasserstein_distance(normalized_attacked, adv_gen)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_l2_aclmd_SKL(normalized_attacked, source_im):
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


def get_weighted_combinations_l2_test3(normalized_attacked, source_im):
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


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  encoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_cosine_enco(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
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


def get_weighted_combinations_cosine_full(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
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
        layer_lip_const = cos(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = wasserstein_distance(z1, z2) 
    #lipschitzt_loss_decoder = wasserstein_distance(normalized_attacked, adv_gen)  

    loss_to_maximize =  (encoder_lip_sum + decoder_lip_sum + (cos(z1, z2)-1.0)**2) * (cos(normalized_attacked, adv_gen)-1.0)**2


    return loss_to_maximize



def get_true_weighted_combinations_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer, wt in zip(model.encoder, layer_weights_un[:14]):
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * wt
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

    for layer, wt in zip(model.decoder, layer_weights_un[14:]):
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * wt
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    lipschitzt_loss_encoder = criterion(z1, z2) * layer_weights_un[15]
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  (lipschitzt_loss_encoder + encoder_lip_sum  + decoder_lip_sum ) * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_k_eq_latent_l2(normalized_attacked, source_im):
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
 
    loss_to_maximize =  criterion(z1, z2)   * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def output_attack_l2(normalized_attacked, source_im):
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

    return loss_to_maximize, adv_gen, source_recon

def output_attack_wasserstein(normalized_attacked, source_im):
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

    return loss_to_maximize, adv_gen, source_recon


def output_attack_SKL(normalized_attacked, source_im):
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

    return loss_to_maximize, adv_gen, source_recon

def get_weighted_combinations_k_eq_latent_wasserstein(normalized_attacked, source_im):
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
 
    loss_to_maximize =  wasserstein_distance(z1, z2)   * wasserstein_distance(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon

def get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im):
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
 
    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2)   * get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_wasserstein(normalized_attacked, source_im):
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

    '''attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0'''

    '''for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = wasserstein_distance(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out'''


    #lipschitzt_loss_encoder = wasserstein_distance(z1, z2) 
    #lipschitzt_loss_decoder = wasserstein_distance(normalized_attacked, adv_gen)  

    loss_to_maximize =  encoder_lip_sum * wasserstein_distance(z1, z2)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_wasserstein_full(normalized_attacked, source_im):
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


    #lipschitzt_loss_encoder = wasserstein_distance(z1, z2) 
    #lipschitzt_loss_decoder = wasserstein_distance(normalized_attacked, adv_gen)  

    loss_to_maximize =  (encoder_lip_sum + decoder_lip_sum + wasserstein_distance(z1, z2)) * wasserstein_distance(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_SKL(normalized_attacked, source_im):
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

    '''attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = get_symmetric_KLDivergence(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out'''


    #lipschitzt_loss_encoder = get_symmetric_KLDivergence(z1, z2) 
    #lipschitzt_loss_decoder = get_symmetric_KLDivergence(normalized_attacked, adv_gen)  

    loss_to_maximize =  encoder_lip_sum * get_symmetric_KLDivergence(z1, z2)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_SKL_full(normalized_attacked, source_im):
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


    #lipschitzt_loss_encoder = get_symmetric_KLDivergence(z1, z2) 
    #lipschitzt_loss_decoder = get_symmetric_KLDivergence(normalized_attacked, adv_gen)  

    loss_to_maximize =  (encoder_lip_sum +  get_symmetric_KLDivergence(z1, z2) + decoder_lip_sum) * get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



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


def get_latent_space_l2_loss(normalized_attacked, source_im):
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
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

    loss_to_maximize =  criterion(z1, z2) 

    return loss_to_maximize

def get_latent_space_cosine_loss(normalized_attacked, source_im):
    
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

    return cos(z1, z2) 


def get_latent_space_wasserstein_loss(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
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

    loss_to_maximize =  wasserstein_distance(z1, z2) 

    return loss_to_maximize, adv_gen, source_recon


def get_latent_space_SKL_loss(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
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

    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2) 

    return loss_to_maximize, adv_gen, source_recon


def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, outputChange: {instability}, deviation: {deviation}")
    print()
    print("attack type", attck_type)    
    adv_div_list.append(deviation.item())
    adv_mse_list.append(adv_mse.item())

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
        plt.savefig("/home/luser/autoencoder_attacks/robustness_eval_saves/optimization_time_plots/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(segment)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "/home/luser/autoencoder_attacks/train_aautoencoders/relevancy_test/layerwise_maximum_damage_attack/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(segment)+".pt")
    np.save("/home/luser/autoencoder_attacks/robustness_eval_saves/adversarial_div_convergence/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(segment)+".npy", adv_div_list)
    np.save("/home/luser/autoencoder_attacks/robustness_eval_saves/adversarial_mse_convergence/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(segment)+".npy", adv_mse_list)


def run_time_plots_and_saves_weighted(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen, layer_weights_un):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, outputChange: {instability}, deviation: {deviation}")
    print()
    #print("layer_weights_un", layer_weights_un)
    #print()
    adv_div_list.append(deviation.item())
    adv_mse_list.append(adv_mse.item())

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
        plt.savefig("/home/luser/autoencoder_attacks/robustness_eval_saves/optimization_time_plots/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_segment_"+str(segment)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "/home/luser/autoencoder_attacks/train_aautoencoders/relevancy_test/layerwise_maximum_damage_attack/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_segment_"+str(segment)+".pt")
    np.save("/home/luser/autoencoder_attacks/robustness_eval_saves/adversarial_div_convergence/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_segment_"+str(segment)+".npy", adv_div_list)
    np.save("/home/luser/autoencoder_attacks/robustness_eval_saves/adversarial_mse_convergence/VAE_beta_b"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(segment)+".npy", adv_mse_list)



if(attck_type == "latent_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_cosine"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize = get_latent_space_cosine_loss(normalized_attacked, source_im)
        loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im)-1.0)**2 

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_wasserstein"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = get_latent_space_wasserstein_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_SKL"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = get_latent_space_SKL_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            if step % 6000 == 0:
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_l2_enco"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_enco(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_l2_full"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_full(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclm_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclm_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "aclmr_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmr_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "acmld_sens_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_sensitivities_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "acmld_sens_l2_corr"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_sensitivities_l2_corr(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "acmld_sens_l2_deco"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_sensitivities_l2_deco(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_wasserstein"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_wasserstein(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_SKL"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_SKL(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_cosine"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_cos(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "test3"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_test3(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "weighted_combi_cosine_enco"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize = get_weighted_combinations_cosine_enco(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_cosine_full"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize = get_weighted_combinations_cosine_full(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "weighted_combi_l2_tw"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "weighted_combinations_wasserstein"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_wasserstein(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "weighted_combinations_wasserstein_full"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_wasserstein_full(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combinations_SKL"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_SKL(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



            

if(attck_type == "weighted_combinations_SKL_full"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_SKL_full(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)





if(attck_type == "weighted_combi_k_eq_latent_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "weighted_combi_k_eq_latent_wasserstein"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_wasserstein(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_k_eq_latent_SKL"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "output_attack_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = output_attack_l2(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "output_attack_wasserstein"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = output_attack_wasserstein(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "output_attack_SKL"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        loss_to_maximize, adv_gen, source_recon = output_attack_SKL(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 6000 == 0:
            with torch.no_grad():
                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_recon, p=2)
                adv_mse = torch.norm(adv_gen - normalized_attacked, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, adv_mse, normalized_attacked, scaled_noise, adv_gen)

