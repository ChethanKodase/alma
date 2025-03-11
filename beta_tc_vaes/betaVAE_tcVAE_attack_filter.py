import torch

import torch.nn as nn

#from nvae.utils import add_sn
#from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.jit import script
import pandas as pd
from conditioning import get_condition_weights

#from nvae.utils import reparameterize

from afp import SimpleCNN

'''

######################################################################################################################################################

conda deactivate
conda deactivate
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/betaVAE_tcVAE_attack_filter.py  --which_gpu 0 --beta_value 5.0 --which_model VAE --desired_norm_l_inf 0.09 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --num_steps 300 --filter_location /home/luser/alma/beta_tc_vaes/filter_storage --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack

'''


import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
#parser.add_argument('--segment', type=int, default=3, help='Segment index')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--beta_value', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--which_model', type=str, default="VAE", help='VAE or TCVAE')
parser.add_argument('--desired_norm_l_inf', type=float, default=5, help='perturbation norm bounding')
parser.add_argument('--model_location', type=str, default=0, help='model directory')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--num_steps', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--filter_location', type=str, default=0, help='model directory')
parser.add_argument('--uni_noise_directory', type=str, default=0, help='data directory')


args = parser.parse_args()

#segment = args.segment
which_gpu = args.which_gpu
beta_value = args.beta_value
which_model = args.which_model
desired_norm_l_inf = args.desired_norm_l_inf
model_location = args.model_location
filter_location = args.filter_location
data_directory = args.data_directory
num_steps = args.num_steps
uni_noise_directory = args.uni_noise_directory

device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

from vae import VAE_big, VAE_big_b

model = VAE_big(device, image_channels=3).to(device)
train_data_size = 162079
epochs = 199
model_type = which_model
model.load_state_dict(torch.load(''+model_location+'/celebA_CNN_'+model_type+''+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
model.eval()


with torch.no_grad():
    cond_nums, cond_normal = get_condition_weights(model)


inter_model = SimpleCNN(image_channels=3).to(device)


import torch.optim as optim

optimizer = optim.Adam(inter_model.parameters(), lr=0.0001)


adv_alpha = 0.5

criterion = nn.MSELoss()

#num_steps = 300

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


def get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_flow = inter_model.all_encs[l_ct](attack_flow)   ###########inter_model
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)#*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    #attack_flow_mu1 = inter_model.all_encs[l_ct](attack_flow)  ###########inter_model
    attack_flow_mu1 = attack_flow


    mu1 = model.fc1(attack_flow_mu1)
    mu2 = model.fc1(source_flow)

    mu_loss = criterion(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    #attack_flow_logvar1 = inter_model.all_encs[l_ct](attack_flow) ###########inter_model
    attack_flow_logvar1 = attack_flow

    logvar1 = model.fc2(attack_flow_logvar1)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    logvar2 = model.fc2(source_flow)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    rep_loss = criterion(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1

    #print("z1", z1.shape)   
    attack_flow = inter_model.all_encs[l_ct](z1) ###########inter_model
    attack_flow = z1

    attack_flow = model.fc3(attack_flow)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        #attack_flow = inter_model.all_encs[l_ct](attack_flow) ###########inter_model
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1

        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_im, attack_flow)

    return loss_to_maximize, attack_flow, source_flow


def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, outputChange: {instability}, deviation: {deviation}")
    print()
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
        plt.savefig("/home/luser/alma/beta_tc_vaes/filter_run_time/"+model_type+"_beta_"+str(beta_value)+"_norm_bound_"+str(desired_norm_l_inf)+"step_"+str(step)+".png")


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
batch_size = 50
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "aclmd_l2a_cond", "aclmd_wasserstein_cond", "aclmd_SKL_cond", "aclmd_cosine_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

chosen_att_ind = 10

adv_div_list = []
for step in range(num_steps):
    for idx, (image, label) in enumerate(testLoader):
        source_im, label = image.to(device), label.to(device)

        optimized_noise = torch.load(""+uni_noise_directory+"/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attack_types[chosen_att_ind])+"_norm_bound_"+str(desired_norm_l_inf)+".pt").to(device) 

        attacked = (source_im + optimized_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())


        loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

        total_loss = loss_to_maximize

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    with torch.no_grad():
        instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
        l_inf_distortion = torch.norm(optimized_noise, p=float('inf'))
        l2_distortion = torch.norm(optimized_noise, p=2)
        deviation = torch.norm(adv_gen - source_recon, p=2)
        get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, optimized_noise, adv_gen)

    
    torch.save(inter_model.state_dict(), ""+filter_location+"/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attack_types[chosen_att_ind])+"_norm_bound_"+str(desired_norm_l_inf)+".torch")