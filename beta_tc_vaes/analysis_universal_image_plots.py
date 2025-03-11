
'''

conda deactivate
conda deactivate
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/analysis_universal_image_plots.py --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --qualitative_plots_directory /home/luser/alma/universal_qualitative --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack


pending things : Output attack 3 metrics : Output attack for VQ-VAE is not feasible because of problems with discrete latent space and gradient calculation issues
SKL combibations

'''


import numpy as np
from matplotlib import pyplot as plt
import torch
from vae import VAE_big, VAE_big_b
import pandas as pd
import os
import random



import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--beta_value', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--which_model', type=str, default=5, help='model to attack')
parser.add_argument('--model_location', type=str, default=0, help='model directory')
parser.add_argument('--l_inf_bound', type=float, default=0.07, help='Index of the GPU to use (0-N)')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--qualitative_plots_directory', type=str, default=0, help='data directory')
parser.add_argument('--uni_noise_directory', type=str, default="noise location", help='model to attack')


args = parser.parse_args()

beta_value = args.beta_value
which_gpu = args.which_gpu
model_location = args.model_location
model_type = args.which_model
data_directory = args.data_directory
qualitative_plots_directory = args.qualitative_plots_directory
l_inf_bound = args.l_inf_bound
uni_noise_directory = args.uni_noise_directory




# 0, 2, 8, 32
segment = 32
which_gpu = 0

#l_inf_bound = 0.12

#l_inf_bound = 0.07
l_inf_bound = 0.09


vae_beta_value = 5.0

#model_type = "TCVAE"
model_type = "VAE"
device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training


#model = VAE_big_b(device, image_channels=3).to(device)

model = VAE_big(device, image_channels=3).to(device)


train_data_size = 162079
epochs = 199
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints/celebA_seeded_CNN_VAE'+str(vae_beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

model.load_state_dict(torch.load(''+model_location+'/celebA_CNN_'+model_type+''+str(vae_beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_'+model_type+''+str(4.0)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

#"/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_VAE5.0_big_trainSize162079_epochs199.torch"

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "weighted_combi_l2_enco", "weighted_combinations_wasserstein", "weighted_combinations_SKL", "weighted_combi_cosine_enco", "weighted_combi_l2_full", "test1", "aclm_l2", "aclmd_l2", "aclmr_l2"]

attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "aclmd_l2a_cond", "aclmd_wasserstein_cond", "aclmd_SKL_cond", "aclmd_cosine_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]



############# plotting for paper ######################

row_one_ims = []
row_two_ims = []
#source_im = torch.load("/home/luser/autoencoder_attacks/test_sets/celebA_test_set.pt")[segment].unsqueeze(0).to(device) 


from torchvision import datasets, transforms

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
batch_size = 200
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

for idx, (image, label) in enumerate(testLoader):
    source_im, label = image.to(device), label.to(device)
    break



just_recon, adv_recon_loss, adv_kl_losses = model(source_im)
row_one_ims.append(source_im)
row_two_ims.append(just_recon)

for i in range(len(attack_types)):

    optimized_noise = torch.load(""+uni_noise_directory+"/"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_types[i])+"_norm_bound_"+str(l_inf_bound)+".pt").to(device) 

    #torch.save(optimized_noise, "/home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".pt")


    attacked = (source_im + optimized_noise)
    normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)

    row_one_ims.append(normalized_attacked)
    row_two_ims.append(adv_gen)


all_row_ims = row_one_ims + row_two_ims
num_ims = len(all_row_ims)

column_labels = ['Original Image'] + attack_types
column_labels = [f"Label {i+1}" for i in range(num_ims // 2)]

#column_labels = ["Original \nImage", "LA \nL-2", "LA \nWasserstein", "LA \nSKL", "LA \nCosine", "OA \nL-2", "OA \nWasserstein", "OA \nSKL", "LMA \nL-2", "LMA \nWasserstein", "LMA \nSKL", "ALMA \nL-2", "ALMA \nWasserstein", "ALMA \nSKL", "ALMA \nCosine", "ALMA \nL-2snesi", "ALMA \nL-2sn_corr", "ACLMA \nL-2sn_deco"]
column_labels = ["Original \nImage", "LA \nL-2", "LA \nWasserstein", "LA \nSKL", "LA \nCosine", "OA \nL-2", "OA \nWasserstein", "OA \nSKL", "LMA \nL-2", "LMA \nWasserstein", "LMA \nSKL", "ALMA \nL-2", "ALMA \nWasserstein", "ALMA \nSKL", "ALMA \nCosine"] #, "ALMA \nL-2snesi", "ALMA \nL-2sn_corr", "ACLMA \nL-2sn_deco"]

with torch.no_grad():

    fig, axes = plt.subplots(2, num_ims//2, figsize=(38, 5), gridspec_kw={'wspace': 0.02, 'hspace': 0.02})  # 2 rows, 10 columns

    # Loop through axes and images
    #for ax, img in zip(axes.flat, all_row_ims):
    for idx, (ax, img) in enumerate(zip(axes.flat, all_row_ims)):
        ax.imshow(img[0].permute(1, 2, 0).cpu().numpy())
        ax.axis('off')  # Hide axes
        # Add text only for the second row images
        if idx < num_ims // 2:  # If the index corresponds to the second row
            col_index = idx % (num_ims // 2)  # Get the column index
            ax.set_title(column_labels[col_index], fontsize=20, pad=10)  # Add text below the image

    #plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Adjust horizontal and vertical space
    #plt.tight_layout()
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
    plt.show()
    plt.savefig(""+qualitative_plots_directory+"/paper"+model_type+"_beta"+str(vae_beta_value)+"_norm_bound_"+str(l_inf_bound)+"_segment_"+str(segment)+".png", bbox_inches='tight')
    plt.close()


