
'''

conda deactivate
conda deactivate
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/analysis_universal_box_plots.py --beta_value 5.0 --which_gpu 1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --l_inf_bound 0.07 --which_model VAE --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --box_plots_directory /home/luser/alma/box_plots


pending things : Output attack 3 metrics : Output attack for VQ-VAE is not feasible because of problems with discrete latent space and gradient calculation issues
SKL combibations

'''


import numpy as np
from matplotlib import pyplot as plt
import torch
from vae import VAE_big, VAE_big_b
import random
import pandas as pd
import torch.nn.functional as F
import seaborn as sns

from torchvision import datasets, transforms
import os




import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--beta_value', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--which_model', type=str, default=5, help='model to attack')
parser.add_argument('--model_location', type=str, default=0, help='model directory')
parser.add_argument('--l_inf_bound', type=float, default=0.07, help='Index of the GPU to use (0-N)')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--box_plots_directory', type=str, default=0, help='data directory')


args = parser.parse_args()

beta_value = args.beta_value
which_gpu = args.which_gpu
model_location = args.model_location
model_type = args.which_model
data_directory = args.data_directory
box_plots_directory = args.box_plots_directory
l_inf_bound = args.l_inf_bound

vae_beta_value = beta_value

device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

model = VAE_big(device, image_channels=3).to(device)


train_data_size = 162079
epochs = 199
model.load_state_dict(torch.load(''+model_location+'/celebA_CNN_'+model_type+''+str(vae_beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))


attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "output_attack_cosine", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "weighted_combi_k_eq_latent_cosine", "aclmd_l2f_cond", "aclmd_wasserstein_cond", "aclmd_SKL_cond", "aclmd_cosine_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]


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

with torch.no_grad():


    all_mse_lists = []
    all_l2_dist_lists = []

    for i in range(len(attack_types)):

        optimized_noise = torch.load("/home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_types[i])+"_norm_bound_"+str(l_inf_bound)+".pt").to(device) 


        attacked = (source_im + optimized_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())
        adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
        print("adv_gen.shape", adv_gen.shape)
        print("normalized_attacked.shape)", normalized_attacked.shape)

        reconstruction_loss = F.mse_loss(normalized_attacked, adv_gen, reduction='none')  # Shape: [50, 3, 64, 64]
        reconstruction_loss_per_image = reconstruction_loss.mean(dim=[1, 2, 3])  # Shape: [50]
        l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]


        numpy_array = reconstruction_loss_per_image.cpu().numpy()
        all_mse_lists.append(numpy_array)
        all_l2_dist_lists.append(l2_distance_per_image.cpu().numpy())

    data = pd.DataFrame({
        "Latent,\nl-2": all_mse_lists[0],
        "Latent, \n wasser\nstein": all_mse_lists[1],
        "Latent, \nSKL": all_mse_lists[2],
        "Latent, \ncosine": all_mse_lists[3],

        "Output,\nl-2": all_mse_lists[4],
        "Output, \n wasser\nstein": all_mse_lists[5],
        "Output, \nSKL": all_mse_lists[6],
        "Output, \ncosine": all_mse_lists[7],

        "LMA,\nl-2": all_mse_lists[8],
        "LMA, \n wasser\nstein": all_mse_lists[9],
        "LMA, \nSKL": all_mse_lists[10],
        "LMA, \ncosine": all_mse_lists[11],

        "ALMA, \nl-2": all_mse_lists[12],
        "ALMA, \wasser\nstein": all_mse_lists[13],
        "ALMA, \nSKL": all_mse_lists[14],
        "ALMA, \ncosine": all_mse_lists[15],
    })

    # Define colors for the boxplots
    colors = [
        'blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow',
        'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'indigo', 'gold'
    ]



    plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

    sns.boxplot(data=data, palette=colors)
    #plt.xlabel("Feature", fontsize=14)  # Adjust the fontsize as needed
    plt.ylabel(r"MSE", fontsize=20)  # Using LaTeX formatting for the ylabel

    plt.xticks(fontsize=12, rotation=45)  # Adjust the fontsize as needed

    print(data.min().min(), data.max().max())
    y_min, y_max = data.min().min(), data.max().max()
    plt.yticks(np.arange(y_min, y_max + 0.01, (y_max - y_min) / 5), fontsize=20)

    plt.tight_layout()  # Adjust layout to prevent cutoff of labels

    plt.savefig(""+box_plots_directory+"/"+model_type+"_norm_bound_"+str(l_inf_bound)+"_.png")
    plt.show()



    data = pd.DataFrame({
        "LA,\nl-2": all_l2_dist_lists[0],
        "LA, \nwasserst.": all_l2_dist_lists[1],
        "LA, \nSKL": all_l2_dist_lists[2],
        "LA, \ncosine": all_l2_dist_lists[3],

        "OA,\nl-2": all_l2_dist_lists[4],
        "OA, \nwasserst.": all_l2_dist_lists[5],
        "OA, \nSKL": all_l2_dist_lists[6],
        "OA, \ncosine": all_l2_dist_lists[7],

        "LMA,\nl-2": all_l2_dist_lists[8],
        "LMA, \nwasserst.": all_l2_dist_lists[9],
        "LMA, \nSKL": all_l2_dist_lists[10],
        "LMA, \ncosine": all_l2_dist_lists[11],

        "ALMA, \nl-2": all_l2_dist_lists[12],
        "ALMA, \nwasserst.": all_l2_dist_lists[13],
        "ALMA, \nSKL": all_l2_dist_lists[14],
        "ALMA, \ncosine": all_l2_dist_lists[15],
    })

    colors = [
        'blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow',
        'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'indigo', 'gold'
    ]

    #colors = ['blue', 'orange', 'green', 'red']


    plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

    sns.boxplot(data=data, palette=colors)
    #plt.xlabel("Feature", fontsize=14)  # Adjust the fontsize as needed
    plt.ylabel(r"L-2 distance", fontsize=24)  # Using LaTeX formatting for the ylabel
    print(data.min().min(), data.max().max())
    y_min, y_max = data.min().min(), data.max().max()
    plt.yticks(np.arange(y_min, y_max + 0.01, (y_max - y_min) / 5), fontsize=24)

    plt.tight_layout()  # Adjust layout to prevent cutoff of labels

    plt.savefig(""+box_plots_directory+"/"+model_type+"_l2_norm_bound_"+str(l_inf_bound)+"_.png")
    plt.show()
