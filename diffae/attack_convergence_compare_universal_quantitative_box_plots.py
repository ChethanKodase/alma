
'''

cd ../diffae
conda activate dt2
python diffae/attack_convergence_compare_universal_quantitative_box_plots.py  --desired_norm_l_inf 0.31 --which_gpu 7 


pending things : Output attack 3 metrics : Output attack for VQ-VAE is not feasible because of problems with discrete latent space and gradient calculation issues
SKL combibations

'''


import numpy as np
from matplotlib import pyplot as plt
import torch
from templates import *




import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')


parser.add_argument('--desired_norm_l_inf', type=float, default=0.08, help='Type of attack')
parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')
parser.add_argument('--ffhq_images_directory', type=str, default=5, help='images directory')
parser.add_argument('--noise_directory', type=str, default=5, help='images directory')



args = parser.parse_args()

desired_norm_l_inf = args.desired_norm_l_inf
which_gpu = args.which_gpu
diffae_checkpoint = args.diffae_checkpoint
ffhq_images_directory = args.ffhq_images_directory
noise_directory = args.noise_directory


#which_gpu = 6
source_segment = 0


#l_inf_bound = 0.12

#desired_norm_l_inf = 0.33

#vae_beta_value = 5.0
device = 'cuda:'+str(which_gpu)+''

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

xts = []
for i in range(100):
    xts.append(i*10000)

attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos"]
attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos_cond_dir_cap"]


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

all_metric_types = ["adv_recons", "adv_divs", "adv_divs_wass", "adv_divs_abs", "adv_divs_wass", "ssim", "psnr"]

#all_metric_types = ["adv_recons", "adv_divs", "adv_divs_abs", "ssim", "psnr"]

metric_type = all_metric_types[1]

for metric_type in all_metric_types:

    ##########################################################################################################################################################################
    ar0 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[0])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
    ar1 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[1])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
    ar2 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[2])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
    ar3 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[3])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
    # combinations
    ar4 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[4])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
    ar5 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[5])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
    ar6 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[6])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
    ar7 = np.load("../diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[7])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)



    print("len(ar0)", len(ar0))
    print("len(ar1)", len(ar1))
    print("len(ar2)", len(ar2))
    print("len(ar3)", len(ar3))
    print("len(ar4)", len(ar4))
    print("len(ar5)", len(ar5))
    print("len(ar6)", len(ar6))
    print("len(ar7)", len(ar7))

    print("np.mean(ar0)", np.mean(ar0))
    print("np.mean(ar1)", np.mean(ar1))
    print("np.mean(ar2)", np.mean(ar2))
    print("np.mean(ar3)", np.mean(ar3))
    print("np.mean(ar4)", np.mean(ar4))
    print("np.mean(ar5)", np.mean(ar5))
    print("np.mean(ar6)", np.mean(ar6))
    print("np.mean(ar7)", np.mean(ar7))

    data = pd.DataFrame({
        "LA,\nl-2": ar0,
        "LA, \n wasserstein": ar1,
        "LA, \nSKL": ar2,
        "LA, \ncosine": ar3,
        "ALMA, \nl-2": ar4,
        "ALMA, \nwasserstein": ar5,
        "ALMA, \nSKL": ar6,
        "ALMA, \ncosine": ar7,
    })

    '''data = pd.DataFrame({
        "latent,l2": ar0,
        "latent, wasserstein": ar1,
        "latent, SKL": ar2,
        "latent, cosine": ar3
    })'''

    # Define colors for the boxplots
    colors = ['blue', 'orange', 'green', 'red', 'lime', 'teal', 'indigo', 'gold']


    #colors = ['blue', 'orange', 'green', 'red']


    plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

    sns.boxplot(data=data, palette=colors)
    #plt.xlabel("Feature", fontsize=14)  # Adjust the fontsize as needed
    plt.ylabel(r"L-2 distance", fontsize=24)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$||x_a - x_a'||_2$", fontsize=12)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$||x_a - x_a'||_2$", fontsize=12)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$\frac{1}{n} \sum (x_a - x_a')^2$", fontsize=14)  # Using LaTeX formatting
    #plt.ylabel(r"$\mathrm{MSE}(x_a, x_a')$", fontsize=14)  # Using LaTeX formatting

    # Increase font size of xticks and yticks
    plt.xticks(fontsize=24, rotation=45)  # Adjust the fontsize as needed
    #plt.yticks(np.arange(200, 400, 5), fontsize=8)  # Adjust the fontsize as needed

    print(data.min().min(), data.max().max())
    y_min, y_max = data.min().min(), data.max().max()
    plt.yticks(np.arange(y_min, y_max + 0.01, (y_max - y_min) / 5), fontsize=24)

    plt.tight_layout()  # Adjust layout to prevent cutoff of labels

    plt.savefig("../diffae/attack_qualitative_untargeted_univ_quantitative/box_plots/"+metric_type+"_diff_ae_box_plots_universal_compare_methods_norm_bound_"+str(desired_norm_l_inf)+"_.png")
    plt.show()


    #len(youngmen_instability), len(oldmen_instability), len(youngwomen_instability), len(oldwomen_instability)