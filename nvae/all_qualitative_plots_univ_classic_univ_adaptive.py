import torch
import torch.nn as nn
from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from torchvision import datasets, transforms
import os
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True  # Makes operations deterministic
    torch.backends.cudnn.benchmark = False  # Disable for full determinism


seed_num = 6

set_seed(seed_num)  # Or any other fixed number
g = torch.Generator()
g.manual_seed(seed_num)
#device = "cuda:1" if torch.cuda.is_available() else "cpu"


'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=7
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd grill/
python nvae/all_qualitative_plots_univ_classic_univ_adaptive.py --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --uni_noise_path nvae/univ_attack_storage/


'''


import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')

parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--nvae_checkpoint_path', type=str, default=0, help='nvae checkpoint directory')
parser.add_argument('--uni_noise_path', type=str, default=0, help='nvae checkpoint directory')


args = parser.parse_args()

data_directory = args.data_directory
nvae_checkpoint_path = args.nvae_checkpoint_path
uni_noise_path = args.uni_noise_path

adaptive_noise = True

#which_gpu = args.which_gpu

#all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men", "women", "young", "old", "youngmen", "oldmen", "youngwomen", "oldwomen", "oldblackmen", "oldblackwomen", "oldwhitemen", "oldwhitewomen", "youndblackmen", "youndblackwomen", "youngwhitemen", "youngwhitewomen" ]

all_features = ["youngmen", "oldmen", "youngwomen", "oldwomen" ]

feature_no = 2
source_segment = 0
select_feature = all_features[feature_no]



# Replace the placeholder values with your actual checkpoint path and parameters
checkpoint_path = ''+nvae_checkpoint_path+'/checkpoint.pt'
save_path = '/path/to/save'
eval_mode = 'sample'  # Choose between 'sample', 'evaluate', 'evaluate_fid'
batch_size = 0

# Load the model
checkpoint = torch.load(checkpoint_path, map_location='cpu')
args = checkpoint['args']

if not hasattr(args, 'ada_groups'):
    args.ada_groups = False

if not hasattr(args, 'min_groups_per_scale'):
    args.min_groups_per_scale = 1

if not hasattr(args, 'num_mixture_dec'):
    args.num_mixture_dec = 10

arch_instance = utils.get_arch_cells(args.arch_instance)  # You may need to replace this with the actual function or import it
model = AutoEncoder(args, None, arch_instance)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.cuda()
model.eval()

xts = []
for i in range(100):
    xts.append(i*10000)
desired_norm_l_inf = 0.03  # Worked very well

#attck_types = ["combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos", "hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos"]

#attck_types = ["hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos", "combi_l2_cond", "combi_wasserstein_cond", "combi_SKL_cond", "combi_cos_cond"] old

if adaptive_noise:
    attck_types = ["la_skl_mcmc", "grill_l2_mcmc"]
else:
    attck_types = ["hlatent_SKL", "combi_l2_cond"]

#root = '/home/luser/autoencoder_attacks/train_aautoencoders/data_faces/img_align_celeba'
#img_list = os.listdir(root)
#print(len(img_list))
     
#df = pd.read_csv("/home/luser/autoencoder_attacks/train_aautoencoders/list_attr_celeba.csv")
#df = df[['image_id', 'Smiling']]

batch_size = 1
img_list = os.listdir(''+data_directory+'/smile/')
img_list.extend(os.listdir(''+data_directory+'/no_smile/'))

transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
celeba_data = datasets.ImageFolder(data_directory, transform=transform)
split_train_frac = 0.95
train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * split_train_frac), len(img_list) - int(len(img_list) * split_train_frac)])
train_data_size = len(train_set)
test_data_size = len(test_set)

print('train_data_size', train_data_size)
print('test_data_size', test_data_size)

trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=False, drop_last=True, worker_init_fn=lambda _: set_seed(42), generator=g)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, worker_init_fn=lambda _: set_seed(42), generator=g)

del trainLoader



#desired_norm_l_infs = [0.03, 0.035, 0.04, 0.05]
#desired_norm_l_infs = [0.035, 0.04, 0.05]

desired_norm_l_infs = [0.04]


############# plotting for paper ######################

row_one_ims = []
row_two_ims = []


all_mse_lists = []
all_l2_dist_lists = []

all_method_means = []
all_method_stds = []


for idx, (source_im, _) in enumerate(testLoader):
    source_im, _ = source_im.cuda(), _

    for i in range(len(attck_types)):
        
        mean_per_per_accum = []
        std_per_per_accum = []

        for desired_norm_l_inf in desired_norm_l_infs:


            if adaptive_noise:
                optimized_noise = torch.load(""+uni_noise_path+"/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
            else:
                #optimized_noise = torch.load(""+uni_noise_path+"/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.pt")
                optimized_noise = torch.load("../NVAE/attack_run_time_univ/attack_noise/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.pt")


            #print("before optimized_noise.max(), optimized_noise.min()", optimized_noise.max(), optimized_noise.min())
            #optimized_noise =optimized_noise.clamp(-0.032, 0.032)
            #optimized_noise =optimized_noise * 0.09

            #print("after optimized_noise.max(), optimized_noise.min()", optimized_noise.max(), optimized_noise.min())

            l2_distance_per_image_aggre = []
            '''for idx, (source_im, _) in enumerate(testLoader):
                source_im, _ = source_im.cuda(), _'''

            attacked = (source_im + optimized_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())
            #print("normalized_attacked.shape", normalized_attacked.shape)
            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(source_im)
            reconstructed_normal = model.decoder_output(logits)
            normal_recon = reconstructed_normal.sample()


            print("adv_gen.shape", adv_gen.shape)


            plt.imshow(normalized_attacked[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
            #plt.set_title('Attacked Image')
            plt.axis('off')
            plt.show()
            plt.savefig("all_universal_qualitative/normalized_attacked_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")
            plt.close()

            plt.imshow(source_im[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
            #plt.set_title('Noise')
            plt.axis('off')
            plt.show()
            plt.savefig("all_universal_qualitative/source_im"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")
            plt.close()

            plt.imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
            #plt.set_title('Attack reconstruction')
            plt.axis('off')
            plt.show()
            plt.savefig("all_universal_qualitative/adv_gen_"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")
            plt.close()


            plt.imshow(normal_recon[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
            #plt.set_title('Attack reconstruction')
            plt.axis('off')
            plt.show()
            plt.savefig("all_universal_qualitative/normal_recon_"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")
            plt.close()


    break