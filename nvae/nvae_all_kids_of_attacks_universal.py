import torch
import torch.nn as nn
from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random

from torchvision import datasets, transforms
import os
import pandas as pd
import torch.optim as optim

'''

0, 1, 2, 3


conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=3
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_l2" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_wass" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_skl" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_cos" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_l2" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_wass" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_skl" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_cos" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint



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

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--nvae_checkpoint_path', type=str, default=0, help='nvae checkpoint directory')


args = parser.parse_args()

#feature_no = args.feature_no
#source_segment = args.source_segment
attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
data_directory = args.data_directory
nvae_checkpoint_path = args.nvae_checkpoint_path


#all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men", "women", "young", "old", "youngmen", "oldmen", "youngwomen", "oldwomen", "oldblackmen", "oldblackwomen", "oldwhitemen", "oldwhitewomen", "youndblackmen", "youndblackwomen", "youngwhitemen", "youngwhitewomen" ]

#all_features = ["youngmen", "oldmen", "youngwomen", "oldwomen" ]

#populations_all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men : 84434", "women : 118165", "young : 156734", "old : 45865", "youngmen : 53448 ", "oldmen : 7003", "youngwomen : 103287", "oldwomen : 1116" ]

#select_feature = all_features[feature_no]



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


batch_size = 15
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

trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, drop_last=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

########delete some stuff to clear memory
del celeba_data
del train_set
del test_set
del trainLoader


# Initialize a list to store batch tensors
batch_list = []

for idx, (source_im, _) in enumerate(testLoader):
    source_im, _ = source_im.cuda(), _
    batch_list.append(source_im)  # Store batch in a list

big_tensor = torch.stack(batch_list)  # Shape: (num_batches, batch_size, C, H, W)
noise_addition = 2.0 * torch.rand(1, 3, 64, 64).cuda() - 1.0

def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Output shape: {output.shape}")



optimizer = optim.Adam([noise_addition], lr=0.0001)
adv_alpha = 0.5
noise_addition.requires_grad = True
criterion = nn.MSELoss()
num_steps = 40000
prev_loss = 0.0


# Dictionary to store layerwise outputs
layerwise_outputs = {}

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

def get_symmetric_KLDivergence(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl

def get_symmetric_KLDivergence_agg(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    var1 = var1 + 1e-6
    var2 = var2 + 1e-6

    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl


def encoder_hook_fn(module, input, output):
    layerwise_outputs[module] = output

# Register hooks for encoder layers
encoder_hook_handles = []

for name, layer in model.enc_tower.named_modules():
    handle = layer.register_forward_hook(encoder_hook_fn)
    encoder_hook_handles.append(handle)



def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen):
    with torch.no_grad():
        print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, deviation: {deviation}, recon mse: {mase_dev}")
        print()
        print("attack type", attck_type)    
        adv_div_list.append(deviation.item())
        adv_mse_list.append(mase_dev.item())
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
        plt.savefig("/mdadm0/chethan_krishnamurth/alma/nvae/optimization_time_plots/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "/mdadm0/chethan_krishnamurth/alma/nvae/univ_attack_storage/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
    #np.save("/mdadm0/chethan_krishnamurth/NVAE/attack_run_time_univ/adv_div_convergence/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.npy", adv_div_list)
    #np.save("/mdadm0/chethan_krishnamurth/NVAE/attack_run_time_univ/adv_mse_convergence/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.npy", adv_mse_list)



if(attck_type == "alma_l2"):
    all_condition_nums = np.load('/mdadm0/chethan_krishnamurth/NVAE/a_cond_analysis/saved_cond_nums/nvae_cond_nums.npy')

    print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())


    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            #print("source_im.shape", source_im.shape)
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            #print("layerwise_outputs", layerwise_outputs)
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            #print("adv_layerwise_outputs", adv_layerwise_outputs)
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += criterion(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += criterion(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:
        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)





if(attck_type == "combi_wasserstein"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for idx, (source_im, _) in enumerate(testLoader):
            source_im, _ = source_im.cuda(), _
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                orig_output = orig_layerwise_outputs[layer]
                loss += -1 *wasserstein_distance(adv_output, orig_output)
            layerwise_outputs.clear()            
            for i in range(len(adv_latent_reps)):
                loss += -1 * wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:
        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)





if(attck_type == "alma_wass"):

    all_condition_nums = np.load('/mdadm0/chethan_krishnamurth/NVAE/a_cond_analysis/saved_cond_nums/nvae_cond_nums.npy')

    print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:
        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "combi_SKL"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for idx, (source_im, _) in enumerate(testLoader):
            source_im, _ = source_im.cuda(), _
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                orig_output = orig_layerwise_outputs[layer]
                loss += -1 * get_symmetric_KLDivergence_agg(adv_output, orig_output)
            layerwise_outputs.clear()            
            for i in range(len(adv_latent_reps)):
                loss += -1 * get_symmetric_KLDivergence_agg(adv_latent_reps[i], orig_latent_reps[i])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:
        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "alma_skl"):

    all_condition_nums = np.load('/mdadm0/chethan_krishnamurth/NVAE/a_cond_analysis/saved_cond_nums/nvae_cond_nums.npy')

    print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss +=  get_symmetric_KLDivergence_agg(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += get_symmetric_KLDivergence_agg(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 *  loss * lat_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:
        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)





if(attck_type == "combi_cos"):
    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for idx, (source_im, _) in enumerate(testLoader):
            source_im, _ = source_im.cuda(), _
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                orig_output = orig_layerwise_outputs[layer]
                loss += -1 * (cos(adv_output, orig_output)-1)**2
            for i in range(len(adv_latent_reps)):
                loss += -1 * (cos(adv_latent_reps[i], orig_latent_reps[i])-1)**2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:
        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "alma_cos"):

    all_condition_nums = np.load('/mdadm0/chethan_krishnamurth/NVAE/a_cond_analysis/saved_cond_nums/nvae_cond_nums.npy')

    print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            '''loss = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                orig_output = orig_layerwise_outputs[layer]
                loss += (cos(adv_output, orig_output)-1)**2
            for i in range(len(adv_latent_reps)):
                loss += (cos(adv_latent_reps[i], orig_latent_reps[i])-1)**2'''

            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss +=  (cos(adv_output, orig_output)-1)**2 * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss +=  (cos(adv_latent_reps[i], orig_latent_reps[i])-1)**2
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:
        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "la_l2"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(100):

        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))

            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm)

            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            source_logits, log_q, log_p, kl_all, kl_diag, orig_latent_reps = model(source_im)


            distortion = torch.norm(noise_addition, 2)

            l2_distortion = torch.norm(scaled_noise, 2)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))


            loss = 0
            for i in range(len(adv_latent_reps)):
                loss = loss + criterion(adv_latent_reps[i], orig_latent_reps[i])

            total_loss = (-1 * loss )

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #if step % 400 == 0:
        with torch.no_grad():
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "la_wass"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(100):

        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))

            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm)

            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            source_logits, log_q, log_p, kl_all, kl_diag, orig_latent_reps = model(source_im)


            distortion = torch.norm(noise_addition, 2)

            l2_distortion = torch.norm(scaled_noise, 2)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))


            loss = 0
            for i in range(len(adv_latent_reps)):
                loss = loss + wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])

            total_loss = (-1 * loss )

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #if step % 400 == 0:
        with torch.no_grad():
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "la_skl"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(100):

        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))

            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm)

            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            source_logits, log_q, log_p, kl_all, kl_diag, orig_latent_reps = model(source_im)


            distortion = torch.norm(noise_addition, 2)

            l2_distortion = torch.norm(scaled_noise, 2)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))


            loss = 0
            for i in range(len(adv_latent_reps)):
                loss = loss + get_symmetric_KLDivergence(adv_latent_reps[i], orig_latent_reps[i])

            total_loss = (-1 * loss )

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #if step % 400 == 0:
        with torch.no_grad():
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "la_cos"):
    adv_div_list = []
    adv_mse_list = []

    for step in range(100):

        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))

            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm)

            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            source_logits, log_q, log_p, kl_all, kl_diag, orig_latent_reps = model(source_im)


            distortion = torch.norm(noise_addition, 2)

            l2_distortion = torch.norm(scaled_noise, 2)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))


            loss = 0
            for i in range(len(adv_latent_reps)):
                loss = loss + (cos(adv_latent_reps[i], orig_latent_reps[i])-1)**2

            total_loss = (-1 * loss )

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #if step % 400 == 0:
        with torch.no_grad():
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)

