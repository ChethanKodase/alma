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
import matplotlib.ticker as ticker

#device = "cuda:1" if torch.cuda.is_available() else "cpu"


'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=3
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/nvae_all_condition_analysis.py


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


# Replace the placeholder values with your actual checkpoint path and parameters
checkpoint_path = '../NVAE/pretrained_checkpoint/checkpoint.pt'
save_path = '/path/to/save'
eval_mode = 'sample'  # Choose between 'sample', 'evaluate', 'evaluate_fid'
batch_size = 0

data_directory = 'data_cel1'

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


##############################################################################################################################################################################
# Data loading
##############################################################################################################################################################################

## test the num of steps 

desired_norm_l_inf = 0.01  # Worked very well

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




##############################################################################################################################################################################
# Data loading ends
##############################################################################################################################################################################
#torch.save(images, '/home/luser/autoencoder_attacks/test_sets/celebA_test_set.pt')


noise_addition = 2.0 * torch.rand(1, 3, 64, 64).cuda() - 1.0

#noise_addition = 0.08 * (2 * noise_addition - 1)
torch.norm(noise_addition, 2)


#desired_norm_l2 = 6.0  # Change this to your desired constant value

def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Output shape: {output.shape}")



#desired_norm_l_inf = 0.04  # Worked very well
import torch.optim as optim
optimizer = optim.Adam([noise_addition], lr=0.0001)
adv_alpha = 0.5
noise_addition.requires_grad = True
criterion = nn.MSELoss()
num_steps = 40000
prev_loss = 0.0


# Dictionary to store layerwise outputs
layerwise_outputs = {}
layerwise_modules = {}  # New dictionary to store actual modules

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
    layerwise_modules[module] = module  # Store actual module
# Register hooks for encoder layers
encoder_hook_handles = []

count = 0
for name, layer in model.enc_tower.named_modules():
    handle = layer.register_forward_hook(encoder_hook_fn)
    encoder_hook_handles.append(handle)
    count+=1
print("count", count)


adv_div_list = []
adv_mse_list = []
for step in range(100):
    for idx, (source_im, _) in enumerate(testLoader):
        source_im, _ = source_im.cuda(), _

        normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
        normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

        layerwise_outputs.clear()
        layerwise_modules.clear()
        _, _, _, _, _, adv_latent_reps = model(normalized_attacked)

        adv_layerwise_outputs = layerwise_outputs.copy()
        print("len(adv_layerwise_outputs)", len(adv_layerwise_outputs))

        check_modules = layerwise_modules.copy()
        print("len(check_modules)", len(check_modules))


        eq = 0
        compli = 0
        total_intra_params = 0
        total_4d_params = 0
        total_real4d = 0
        total_pseud04d = 0
        total_2d_params = 0
        total_3d_params = 0
        total_1d_params = 0
        all_condition_nums = []
        conv_layer_conds = []
        for module in check_modules.values():  # Iterate over actual modules
            params = list(module.parameters())  # Extract learnable parameters
            condition_number_sum = 0
            intra_param_ind = 0
            if params:  # If module has parameters (e.g., Conv, Linear, etc.)
                eq+=1
                #print("len(params)", len(params))
                for param in params:
                    total_intra_params+=1
                    if(len(param.shape)==1):
                        condition_number = 1.0
                        total_1d_params+=1
                        intra_param_ind+=1
                    if(len(param.shape)==2):
                        if(param.shape[-1]==1):
                            condition_number = 1.0
                            intra_param_ind+=1
                        else:
                            W_matrix = param.view(param.shape[0], -1)  # Flatten kernels into a 2D matrix
                            #print("W_matrix.shape", W_matrix.shape)
                            U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                            condition_number = (S.max() / S.min()).item()
                            #print("condition_number", condition_number)
                            
                            '''if condition_number == float("inf"): 
                                condition_number = 10.25
                            else:
                                condition_number = condition_number.item()'''

                            intra_param_ind+=1                              
                        total_2d_params+=1
                    if(len(param.shape)==3):
                        condition_number = 1.0
                        total_3d_params+=1
                        intra_param_ind+=1
                    if(len(param.shape)==4):
                        total_4d_params+=1
                        if(param.shape[1]==1 and param.shape[2]==1 and param.shape[3]==1):
                            total_pseud04d +=1
                            condition_number = 1.5
                            intra_param_ind+=1
                        else:
                            total_real4d+=1
                            W_matrix = param.view(param.shape[0], -1)  # Flatten kernels into a 2D matrix
                            U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                            condition_number = (S.max() / S.min())
                            condition_number = condition_number.item()
                            intra_param_ind+=1

                            conv_layer_conds.append(condition_number)

                    condition_number_sum += condition_number
                condition_number_avg = condition_number_sum/len(params)
            else:  # If module has no parameters (e.g., Identity, ReLU, etc.)
                if isinstance(module, (nn.ReLU)):
                    compli+=1
                    condition_number_avg = 10.0 # for all positive input relu can have condition number 1. with 1 negative input condition number can be 0
                if isinstance(module, (nn.Sigmoid)):
                    compli+=1
                    condition_number_avg = 5.0
                if isinstance(module, (nn.Tanh)):
                    compli+=1
                    condition_number_avg = 2.0
                if isinstance(module, (nn.Softmax)):
                    compli+=1
                    condition_number_avg = 2.0
                else:
                    compli+=1
                    condition_number_avg = 1.0

            all_condition_nums.append(condition_number_avg)
        all_condition_nums = np.array(all_condition_nums)
        non_inf_conds = all_condition_nums[all_condition_nums!=float("inf")]

        all_condition_nums[all_condition_nums==float("inf")] = non_inf_conds.max()
        print("non_inf_conds.shape", non_inf_conds.shape)
        print("non inf max and mins: ", non_inf_conds.max())
        print("non inf max and mins: ", non_inf_conds.min())
        print("len(all_condition_nums)", len(all_condition_nums))
        print("max(all_condition_nums)", (all_condition_nums.max()))
        print("max(all_condition_nums)", (all_condition_nums.min()))
        np.save('nvae/saved_cond_nums/nvae_cond_nums.npy', all_condition_nums)
        print()
        
        print("max(conv_layer_conds), min(conv_layer_conds)", max(conv_layer_conds), min(conv_layer_conds))
        
        
        filtered_g_cond_nums = [value for value in all_condition_nums if value != 1.0]
        plt.figure(figsize=(6, 8))  # Set figure size
        plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
        
        plt.ylabel("Block index", fontsize=24)
        plt.xlabel("$\kappa$", fontsize=24)

        # Custom scientific notation formatter for y-axis
        def sci_notation_formatter(y, _):
            if y == 0:
                return "0"  # Display zero as "0" instead of "0e0"
            return f"{int(y):.0e}".replace("+", "").replace("e0", "e")

        formatter = ticker.FuncFormatter(sci_notation_formatter)
        plt.gca().xaxis.set_major_formatter(formatter)
        
        plt.yticks(fontsize=24)
        #plt.xticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=14)
        plt.xticks(fontsize=24)

        plt.tight_layout()  # Adjust layout to prevent cutoff of labels
        plt.savefig("conditioning_analysis/nvae_conditioning_chart_.png")
        plt.show()

        layerwise_outputs.clear()

        break
    break
    