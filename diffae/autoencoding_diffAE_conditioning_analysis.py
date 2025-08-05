


#%load_ext autoreload
#%autoreload 2

'''




cd grill
conda activate dt2
python diffae/autoencoding_diffAE_conditioning_analysis.py --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints


####################################################################################################################################



'''


from templates import *
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.nn import DataParallel
import torch.nn.functional as F

from torch.utils.data import DataLoader

import matplotlib.ticker as ticker

import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')

args = parser.parse_args()



which_gpu = args.which_gpu
diffae_checkpoint = args.diffae_checkpoint


device = 'cuda:'+str(which_gpu)+''


conf = ffhq256_autoenc()

#conf = ffhq256_autoenc_latent()
print(conf.name)
model = LitModel(conf)
state = torch.load(f"{diffae_checkpoint}/{conf.name}/last.ckpt", map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);



import torch

act_cond_num = []
min_sing_vals = []
max_sing_vals = []

def get_layer_pert_recon(model):
    g_cond_nums = []
    for i, block in enumerate(model.ema_model.encoder.input_blocks):  

        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    act_cond_num.append(condition_number.item())
                    min_sing_vals.append(S.min().item())
                    max_sing_vals.append(S.max().item())

                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)
                    act_cond_num.append(1.0)
                    min_sing_vals.append(1e10)
                    max_sing_vals.append(1e-10)



        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    act_cond_num.append(condition_number.item())
                    min_sing_vals.append(S.min().item())
                    max_sing_vals.append(S.max().item())
                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)
                    act_cond_num.append(1.0)
                    min_sing_vals.append(1e10)
                    max_sing_vals.append(1e-10)

        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    for i, block in enumerate(model.ema_model.encoder.out):
        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    act_cond_num.append(condition_number.item())
                    min_sing_vals.append(S.min().item())
                    max_sing_vals.append(S.max().item())
                    b_cond_nums.append(condition_number.item())
                else:
                    condition_number = 1.0
                    b_cond_nums.append(condition_number)
                    act_cond_num.append(1.0)
                    min_sing_vals.append(1e10)
                    max_sing_vals.append(1e-10)

            else:
                b_cond_nums.append(1.0)
                act_cond_num.append(1.0)
                min_sing_vals.append(1e10)
                max_sing_vals.append(1e-10)



        if isinstance(block, (nn.SiLU, nn.AdaptiveAvgPool2d, nn.Flatten)):
            b_cond_nums = [1.0]
        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    cond_nums_array = np.array(g_cond_nums)

    actual_cond_nums_array = np.array(act_cond_num)
    min_sing_vals_array = np.array(min_sing_vals)
    max_sing_vals_array = np.array(max_sing_vals)


    cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)

    return cond_nums_normalized, cond_nums_array, actual_cond_nums_array, min_sing_vals_array, max_sing_vals_array

cond_nums_normalized, g_cond_nums, actual_cond_nums_array, min_sing_vals_array, max_sing_vals_array = get_layer_pert_recon(model)


filtered_g_cond_nums = [value for value in g_cond_nums if value != 1.0]



# Create a bar chart
plt.figure(figsize=(6, 9))  # Set figure size
plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)

# Labels and title
plt.ylabel("Block index", fontsize=24)
plt.xlabel("$\kappa$", fontsize=24)
#plt.title("Bar Chart of g_cond_nums")


def sci_notation_formatter(y, _):
    if y == 0:
        return "0"  # Display zero as "0" instead of "0e0"
    return f"{int(y):.0e}".replace("+", "").replace("e0", "e")

formatter = ticker.FuncFormatter(sci_notation_formatter)
plt.gca().xaxis.set_major_formatter(formatter)


plt.xticks(fontsize=24)
plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent cutoff of labels

plt.savefig("conditioning_analysis/diff_ae_conditioning_chart_.png")

plt.show()


####################################### actual condition numbers

filtered_g_cond_nums = [value for value in actual_cond_nums_array if value != 1.0]
plt.figure(figsize=(4, 6))  # Set figure size
plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
plt.ylabel("Layer index", fontsize=28)
plt.xlabel("$\kappa$", fontsize=28)
def sci_notation_formatter(y, _):
    if y == 0:
        return "0"  # Display zero as "0" instead of "0e0"
    return f"{int(y):.0e}".replace("+", "").replace("e0", "e")
formatter = ticker.FuncFormatter(sci_notation_formatter)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(fontsize=28, rotation=45)
#plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)
step = 4
yticks = list(range(1, len(filtered_g_cond_nums), step))

plt.yticks(yticks, yticks, fontsize=28)

plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("conditioning_analysis/diff_ae_conditioning_chart_actual_k.png")
plt.show()
plt.close()

####################################### minimum condition numbers

filtered_g_cond_nums = [value for value in min_sing_vals_array if value != 1e10]
print("min singular values ", filtered_g_cond_nums)
plt.figure(figsize=(4, 6))  # Set figure size
plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
plt.ylabel("Layer index", fontsize=28)
plt.xlabel("$\sigma_{min}$", fontsize=28)
'''def sci_notation_formatter(y, _):
    if y == 0:
        return "0"  # Display zero as "0" instead of "0e0"
    return f"{int(y):.0e}".replace("+", "").replace("e0", "e")'''
#formatter = ticker.FuncFormatter(sci_notation_formatter)
#plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(fontsize=28, rotation=45)
#plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)
step = 4
yticks = list(range(1, len(filtered_g_cond_nums), step))
plt.yticks(yticks, yticks, fontsize=28)
plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("conditioning_analysis/diff_ae_min_sing_vals.png")
plt.show()

print()

filtered_g_cond_nums = [value for value in max_sing_vals_array if value != 1e-10]
print("max singular values ", filtered_g_cond_nums)
plt.figure(figsize=(4, 6))  # Set figure size
plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
plt.ylabel("Layer index", fontsize=28)
plt.xlabel("$\sigma_{max}$", fontsize=28)
'''def sci_notation_formatter(y, _):
    if y == 0:
        return "0"  # Display zero as "0" instead of "0e0"
    return f"{int(y):.0e}".replace("+", "").replace("e0", "e")'''
#formatter = ticker.FuncFormatter(sci_notation_formatter)
#plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(fontsize=28, rotation=45)
#plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)
step = 4
yticks = list(range(1, len(filtered_g_cond_nums), step))
plt.yticks(yticks, yticks, fontsize=28)
plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("conditioning_analysis/diff_ae_max_sing_vals.png")
plt.show()