


#%load_ext autoreload
#%autoreload 2

'''




cd alma
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




data = ImageDataset('imgs_align_uni_ad', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
print("{len(data)}", len(data))


import torch

attck_type = "combi_cos_cond_dir"
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
                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)

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
                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)

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
                    print("condition_number", condition_number)
                    b_cond_nums.append(condition_number.item())
                else:
                    condition_number = 1.0
                    b_cond_nums.append(condition_number)
            else:
                b_cond_nums.append(1.0)
        if isinstance(block, (nn.SiLU, nn.AdaptiveAvgPool2d, nn.Flatten)):
            b_cond_nums = [1.0]
        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    cond_nums_array = np.array(g_cond_nums)

    if(attck_type == "combi_cos_cond"):
        cond_nums_normalized = (np.sum(cond_nums_array) - cond_nums_array) / np.sum(cond_nums_array)
    if(attck_type == "combi_cos_cond_dir"):
        cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)

    return cond_nums_normalized, cond_nums_array

if(attck_type == "combi_cos_cond_dir" or attck_type == "combi_cos_cond"):
    cond_nums_normalized, g_cond_nums = get_layer_pert_recon(model)
    print("g_cond_nums", g_cond_nums)
    print("cond_nums_normalized", cond_nums_normalized)

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


