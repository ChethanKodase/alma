


#%load_ext autoreload
#%autoreload 2

'''

cd alma
conda activate dt2
python diffae/autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.35 --attck_type latent_l2 --which_gpu 3



conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.31 --attck_type latent_l2 --which_gpu 3
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type latent_l2 --which_gpu 3
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.33 --attck_type latent_l2 --which_gpu 3
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type latent_l2 --which_gpu 3
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.28 --attck_type latent_l2 --which_gpu 3
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type latent_l2 --which_gpu 3
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.26 --attck_type latent_l2 --which_gpu 3



conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.31 --attck_type latent_wasserstein --which_gpu 5
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type latent_wasserstein --which_gpu 5
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.33 --attck_type latent_wasserstein --which_gpu 5
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type latent_wasserstein --which_gpu 5
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.28 --attck_type latent_wasserstein --which_gpu 5
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type latent_wasserstein --which_gpu 5


conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.31 --attck_type latent_SKL --which_gpu 6
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type latent_SKL --which_gpu 6
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.33 --attck_type latent_SKL --which_gpu 6
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type latent_SKL --which_gpu 6
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.28 --attck_type latent_SKL --which_gpu 6
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type latent_SKL --which_gpu 6







conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type latent_cosine --which_gpu 2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.28 --attck_type latent_cosine --which_gpu 2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type latent_cosine --which_gpu 2


####################################################################################################################################


conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.31 --attck_type combi_l2 --which_gpu 7
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type combi_l2 --which_gpu 7
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.33 --attck_type combi_l2 --which_gpu 7
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type combi_l2 --which_gpu 7
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.28 --attck_type combi_l2 --which_gpu 7
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type combi_l2 --which_gpu 7


conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type combi_wasserstein --which_gpu 1
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.28 --attck_type combi_wasserstein --which_gpu 1
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type combi_wasserstein --which_gpu 1
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.31 --attck_type combi_wasserstein --which_gpu 4
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type combi_wasserstein --which_gpu 3
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.33 --attck_type combi_wasserstein --which_gpu 6


conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.28 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.31 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.33 --attck_type combi_SKL --which_gpu 5


conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type combi_cos_cond_corr --which_gpu 2


conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.26 --attck_type combi_cos_cond_corr --which_gpu 3

conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.29 --attck_type combi_cos_cond_corr --which_gpu 4



conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.27 --attck_type combi_cos_cond_corr_cap --which_gpu 7

conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.32 --attck_type combi_cos_cond_dir --which_gpu 7

conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.33 --attck_type combi_cos_cond_dir_cap --which_gpu 7

####################################################################################################################################
Ignore



conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.21 --attck_type combi_cos --which_gpu 4



conda activate dt2
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.26 --attck_type combi_cos_cond_dir_ef --which_gpu 5

####################################################################################################################################


Imp

python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.22 --attck_type latent_cosine --which_gpu 5
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.25 --attck_type combi_cos_cond_corr --which_gpu 1
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.25 --attck_type combi_cos_cond_dir --which_gpu 3





python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.24 --attck_type combi_l2 --which_gpu 4
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.24 --attck_type combi_wasserstein --which_gpu 4
python autoencoding_attack_universal.py --source_segment 0 --desired_norm_l_inf 0.24 --attck_type combi_SKL --which_gpu 4


'''


from templates import *
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.nn import DataParallel
import torch.nn.functional as F

from torch.utils.data import DataLoader


import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--source_segment', type=int, default=0, help='Source segment')
parser.add_argument('--attck_type', type=str, default=5, help='Type of attack')
parser.add_argument('--desired_norm_l_inf', type=float, default=0.08, help='Type of attack')

args = parser.parse_args()

which_gpu = args.which_gpu
source_segment = args.source_segment
attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf


device = 'cuda:'+str(which_gpu)+''


conf = ffhq256_autoenc()

#conf = ffhq256_autoenc_latent()
print(conf.name)
model = LitModel(conf)
state = torch.load(f'../diffae/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);


total_params = sum(p.numel() for p in model.ema_model.parameters())
trainable_params = sum(p.numel() for p in model.ema_model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
#print("model.ema_model.encoder", model.ema_model.encoder)
#print("model.encode", model.encode)



data = ImageDataset('../diffae/imgs_align_uni_ad', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
print("{len(data)}", len(data))

#segment = 1
#desired_norm_l_inf = 0.7  # Worked very well 0.15 is goog


batch_size = 25
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

batch_list = []
for source_im in train_loader:
    batch = source_im['img'].to(device)  # Move batch to GPU
    #print("batch.shape", batch.shape)
    batch_list.append(batch)  # Store batch in a list
    #print("len(batch_list)", len(batch_list))
    #if(len(batch_list)==3):
        #break
big_tensor = torch.stack(batch_list)  # Shape: (num_batches, batch_size, C, H, W)
print("big_tensor.shape", big_tensor.shape)
del batch_list
del train_loader


source_im = data[source_segment]['img'][None].to(device)
plt.imshow(source_im[0].permute(1, 2, 0).cpu().numpy())
plt.show()



print("source_im.max()", source_im.max())
print("source_im.min()", source_im.min())

import matplotlib.pyplot as plt

'''cond = model.encode(source_im.to(device))
xT = model.encode_stochastic(source_im.to(device), cond, T=250)
source_recon = model.render(xT, cond, T=20)'''


#noise_addition = 2.0 * torch.rand(1, 3, 256, 256).to(device) - 1.0
#print("source_im.shape", source_im.shape)
#########
noise_addition = torch.rand(source_im.shape).to(device)
noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  

#noise_addition = noise_addition * (source_im.max() - 0.0) + 0.0  

#########

optimizer = optim.Adam([noise_addition], lr=0.0001)
noise_addition.requires_grad = True
source_im.requires_grad = True

adv_alpha = 0.5

criterion = nn.MSELoss()

num_steps = 1000000
from geomloss import SamplesLoss

########################################################
#attck_type = "combi_cos_cond"
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
    if(attck_type == "combi_cos_cond_corr"):
        #cond_nums_array[cond_nums_array>50.0]=50.0
        cond_nums_array_temp = cond_nums_array[:-1]
        cond_nums_compli = np.sum(cond_nums_array_temp) - cond_nums_array_temp
        cond_nums_normalized = cond_nums_compli/ np.sum(cond_nums_compli)
        cond_nums_normalized = np.concatenate([cond_nums_normalized, np.array([0.0])])
        #cond_nums_compli = (cond_nums_array.max() - cond_nums_array)+1e-4 
        #print("cond_nums_compli", cond_nums_compli)
        #cond_nums_normalized = cond_nums_compli/ np.sum(cond_nums_compli)
        #cond_nums_compli = np.sum(cond_nums_array) - cond_nums_array
        #cond_nums_normalized = cond_nums_compli/ np.sum(cond_nums_compli)
    if(attck_type == "combi_cos_cond_corr_cap"):
        temp = cond_nums_array[cond_nums_array<500.0]
        cond_nums_array[cond_nums_array>500.0] = temp.max()
        cond_nums_compli = np.sum(cond_nums_array) - cond_nums_array
        print("cond_nums_compli", cond_nums_compli)
        cond_nums_normalized = cond_nums_compli/ np.sum(cond_nums_compli)
        #cond_nums_compli = np.sum(cond_nums_array) - cond_nums_array
        #cond_nums_normalized = cond_nums_compli/ np.sum(cond_nums_compli)

    if(attck_type == "combi_cos_cond_dir_ef"):
        cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)
    if(attck_type == "combi_cos_cond_dir" or attck_type == "combi_wasserstein" or attck_type == "combi_SKL" or attck_type == "combi_l2" or attck_type == "latent_cosine" or attck_type == "latent_l2" or attck_type == "latent_wasserstein" or attck_type == "latent_SKL"):
        cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)
    if(attck_type == "combi_cos_cond_dir_cap"):
        cond_nums_array[cond_nums_array>50.0]=50.0
        cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)

    return cond_nums_normalized

#if(attck_type == "combi_cos_cond_dir" or attck_type == "combi_cos_cond" or attck_type == "combi_cos_cond_corr" or attck_type=="combi_cos_cond_dir_ef"):
with torch.no_grad():
    cond_nums_normalized = get_layer_pert_recon(model)
    print("cond_nums_normalized", cond_nums_normalized)
#################################################################################################################

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def poincare_cos(a, b):

    a = a.view(-1)
    b = b.view(-1)

    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)

    norm_a = torch.clamp(norm_a, max=1 - 1e-6)
    norm_b = torch.clamp(norm_b, max=1 - 1e-6)

    #assert norm_a < 1 - 1e-7 and norm_b < 1 - 1e-7, "Points must lie inside the Poincaré ball."

    a = a / (1 - norm_a**2 + 1e-6)
    b = b / (1 - norm_b**2 + 1e-6)
    
    return (a * b).sum()

import torch


def lorentz_inner(x, y):
    time_like = -(x[0] * y[0])    
    space_like = (x[1:] * y[1:]).sum()
    return time_like + space_like



'''def lorentz_inner(x, y):
    return -x[0] * y[0] + torch.dot(x[1:], y[1:])'''

def hyperbolic_cos_hyperboloid(a, b):

    a = a.view(-1)
    b = b.view(-1)
    
    a = a / torch.sqrt(lorentz_inner(a, a) + 1e-6)
    b = b / torch.sqrt(lorentz_inner(b, b) + 1e-6)

    similarity = -lorentz_inner(a, b)
    return similarity


def poincare_distance(x, y):

    norm_x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    

    norm_x_sq = torch.clamp(norm_x_sq, max=1 - 1e-6)
    norm_y_sq = torch.clamp(norm_y_sq, max=1 - 1e-6)
    

    euclidean_sq = torch.sum((x - y)**2, dim=-1, keepdim=True)
    

    arg = 1 + 2 * euclidean_sq / ((1 - norm_x_sq) * (1 - norm_y_sq))
    arg = torch.clamp(arg, min=1 + 1e-6) 
    

    distance = -1 * torch.arccosh(arg)
    
    return distance.squeeze(-1)  

def angular_distance(x, y):

    dot_product = torch.sum(x * y, dim=-1, keepdim=True)

    norm_x = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + 1e-6)
    norm_y = torch.sqrt(torch.sum(y**2, dim=-1, keepdim=True) + 1e-6)
    
    cosine_similarity = dot_product / (norm_x * norm_y)

    cosine_similarity = torch.clamp(cosine_similarity, min=-1.0, max=1.0)

    angular_dist = torch.arccos(cosine_similarity)
    
    return angular_dist.squeeze(-1)  # Remove singleton dimensions if any


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
def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, optimized_noise, adv_gen):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, deviation: {deviation}")
    print()
    print("attack type", attck_type)    
    adv_div_list.append(deviation.item())
 
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(((normalized_attacked[0]+1)/2).permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(optimized_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig("../diffae/attack_run_time_univ/attack_plot/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".png")

    #optimized_noise = scaled_noise
    print("optimized_noise.shape", optimized_noise.shape)
    torch.save(optimized_noise, "../diffae/attack_run_time_univ/attack_noise/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".pt")
    np.save("../diffae/attack_run_time_univ/adv_div_convergence/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", adv_div_list)

def get_latent_space_l2_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return F.mse_loss(embed, attacked_embed, reduction='sum')

def get_latent_space_cosine_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return cos(embed, attacked_embed)

def get_latent_space_l2_cosine_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return cos(embed, attacked_embed) + F.mse_loss(embed, attacked_embed, reduction='sum')


def get_latent_space_hyperbolic_loss1(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return hyperbolic_cos_hyperboloid(embed, attacked_embed)



def get_latent_space_hyperbolic_loss2(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return poincare_distance(embed, attacked_embed)

def get_latent_space_hyperbolic_loss3(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return poincare_cos(embed, attacked_embed)

def get_latent_space_stat_l2_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    #embed.requires_grad = True
    #attacked_embed.requires_grad = True

    xT_i = model.encode_stochastic(source_im.to(device), embed, T=2)
    xT_a = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=2)

    #return F.mse_loss(embed, attacked_embed, reduction='sum')
    xT_i.requires_grad = True
    xT_a.requires_grad = True


    recon = model.render(xT_i, embed, T=2)
    adv_gen = model.render(xT_a, attacked_embed, T=2)


    return F.mse_loss(recon, adv_gen, reduction='sum') * F.mse_loss(embed, attacked_embed, reduction='sum')


def get_latent_space_wasserstein_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return wasserstein_distance(embed, attacked_embed)

def get_latent_space_SKL_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return get_symmetric_KLDivergence(embed, attacked_embed)

def get_latent_space_mixed_sum_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return get_symmetric_KLDivergence(embed, attacked_embed) + F.mse_loss(embed, attacked_embed, reduction='sum') + get_symmetric_KLDivergence(embed, attacked_embed)

def get_latent_space_mixed_prod_loss(normalized_attacked, source_im):

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return get_symmetric_KLDivergence(embed, attacked_embed) * F.mse_loss(embed, attacked_embed, reduction='sum') * get_symmetric_KLDivergence(embed, attacked_embed)

def get_combined_l2_loss(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (cond_nums_normalized[block_count])
        block_count+=1

    #embed = model.encode(source_im.to(device))
    #attacked_embed = model.encode(normalized_attacked.to(device))

    #return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 
    return encoder_lip_sum * F.mse_loss(x, x_p, reduction='sum') 

def get_combined_cosine_loss_backup(normalized_attacked_e, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += (cos(x, x_p)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += (cos(x, x_p)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.out):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += (cos(x, x_p)-1)**2 

    embed = model.encode(source_im.to(device))
    normalized_attacked_e = model.encode(normalized_attacked_e.to(device))

    #return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 
    return encoder_lip_sum * (cos(embed, attacked_embed)-1)**2 


def get_combined_cosine_loss(normal_x, source_x):

    encoder_lip_sum = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



def get_combined_cosine_loss_cond(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1
        #print("block_count", block_count)

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



'''def get_combined_cosine_loss_lw(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += (cos(x, x_p)-1)**2 * layer_wise_weights[block_count] 
        block_count+=1  

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += (cos(x, x_p)-1)**2 * layer_wise_weights[block_count] 
        block_count+=1  

    for i, block in enumerate(model.ema_model.encoder.out):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += (cos(x, x_p)-1)**2 * layer_wise_weights[block_count] 
        block_count+=1  

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    #return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 
    return encoder_lip_sum * (cos(embed, attacked_embed)-1)**2 '''


def get_combined_wasserstein_loss(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    #embed = model.encode(source_im.to(device))
    #attacked_embed = model.encode(normalized_attacked.to(device))

    return encoder_lip_sum * wasserstein_distance(x, x_p) 


def get_combined_SKL_loss(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return encoder_lip_sum * get_symmetric_KLDivergence(embed, attacked_embed) 


if(attck_type == "latent_l2"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            #source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                print("1 normalized_attacked.shape", normalized_attacked.shape)
                normalized_attacked = normalized_attacked[0].unsqueeze(0)
                print("2 normalized_attacked.shape", normalized_attacked.shape)
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
                #source_im = normalized_attacked


if(attck_type == "latent_wasserstein"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            #source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_latent_space_wasserstein_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
                #source_im = normalized_attacked

if(attck_type == "latent_SKL"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            #source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_latent_space_SKL_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
                #source_im = normalized_attacked

if(attck_type == "latent_cosine"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            #source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im)-1.0)**2 

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
                #source_im = normalized_attacked

if(attck_type == "combi_cos"):
    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            #source_im = source_im['img'].to(device)
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            total_loss = -1 * get_combined_cosine_loss(normalized_attacked, source_im)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "combi_cos_cond"):
    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            #source_im = source_im['img'].to(device)
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "combi_cos_cond_corr"):
    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            #source_im = source_im['img'].to(device)
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "combi_cos_cond_corr_cap"):
    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            #source_im = source_im['img'].to(device)
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)




from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint

scaler = GradScaler()

import torch.profiler

# Define profiler
prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # Logs for TensorBoard
    record_shapes=True, 
    profile_memory=True,  # Enables memory tracking
    with_stack=True,
    with_flops=True
)


if(attck_type == "combi_cos_cond_dir_ef"):

    prof.start()  # Start profiling

    adv_div_list = []
    for step in range(1):
        for source_im in big_tensor:
            #source_im = source_im['img'].to(device)
            print("source_im.shape", source_im.shape)
            optimizer.zero_grad()

            with autocast(enabled=False):  # Disable AMP for attack calculations
                with torch.autograd.profiler.record_function("forward_pass"):
                    normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
                    normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
                    total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)

            with torch.autograd.profiler.record_function("backward_pass"):
                total_loss.backward()
                optimizer.step()
            print("optimization step")
        prof.step()  # Step the profiler after each iteration
    prof.stop()  # Stop profiling

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))  # Print summary

    print("step", step)
    if(step%50==0):
        with torch.no_grad():
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #print("instability", instability)
            scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)



'''if(attck_type == "combi_cos_cond_dir_ef"):

    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            source_im = source_im['img'].to(device)
            optimizer.zero_grad()

            with autocast(enabled=False):  # Disable AMP for attack calculations
                normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
                normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
                total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)'''




if(attck_type == "combi_cos_cond_dir"):
    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            #source_im = source_im['img'].to(device)
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "combi_cos_cond_dir_cap"):
    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            #source_im = source_im['img'].to(device)
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


'''if(attck_type == "combi_cos_lw"):
    adv_div_list = []
    for step in range(150):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_combined_cosine_loss_lw(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        attacked_embed = model.encode(normalized_attacked.to(device))
        xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
        adv_gen = model.render(xT_ad, attacked_embed, T=20)
        instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
        print("instability", instability)
        l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
        l2_distortion = torch.norm(scaled_noise, p=2)
        deviation = torch.norm(adv_gen - source_im, p=2)
        #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
        get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
        #source_im = normalized_attacked'''


if(attck_type == "combi_l2"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            #source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_combined_l2_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #print("instability", instability)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
                #source_im = normalized_attacked


if(attck_type == "combi_wasserstein"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            #source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

            loss_to_maximize = get_combined_wasserstein_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "combi_SKL"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            #print(f"Batch shape: {batch['img'].shape}")
            #source_im = batch['img'].to(device)
            #print("source_im.shape", source_im.shape)

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_combined_SKL_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
