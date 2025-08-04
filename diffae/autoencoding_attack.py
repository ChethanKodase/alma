


#%load_ext autoreload
#%autoreload 2

'''

cd /diffae
conda activate /miniconda3/envs/dt2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type latent_l2 --which_gpu 2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type latent_SKL --which_gpu 2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type latent_wasserstein --which_gpu 2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type latent_cosine --which_gpu 2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type combi_l2 --which_gpu 2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type combi_wasserstein --which_gpu 2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type combi_cos --which_gpu 2

python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type latent_l2 --which_gpu 2
python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type latent_SKL --which_gpu 2
python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type latent_wasserstein --which_gpu 2
python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type latent_cosine --which_gpu 2
python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type combi_l2 --which_gpu 2
python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type combi_wasserstein --which_gpu 2
python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack.py --source_segment 11 --desired_norm_l_inf 0.08 --attck_type combi_cos --which_gpu 2


python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type latent_l2 --which_gpu 2
python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type latent_SKL --which_gpu 2
python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type latent_wasserstein --which_gpu 2
python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type latent_cosine --which_gpu 2
python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type combi_l2 --which_gpu 2
python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type combi_wasserstein --which_gpu 2
python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type combi_SKL --which_gpu 2
python autoencoding_attack.py --source_segment 12 --desired_norm_l_inf 0.08 --attck_type combi_cos --which_gpu 2

####################################################################################################################################

python autoencoding_attack.py --source_segment 10 --desired_norm_l_inf 0.08 --attck_type combi_cos_lw --which_gpu 2


'''


from templates import *
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.nn import DataParallel
import torch.nn.functional as F


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
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);

#print("model.ema_model.encoder", model.ema_model.encoder)
#print("model.encode", model.encode)
sensitivities = np.load("/diffae/a_sensitivity_test/sensitivity_list/sensitivity.npy")
sens_array = np.array(sensitivities)

layer_wise_weights = sens_array/np.sum(sens_array)

print("layer_wise_weights", layer_wise_weights)


data = ImageDataset('imgs_align_uni_ad', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
print("{len(data)}", len(data))

#segment = 1
#desired_norm_l_inf = 0.7  # Worked very well 0.15 is goog

#batch = data[source_segment]['img'][None]

#print("batch.shape", batch.shape)

source_im = data[source_segment]['img'][None].to(device)
plt.imshow(source_im[0].permute(1, 2, 0).cpu().numpy())
plt.show()
plt.savefig('/diffae/attack_run_time/attack_plot/see.png')

torch.save(source_im, "/diffae/attack_run_time/attacked_image/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".pt")


print("source_im.max()", source_im.max())
print("source_im.min()", source_im.min())

import matplotlib.pyplot as plt

cond = model.encode(source_im.to(device))
xT = model.encode_stochastic(source_im.to(device), cond, T=250)
source_recon = model.render(xT, cond, T=20)


#noise_addition = 2.0 * torch.rand(1, 3, 256, 256).to(device) - 1.0

#########
noise_addition = torch.rand(source_im.shape).to(device)
#noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  

noise_addition = noise_addition * (source_im.max() - 0.0) + 0.0  

#########

optimizer = optim.Adam([noise_addition], lr=0.0001)
noise_addition.requires_grad = True
source_im.requires_grad = True

adv_alpha = 0.5

criterion = nn.MSELoss()

num_steps = 200000
from geomloss import SamplesLoss

# Define Sinkhorn Loss
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)  # p=2 for squared distance, blur is the regularization parameter

'''def wasserstein_distance(tensor_a, tensor_b):

    # Flatten the tensors to compare distributions
    tensor_a_flat = tensor_a.view(tensor_a.size(0), -1)
    tensor_b_flat = tensor_b.view(tensor_b.size(0), -1)

    # Compute Sinkhorn distance
    return sinkhorn_loss(tensor_a_flat, tensor_b_flat)'''

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

def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, deviation: {deviation}")
    print()
    print("attack type", attck_type)    
    adv_div_list.append(deviation.item())
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(((normalized_attacked[0]+1)/2).permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(scaled_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig("/diffae/attack_run_time/attack_plot/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "/diffae/attack_run_time/attack_noise/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".pt")
    np.save("/diffae/attack_run_time/adv_div_convergence/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", adv_div_list)

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
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') 

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    #return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 
    return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 


def get_combined_l2_loss_full(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') 

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') 

    for i, block in enumerate(model.ema_model.encoder.out):
        #print('i', i)
        x = block(x)
        x_p = block(x_p)
        #encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (i**(2) / 20**2 )
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') 

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    #return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 
    return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 


def get_combined_cosine_loss(normalized_attacked, source_im):

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
    attacked_embed = model.encode(normalized_attacked.to(device))

    #return encoder_lip_sum * F.mse_loss(embed, attacked_embed, reduction='sum') 
    return encoder_lip_sum * (cos(embed, attacked_embed)-1)**2



def get_combined_cosine_loss_lw(normalized_attacked, source_im):

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
    return encoder_lip_sum * (cos(embed, attacked_embed)-1)**2


def get_combined_wasserstein_loss(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p)

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p)

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p)

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return encoder_lip_sum * wasserstein_distance(embed, attacked_embed) 


def get_combined_SKL_loss(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p)

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p)

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p)

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return encoder_lip_sum * get_symmetric_KLDivergence(embed, attacked_embed) 

if(attck_type == "latent_l2"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
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
            #source_im = normalized_attacked

if(attck_type == "latent_cosine"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im)-1.0)**2 

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():
            if step % 10000 == 0:
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
                #source_im = normalized_attacked

if(attck_type == "latent_l2_cosine"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im)-1.0)**2 + get_latent_space_l2_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
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
            #source_im = normalized_attacked

if(attck_type == "latent_hyperbolic1"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        loss_to_maximize = ( get_latent_space_hyperbolic_loss1(normalized_attacked, source_im)-1.0 )**2

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
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
            #source_im = normalized_attacked

if(attck_type == "latent_hyperbolic2"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        loss_to_maximize = ( get_latent_space_hyperbolic_loss2(normalized_attacked, source_im)-1.0 )**2

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
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
            #source_im = normalized_attacked

if(attck_type == "latent_hyperbolic3"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        loss_to_maximize = ( get_latent_space_hyperbolic_loss3(normalized_attacked, source_im)-1.0 )**2

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
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
            #source_im = normalized_attacked

if(attck_type == "latent_l2_expst1"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = 2.0 * ((attacked-attacked.min())/(attacked.max()-attacked.min())) - 1.0
        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)
        #normalized_attacked = attacked
        #loss_to_maximize = get_latent_space_l2_loss_exp(normalized_attacked, source_im)
        loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im)-1.0)**2

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
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
            #source_im = normalized_attacked


if(attck_type == "latent_l2_stat"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = 2.0 * ((attacked-attacked.min())/(attacked.max()-attacked.min())) - 1.0

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_latent_space_stat_l2_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 2 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = 2.0 * ((attacked-attacked.min())/(attacked.max()-attacked.min())) - 1.0

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_latent_space_wasserstein_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_SKL"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = 2.0 * ((attacked-attacked.min())/(attacked.max()-attacked.min())) - 1.0

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_latent_space_SKL_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_mixed_sum"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = 2.0 * ((attacked-attacked.min())/(attacked.max()-attacked.min())) - 1.0

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_latent_space_mixed_sum_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_mixed_prod"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = 2.0 * ((attacked-attacked.min())/(attacked.max()-attacked.min())) - 1.0

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_latent_space_mixed_prod_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "combi_l2"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_combined_l2_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
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
            #source_im = normalized_attacked


if(attck_type == "combi_cos"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_combined_cosine_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            if step % 10000 == 0:
                #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
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
                #source_im = normalized_attacked



if(attck_type == "combi_cos_lw"):
    adv_div_list = []
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        #loss_to_maximize, adv_gen, source_recon = get_latent_space_l2_loss(normalized_attacked, source_im)

        loss_to_maximize = get_combined_cosine_loss_lw(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
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
            #source_im = normalized_attacked



if(attck_type == "combi_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):

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

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
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
    for step in range(num_steps):

        current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        attacked = (source_im + scaled_noise)
        normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

        loss_to_maximize = get_combined_SKL_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if step % 10000 == 0:
        if step % 10000 == 0:
            #instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            attacked_embed = model.encode(normalized_attacked.to(device))
            xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
            adv_gen = model.render(xT_ad, attacked_embed, T=20)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_im, p=2)
            #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
