import torch
import torch.nn as nn
import torch.nn.functional as F


criterion = nn.MSELoss()


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

def get_latent_space_l2_loss(normalized_attacked, source_im, model, device):
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  criterion(z1, z2) 

    return loss_to_maximize

def get_latent_space_wasserstein_loss(normalized_attacked, source_im, model, device):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  wasserstein_distance(z1, z2) 

    return loss_to_maximize, adv_gen, source_recon


def compute_mean_and_variance(tensor):
    flattened_tensor = torch.flatten(tensor)  # Flatten the tensor
    mean = torch.mean(flattened_tensor)  # Compute mean
    variance = torch.var(flattened_tensor, unbiased=False)  # Compute variance (unbiased=False for population variance)
    return mean, variance

def get_symmetric_KLDivergence(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl




def get_latent_space_SKL_loss(normalized_attacked, source_im, model, device):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2) 

    return loss_to_maximize, adv_gen, source_recon


def get_latent_space_cosine_loss(normalized_attacked, source_im, model, device):
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    return (cos(z1, z2)-1.0)**2 




def get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2)   * get_symmetric_KLDivergence(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def output_attack_l2(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def output_attack_wasserstein(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = wasserstein_distance(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def output_attack_SKL(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def output_attack_cosine(normalized_attacked, source_im, model):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
     
    loss_to_maximize = (cos(normalized_attacked, adv_gen)-1.0)**2

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_k_eq_latent_l2(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  criterion(z1, z2)   * criterion(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_k_eq_latent_wasserstein(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  wasserstein_distance(z1, z2)   * wasserstein_distance(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2)   * get_symmetric_KLDivergence(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_k_eq_latent_cos(normalized_attacked, source_im, model, device):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  ((cos(z1, z2)-1.0)**2)   * (cos(source_recon, adv_gen)-1.0)**2

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im, cond_normal, model, device):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = criterion(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = criterion(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)

    return loss_to_maximize, attack_flow, source_flow



def get_weighted_combinations_aclmd_wasserstein_cond(normalized_attacked, source_im, cond_normal, model, device):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += wasserstein_distance(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = wasserstein_distance(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = wasserstein_distance(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = wasserstein_distance(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += wasserstein_distance(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * wasserstein_distance(source_flow, attack_flow)

    return loss_to_maximize, attack_flow, source_flow



def get_weighted_combinations_aclmd_SKL_cond(normalized_attacked, source_im, cond_normal, model, device):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += get_symmetric_KLDivergence(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = get_symmetric_KLDivergence(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = get_symmetric_KLDivergence(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = get_symmetric_KLDivergence(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += get_symmetric_KLDivergence(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * get_symmetric_KLDivergence(source_flow, attack_flow)

    return loss_to_maximize, attack_flow, source_flow



def get_weighted_combinations_aclmd_cos_cond(normalized_attacked, source_im, cond_normal, model, device):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += (cos(attack_out, source_out)-1.0)**2  *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = (cos(mu1, mu2)-1.0)**2  *cond_normal[l_ct]
    l_ct += 1

    rep_loss = (cos(std1 * esp1, std2 * esp2)-1.0)**2 *cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = (cos(attack_flow, source_flow)-1.0)**2  *cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += (cos(attack_out, source_out)-1.0)**2 *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * (cos(source_flow, attack_flow)-1.0)**2

    return loss_to_maximize, attack_flow, source_flow