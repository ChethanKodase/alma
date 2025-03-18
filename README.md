# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

# Code for beta-VAE and TC-VAE attacks

##### Install conda environment for adversarial attacks on beta-VAE and TC-VAE

<pre>
```
conda env create -f environment1.yml

```
</pre>


##### Initial setting up

To clone the repository, cd into the repository and create and activate the environment, run the below commands:

<pre>
```
git clone https://github.com/ChethanKodase/alma.git
cd alma
conda env create -f environment1.yml
conda activate your_env
```
</pre>



##### Training beta-VAE:

Follow the commands and assign the locations of your data, checkpoints storage and run time plots as below: 


<pre>
```
python beta_tc_vaes/vae_celebA_training.py --which_gpu 1 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-4 --run_time_plot_dir a_training_runtime --checkpoint_storage vae_checkpoints --model_type VAE
```
</pre>


##### Training TC-VAE:

<pre>
```
python beta_tc_vaes/vae_celebA_training.py --which_gpu 1 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-4 --run_time_plot_dir a_training_runtime --checkpoint_storage vae_checkpoints --model_type TCVAE
```
</pre>

##### TO get condition number of TC-VAE and beta VAE : 

<pre>
```
python beta_tc_vaes/betaVAE_tcVAE_conditioning_analysis.py  --which_gpu 1 --beta_value 5.0 --which_model VAE --checkpoint_storage vae_checkpoints
python beta_tc_vaes/betaVAE_tcVAE_conditioning_analysis.py  --which_gpu 1 --beta_value 5.0 --which_model TCVAE --checkpoint_storage vae_checkpoints
```
</pre>
The condition number plots will b e stored in `alma/conditioning_analysis`

##### Maximum damage attack on beta-VAE and TC-VAE

##### universal attacks: 

<pre>
```
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
```
</pre>
Change `--desired_norm_l_inf` value to required L-inf norm bound on the perturbation 
Change the arguments for `--attck_type` to `latent_l2, latent_wasserstein, latent_SKL, latent_cosine, output_attack_l2, output_attack_wasserstein, output_attack_SKL, output_attack_cosine, lma_l2, lma_wass, lma_skl, lma_cos, alma_l2, alma_wass, alma_skl, alma_cos` to run attacks using rest of all the methods

##### To compare the different adversarial objectives for universal attacks for a given L_infinity norm bound:

<pre>
```
python beta_tc_vaes/analysis_universal_box_plots.py --beta_value 5.0 --which_gpu 1 --model_location vae_checkpoints --l_inf_bound 0.07 --which_model VAE --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --box_plots_directory box_plots --uni_noise_directory beta_tc_vaes/univ_attack_storage
```
</pre>

assign which_model -> TCVAE to plot the same for TCVAE

##### To get qualitative image plots comparing maximum damage attacks between different methods

<pre>
```
python beta_tc_vaes/analysis_universal_image_plots.py --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --qualitative_plots_directory /home/luser/alma/universal_qualitative --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack
```
</pre>


##### To train adversarial filter plugin

<pre>
```
python beta_tc_vaes/betaVAE_tcVAE_attack_filter.py  --which_gpu 0 --beta_value 5.0 --which_model VAE --desired_norm_l_inf 0.09 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --num_steps 300 --filter_location /home/luser/alma/beta_tc_vaes/filter_storage --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack
```
</pre>

##### To plot damage distribution for all epsilon values

<pre>
```
python beta_tc_vaes/analysis_universal_epsilon_variation.py --which_gpu 1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --which_model TCVAE --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack --damage_distributions_address /home/luser/alma/damage_distributions_variation
```
</pre>

# Code for NVAE attacks

We consider the official implementation of NVAE from https://github.com/NVlabs/NVAE. We take the pretrained weights from  the oficial publishers and implement adversarial attacks

#### clone the nvae official repository using the code below: 

git clone https://github.com/NVlabs/NVAE.git

Follow the instructions from https://github.com/NVlabs/NVAE and download the checkpoints for celebA 64 dataset from https://drive.google.com/drive/folders/14DWGte1E7qnMTbAs6b87vtJrmEU9luKn 


#### To create the environment and install dependencies for adversarial attacks on NVAAE

<pre>
```
conda deactivate
cd alma
python -m venv nvaeenv
source nvaeenv/bin/activate
pipenv install -r requirements.txt
```
</pre>


#### To get condition number plots for NVAE

<pre>
```
python nvae/nvae_all_condition_analysis.py --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
```
</pre>



#### To run universal adversarial attacks on NVAE

<pre>
```
cd alma/
source nvaeenv/bin/activate
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_l2" --desired_norm_l_inf 0.035 --data_directory your_data_directory --nvae_checkpoint_path your_checkpoint_directory
```
</pre>

#### To plot distributions of maximum damage attacks using all attacks methods for a given perturbation norms

<pre>
```
python nvae/nvae_all_convergence_qualitative_plots_universal_box_plots.py --data_directory data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise --desired_norm_l_inf 0.05```
</pre>


#### To plot distributions of maximum damage attacks using all attacks methods for different perturbation norms

<pre>
```
python nvae/nvae_all_convergence_epsilon_variation.py --data_directory data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise
```
</pre>


#### To train Adversarial Filter Plugin(AFP) for NVAE

<pre>
```
python nvae/nvae_all_filtration.py --feature_no 2 --source_segment 0 --attck_type "combi_l2_cond" --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --data_directory data_cel1  --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise --desired_norm_l_inf 0.05 --filter_location nvae/filter_storage
```
</pre>


#### To get filtered reconstructions using Adversarial Filter Plugin(AFP) for NVAE and compare 

<pre>
```
python nvae/nvae_all_filtration_analysis.py --feature_no 2 --source_segment 0 --attck_type "combi_l2_cond" --data_directory data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --afp_plugin_location ../NVAE/filtration/filter_model --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise --compare_plots_storage nvae/filter_storage/analysis_comparision
```
</pre>


# Code of DiffAE attacks


Arguments for commands: 
1. `desired_norm_l_inf`:  L-infinity bound on the added adversarial noise
2. `attck_type` : Choose the attack method from `la_l2, la_wass, la_skl, la_skl, la_cos, alma_l2, alma_wass, alma_skl, alma_cos`. Descriptions for these methods are given in our paper.
3. `diffae_checkpoint` : Address of the downloaded trained DiffAE model weights from the publishers of https://arxiv.org/pdf/2111.15640 , code: https://github.com/phizaz/diffae 
4. `ffhq_images_directory` : address of the FFHQ images directory
5. `noise_directory` : Directory where the optimized noise is saved.
6. `which_gpu` : Enter the index of the GPU you want to use 


#### Install the conda environment required and activate:


<pre>
```
conda env create -f environment2.yml
cd alma
conda activate your_diffae_environment
```
</pre>


#### To get layerwise condition numbers plots for DiffAE:


<pre>
```
python diffae/autoencoding_diffAE_conditioning_analysis.py --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints
```
</pre>

Layerwise condition number chart will be saved to `alma/conditioning_analysis`


#### To optimize universal adversarial attacks on DiffAE:


Follow https://github.com/phizaz/diffae to download the checkpoints and FFHQ dataset 

<pre>
```
python diffae/autoencoding_attack_universal.py --desired_norm_l_inf 0.35 --attck_type alma_cos --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
```
</pre>
Optimized noise will be saved in `alma/diffae/noise_storage`


![DiffAE Qualitative](diffae/showcase/paperDiffAE_all_attacks_norm_bound_0.08_segment_3.png)
*Qualitative comparison of DiffAE reconstructions of adversarial examples crafted from  different attacks methods*

#### To get a comparative box plot of all the adversarial attack methods for fiven perturbation norm run : 

<pre>
```
python diffae/attack_universal_quantitative.py --desired_norm_l_inf 0.31 --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --noise_directory ../diffae/attack_run_time_univ/attack_noise
python diffae/attack_convergence_compare_universal_quantitative_box_plots.py  --desired_norm_l_inf 0.31 --which_gpu 7 

```
</pre>

#### To plot adversrial reconstruction loss distribution for all attack methods for a set of L-infinity norms 

<pre>
```
python diffae/attack_convergence_epsilon_variation.py --epsilon_list 0.27 0.28 0.29 0.3 0.31 0.32 0.33
```
</pre>