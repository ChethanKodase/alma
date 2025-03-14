# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

# Code for beta-VAE and TC-VAE

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

# Code for NVAE

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



#### To run universal adversarial attacks on NVAE

<pre>
```
cd alma/
source nvaeenv1/bin/activate
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_l2" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_wass" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_skl" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "alma_cos" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_l2" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_wass" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_skl" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_cos" --desired_norm_l_inf 0.035 --data_directory /mdadm0/chethan_krishnamurth/data_cel1 --nvae_checkpoint_path /mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint

```
</pre>