# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

#create the environment for adversarial attacks in beta-VAE and TC-VAE using yml file

### install conda environment for adversarial attacks on beta-VAE and TC-VAE

<pre>
```
conda env create -f environment1.yml

```
</pre>


### Initial setting up

To clone the repository, cd into the repository and create and activate the environment, run the below commands:

<pre>
```
git clone https://github.com/ChethanKodase/alma.git
cd alma
conda env create -f environment1.yml
conda activate your_env
```
</pre>



### Training beta-VAE:

Follow the commands and assign the locations of your data, checkpoints storage and run time plots as below: 


<pre>
```
python beta_tc_vaes/vae_celebA_training.py --which_gpu 1 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-4 --run_time_plot_dir a_training_runtime --checkpoint_storage vae_checkpoints --model_type VAE
```
</pre>


### Training TC-VAE:

<pre>
```
python beta_tc_vaes/vae_celebA_training.py --which_gpu 1 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-4 --run_time_plot_dir a_training_runtime --checkpoint_storage vae_checkpoints --model_type TCVAE
```
</pre>

### TO get condition number of TC-VAE and beta VAE : 

<pre>
```
python beta_tc_vaes/betaVAE_tcVAE_conditioning_analysis.py  --which_gpu 1 --beta_value 5.0 --which_model VAE --checkpoint_storage vae_checkpoints
python beta_tc_vaes/betaVAE_tcVAE_conditioning_analysis.py  --which_gpu 1 --beta_value 5.0 --which_model TCVAE --checkpoint_storage vae_checkpoints
```
</pre>
The condition number plots will b e stored in `alma/conditioning_analysis`

### Maximum damage attack on beta-VAE and TC-VAE

## universal attacks: 

<pre>
```
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location vae_checkpoints  --num_steps 300 --runtime_plots_location beta_tc_vaes/optimization_time_plots --attack_store_location beta_tc_vaes/univ_attack_storage
```
</pre>
Change `--desired_norm_l_inf` value to required L-inf norm bound on the perturbation 
Change the arguments for `--attck_type` to `latent_l2, latent_wasserstein, latent_SKL, latent_cosine, output_attack_l2, output_attack_wasserstein, output_attack_SKL, output_attack_cosine, lma_l2, lma_wass, lma_skl, lma_cos, alma_l2, alma_wass, alma_skl, alma_cos` to run attacks using rest of all the methods

## To compare the different adversarial objectives for universal attacks for a given L_infinity norm bound:

<pre>
```
python beta_tc_vaes/analysis_universal_box_plots.py --beta_value 5.0 --which_gpu 1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --l_inf_bound 0.07 --which_model VAE --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --box_plots_directory /home/luser/alma/box_plots --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack
```
</pre>


## To get qualitative image plots comparing maximum damage attacks between different methods

<pre>
```
python beta_tc_vaes/analysis_universal_image_plots.py --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --qualitative_plots_directory /home/luser/alma/universal_qualitative --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack
```
</pre>


## To train adversarial filter plugin

<pre>
```
python beta_tc_vaes/betaVAE_tcVAE_attack_filter.py  --which_gpu 0 --beta_value 5.0 --which_model VAE --desired_norm_l_inf 0.09 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --num_steps 300 --filter_location /home/luser/alma/beta_tc_vaes/filter_storage --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack
```
</pre>

## To plot damage distribution for all epsilon values

<pre>
```
python beta_tc_vaes/analysis_universal_epsilon_variation.py --which_gpu 1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --which_model TCVAE --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --uni_noise_directory /home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack --damage_distributions_address /home/luser/alma/damage_distributions_variation
```
</pre>