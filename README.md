# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

#create the environment for adversarial attacks in beta-VAE and TC-VAE using yml file

### install conda environment for adversarial attacks on beta-VAE and TC-VAE

<pre>
```
conda env create -f environment1.yml

```
</pre>


### Training beta-VAE:

Follow the commands and assign the locations of your data, checkpoints storage and run time plots as below: 


<pre>
```
cd alma/beta_tc_vaes/
conda activate your_env
python beta_tc_vaes/vae_celebA_training.py --which_gpu 1 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-4 --run_time_plot_dir /home/luser/alma/a_training_runtime --checkpoint_storage /home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints

```
</pre>


### Training TC-VAE:

<pre>
```
python TC_vae_celebA_training.py --which_gpu 0 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-6 --run_time_plot_dir /home/luser/autoencoder_attacks/a_training_runtime --checkpoint_storage /home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints
```
</pre>

### TO get condition number of TC-VAE and beta VAE : 

<pre>
```
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/betaVAE_tcVAE_conditioning_analysis.py  --which_gpu 0 --beta_value 5.0 --which_model VAE --checkpoint_storage /home/luser/autoencoder_attacks/saved_celebA/checkpoints
```
</pre>


### Maximum damage attack on beta-VAE and TC-VAE

## universal attacks: 

<pre>
```
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_l2 --which_model VAE --desired_norm_l_inf 0.1 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints  --num_steps 300
```
</pre>


## To compare the different adversarial objectives for universal attacks for a given L_infinity norm bound:

<pre>
```
python beta_tc_vaes/analysis_universal_box_plots.py --beta_value 5.0 --which_gpu 1 --model_location /home/luser/autoencoder_attacks/saved_celebA/checkpoints --l_inf_bound 0.07 --which_model VAE --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --box_plots_directory /home/luser/alma/box_plots
```
</pre>
