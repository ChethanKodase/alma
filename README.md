# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

#create the environment for adversarial attacks in beta-VAE and TC-VAE using yml file

### install conda environment for adversarial attacks on beta-VAE and TC-VAE

conda env create -f environment1.yml

### Training beta-VAE:

Follow the commands and assign the locations of your data, checkpoints storage and run time plots as below: 


<pre>
```
cd alma/beta_tc_vaes/
conda activate your_env
python vae_celebA_training.py \
    --which_gpu 0 \
    --beta_value 5.0 \
    --data_directory location of yor data \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-6 \
    --run_time_plot_dir your run time plot directory \
    --checkpoint_storage your checkpoint storage directory
```
</pre>


### Training TC-VAE:

<pre>
```
python TC_vae_celebA_training.py \
    --which_gpu 0 \
    --beta_value 5.0 \
    --data_directory location of yor data \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-6 \
    --run_time_plot_dir your run time plot directory \
    --checkpoint_storage your checkpoint storage directory
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
cd alma
conda activate /home/luser/anaconda3/envs/inn
python beta_tc_vaes/betaVAE_all_kinds_of_attacks_universal.py  --which_gpu 0 --beta_value 5.0 --attck_type aclmd_l2a_cond --which_model VAE --desired_norm_l_inf 0.1
```
</pre>