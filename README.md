# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

#create the environment for adversarial attacks in beta-VAE and TC-VAE using yml file

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