# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

#create the environment for adversarial attacks in beta-VAE and TC-VAE using yml file

### Training beta-VAEs and TC-VAEs:

Follow the commands : 

`conda deactivate`

`cd alma/beta_tc_vaes/`

`conda activate your_env`


<pre>
```
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
