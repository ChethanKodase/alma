# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

#create the environment for adversarial attacks in beta-VAE and TC-VAE using yml file

### Training beta-VAEs and TC-VAEs:

Follow the commands : 

`conda deactivate`

`cd alma/beta_tc_vaes/`

`conda activate your_env`


<pre>
```bash
python vae_celebA_training.py \
    --which_gpu 0 \
    --beta_value 5.0 \
    --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-6 \
    --run_time_plot_dir /home/luser/autoencoder_attacks/a_training_runtime \
    --checkpoint_storage /home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints
```
</pre>
