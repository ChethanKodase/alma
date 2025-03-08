# alma
ALMA-Aggregated-Lipschitz-Maximization-Attack-for-Autoencoders

#create the environment for adversarial attacks in beta-VAE and TC-VAE using yml file

### Training beta-VAEs and TC-VAEs:

Follow the commands : 

`conda deactivate`

`cd alma/beta_tc_vaes/`

`conda activate your_env`

`python vae_celebA_training.py --which_gpu 0 --beta_value 1.0`

