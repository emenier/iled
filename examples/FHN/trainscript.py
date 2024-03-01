import argparse
import os
import os.path as osp
from torch.utils.data import DataLoader
import sys
import torch
import torch.nn as nn
import numpy as np
import iled
parser = argparse.ArgumentParser(
        description="Train an iLED model on the FHN case",
    )
parser.add_argument("--work_dir", type=str, 
                        help="Path to the work directory where data and runs will be saved")
parser.add_argument("--run_name", type=str, help="Name of the training run")
args = parser.parse_args()


seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

work_dir = args.work_dir
save_dir = args.run_name
if not osp.exists(osp.join(work_dir,'runs')):
    os.mkdir(osp.join(work_dir,'runs'))

train_dataset = iled.data_utils.FHNDataset(osp.join(work_dir, "data"), "train")
val_dataset = iled.data_utils.FHNDataset(osp.join(work_dir, "data"), "val")

act= nn.ReLU()

#### AutoEncoder ####
encoder_config = iled.cnn1d.CNN1DEncoderConfig(
    activation=act,
    activation_output=nn.Identity(),
    kernel_size=5,
    dim_input=101,
    dim_latent=2,
    layer_channels=[2, 8, 16, 32, 4],
    latent_centering=True,
)
decoder_config = iled.cnn1d.CNN1DDecoderConfig(
    activation=act,
    activation_output=iled.activations.TanhPlus(),
    kernel_size=5,
    dim_input=101,
    dim_latent=2,
    layer_channels=[4, 32, 16, 8, 2],
    unflatten_shape=(4, 8),
    bias=True,
)
scaler_config = iled.scalers.MinMaxScalerConfig(
    data_min=train_dataset.data_min, data_max=train_dataset.data_max
)

ae_config = iled.autoencoders.AutoEncoderConfig(encoder_config, decoder_config)

#### Dynamics ####
dynamics_config = iled.splitdynamics.SplitDynamicsConfig(
    dim_latent=2,
    dim_hidden=16,
    activation=nn.SiLU(),
    linear_operator="unconstrained",
    nl_operator="unconstrained",
    nl_width=32,
    default_substeps=1,
    nl_n_hidden_layers=2,
    zero_init=True,
)

#### End To End Model ####
end_to_end_config = iled.endtoend.EndToEndConfig(
    n_warmup=20,
    init_nTmax=20,
    substeps=1,
    ae_config=ae_config,
    dynamics_config=dynamics_config,
)

#### Trainer ####
losses_and_scales = {
    "reconstruction": ["mse", 1],
    "latent_forecast": ["mse", 1],
    #"reconstructed_forecast": ["mse", 1e-2],
    "nl_penalisation": ["norm_loss", 1e-4],
    #'latent_center': ['centering_loss',1e-1]
}

trainer_config = iled.trainer.TrainerConfig(
    model_config=end_to_end_config,
    scaler_config=scaler_config,
    losses_and_scales=losses_and_scales,
    dtype=torch.float,
    save_path=osp.join(work_dir, "runs", save_dir),
    lr=1e-4,
    weight_decay=1e-6,
    max_patience=200,
    t_increment_patience=25,
    nT_increment=5,
    target_length=120,
    cuda=torch.cuda.is_available(),
    validate_every=1,
    checkpoint_every=10,
    lr_reduction_order = 2
)

#### Instantiation and training ####
trainer = trainer_config.make()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

trainer.train(train_loader, val_loader)
