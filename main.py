from datetime import datetime
from pathlib import Path
import os

import lightning.pytorch
import torch
from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.cli import LightningCLI
from loss import SatCLIPLoss
from model import SatCLIP

torch.set_float32_matmul_precision('high')

class SatCLIPLightningModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        GS_dim=64,                          # Google Satellite data embedding dimension.
        GS_model='resnet18',                # Google Satellite data model.
        GS_trainable=False,                 # Google Satellite data model trainable. Only use False when a pretrained model is used.
        OSM_dim=64,                         # Open Street Map data embedding dimension.
        OSM_model='hex_conv',               # Open Street Map data model.
        OSM_trainable=True,                 # Open Street Map data model trainable. Only use False when a pretrained model is used.
        hex_numb_rings=3,                   # Number of hexagonal rings used in the hexagonal input data.
        hex_in_channels=6,                  # Number of input channels in the hexagonal data.
        OSM_conv_layers=[16, 8],            # Number of filters in the convolutional layers of the hexagonal data encoder, tuple means multiple layers.
        S2_dim=64,                          # Sentinel 2 data embedding dimension.
        S2_model='moco_vit16',              # Sentinel 2 data model.
        S2_trainable=False,                 # Sentinel 2 data model trainable. Only use False when a pretrained model is used.
        vision_width=128,                   # Vision width for ResNet or ViT fitting. Not used when pretrained model is used.
        vision_patch_size=32,               # Used when fitting a ViT model. Not used when pretrained model is used.
        FM_dim=64,                          # Floodmaps data embedding dimension.
        FM_model='conv',                    # Floodmaps data model.
        FM_trainable=True,                  # Floodmaps data model trainable. Only use False when a pretrained model is used.
        FM_conv_layers=[16, 8],             # Number of filters in the convolutional layers of the floodmap data encoder, tuple means multiple layers.
        Combined_dim=64,                    # Combined data embedding dimension.
        Combined_layers=2,                  # Hidden layers from concatenation of views to embedding.
        Combined_capacity=512,              # Number of nodes in the hidden layers between views and embedding.
        pos_encoder='w3034',                # Type of positional encoder.
        loc_encoder='mlp',                  # Type of location encoder.
        loc_layers=8,                       # Number of hidden layers in location encoder.
        loc_capacity=512,                   # Number of nodes in the hidden layers in location encoder.
        image_resolution=256,               # Resolution of all images, always use 256.
        learning_rate=0.0001,               # Learning rate for gradient descent.
        weight_decay=0.01,                  # Weight decay for gradient descent.
        checkpoint_path=None,               # Path to checkpoint if a trained model is to be preloaded.
        frequency_num=16,                   # Does not matter with analytic SH.
        max_radius=0.01,                    # Does not matter with analytic SH.
        min_radius=0.000001,                # Does not matter with analytic SH.
        harmonics_calculation="analytic",
        legendre_polys=64,                  # Number of Legendre polynomials to be used in positional encoder - more results in a finer grid over the earth.
        sh_embedding_dims=32                # Not used.
    ) -> None:
        super().__init__()

        self.model = SatCLIP(
            GS_dim=GS_dim,
            GS_model=GS_model,
            GS_trainable=GS_trainable,
            OSM_dim=OSM_dim,
            OSM_model=OSM_model,
            OSM_trainable=OSM_trainable,
            hex_numb_rings=hex_numb_rings,
            hex_in_channels=hex_in_channels,
            OSM_conv_layers=OSM_conv_layers,
            S2_dim=S2_dim,
            S2_model=S2_model,
            S2_trainable=S2_trainable,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            FM_dim=FM_dim,
            FM_model=FM_model,
            FM_trainable=FM_trainable,
            FM_conv_layers=FM_conv_layers,
            Combined_dim=Combined_dim,
            Combined_layers=Combined_layers,
            Combined_capacity=Combined_capacity,
            pos_encoder=pos_encoder,
            loc_encoder=loc_encoder,
            loc_layers=loc_layers,
            loc_capacity=loc_capacity,
            image_resolution=image_resolution,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            harmonics_calculation=harmonics_calculation,
            legendre_polys=legendre_polys,
            sh_embedding_dims=sh_embedding_dims,
        )

        self.loss_fun = SatCLIPLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        # Load pretrained weights if checkpoint_path is provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No such file or directory: '{checkpoint_path}'")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dictionaries for each block
        if 'GSview' in checkpoint:
            self.model.GSview.load_state_dict(checkpoint['GSview'], strict=False)
        if 'OSMview' in checkpoint:
            self.model.OSMview.load_state_dict(checkpoint['OSMview'], strict=False)
        if 'S2view' in checkpoint:
            self.model.S2view.load_state_dict(checkpoint['S2view'], strict=False)
        if 'FMview' in checkpoint:
            self.model.FMview.load_state_dict(checkpoint['FMview'], strict=False)
        if 'combined_view' in checkpoint:
            self.model.combined_view.load_state_dict(checkpoint['combined_view'], strict=False)
        if 'location' in checkpoint:
            self.model.location.load_state_dict(checkpoint['location'], strict=False)
        
        print("Checkpoint loaded successfully.")

    def common_step(self, batch, batch_idx):
        gs = batch["gs"]
        osm = batch["osm"]
        s2 = batch["s2"]
        fm = batch["fm"]
        t_points = batch["point"].float()
        logits_per_input, logits_per_coord = self.model(gs, osm, s2, fm, t_points)
        return self.loss_fun(logits_per_input, logits_per_coord)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )

        return optimizer

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")
        parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to load pretrained weights.")

def cli_main(default_config_filename="./configs/default.yaml"):
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")
    # modify configs/default.yaml for learning rate etc.
    cli = MyLightningCLI(
        model_class=SatCLIPLightningModule,
        datamodule_class=S2GeoDataModule,
        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 16,
            "log_every_n_steps": 10,
        },
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"SatCLIP_S2_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name
        # this seems to be necessary to force logging of datamodule hyperparams
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    # Create folder to log configs
    # NOTE: Lightning does not handle config paths with subfolders
    dirname_cfg = Path(default_config_filename).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )

if __name__ == "__main__":

    print(f"Starting run at time {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")

    config_fn = "./configs/default.yaml"

    #A100 go vroom vroom 🚗💨
    if torch.cuda.get_device_name(device=0)=='NVIDIA A100 80GB PCIe':
        torch.backends.cuda.matmul.allow_tf32 = True
        print('Superfastmode! 🚀')
    else:
        print('Not superfastmode 😢')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('medium')

    cli_main(config_fn)