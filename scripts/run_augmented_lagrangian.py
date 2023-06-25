import os
import sys
import argparse
import pickle
from pathlib import Path
import time
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from _helpers import configure_logging

sys.path.insert(1, "/Users/martihj/gitsource/wind-covariation") # Absolute path due to issues with snakemake
from Wind.AugmentedLagrangian.augmented_lagrangian import Hyperparameters, AugmentedLagrangian
from Wind.AugmentedLagrangian.loss_functions import loss_mean_var
from Wind.load_data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("run_augmented_lagrangian", area="norwegian", bias="bias_false", alpha=0.0)

    configure_logging(snakemake)
    logger.info("Started augmented lagrangian script.")

    device = "cpu"

    data_folder = Path(snakemake.input.data_folder )
    data_loader = DataLoader(data_folder_path=data_folder)
    
    alpha = float(snakemake.params.alpha)
    area = snakemake.params.area
    bias = snakemake.params.bias
    output_path = snakemake.output.model

    logger.info(f"{area}, {bias}, {alpha}")

    norwegian_areas = data_loader.df_nve_wind_locations["location"].to_list()
    all_areas = data_loader.df_locations["location"].to_list()

    loss_fn_str = f"-{alpha}*Mean+{1-alpha}*Var"

    hparams = Hyperparameters(
        **{
            "n_steps": 250,
            "n_iterations": 1000,
            "rho": 2.0,
            "rho_scaling": 1.0,
            "step_size": 0.02,
            "init_lmbda": -0.05,
            "init_mu": 0.001,
        }
    )

    df = data_loader.df
 
    area_map = {
        "all": all_areas,
        "norwegian": norwegian_areas
    }

    use_bias = True if bias == "bias_true" else False

    trainable_cols = df.columns.isin(area_map[area])
    
    writer = SummaryWriter(log_dir=f"logs/{loss_fn_str}/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    model = AugmentedLagrangian(
        df=df,
        trainable_cols=trainable_cols,
        use_bias = use_bias,
        hyperparameters=hparams,
        loss_fn=loss_mean_var,
        writer=writer,
        alpha=alpha,
        device=device,
        logger=logger,
    )
    model.train()

    out_path = Path(output_path)
    
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
