import os
import sys
import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from Wind.load_data import DataLoader
from Wind.AugmentedLagrangian.loss_functions import loss_var, loss_mean_var
from Wind.AugmentedLagrangian.augmented_lagrangian import Hyperparameters, AugmentedLagrangian


def main():
    device = "cpu"

    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True, help="")
    parser.add_argument("--models-output", required=True, help="")
    parser.add_argument("--alpha", required=True, help="")
    args = parser.parse_args()

    # Use Wind dataset
    data_folder = Path(args.data_folder)
    data_loader = DataLoader(data_folder_path=data_folder)

    norwegian_areas = data_loader.df_nve_wind_locations["location"].to_list()
    all_areas = data_loader.df_locations["location"].to_list()

    alpha = float(args.alpha)

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

    writer = SummaryWriter(log_dir=f"logs/{loss_fn_str}/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    model = AugmentedLagrangian(
        df=data_loader.df, hyperparameters=hparams, loss_fn=loss_mean_var, writer=writer, alpha=alpha, device=device
    )
    model.train()

    models_output_file = Path(args.models_output)
    with open(models_output_file, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
