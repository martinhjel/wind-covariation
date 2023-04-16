import argparse
import pickle
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from Wind.load_data import DataLoader, get_test_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Wind.AugmentedLagrangian.augmented_lagrangian import Hyperparameters, AugmentedLagrangian
from Wind.AugmentedLagrangian.loss_functions import loss_var, loss_mean_var

device = "cpu"

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="")
parser.add_argument("--table_output", required=True, help="")
parser.add_argument("--models_output", required=True, help="")
args = parser.parse_args()


# Use Wind dataset
data_loader = DataLoader(data_folder_path=args.input)

norwegian_areas = data_loader.df_nve_wind_locations["location"].to_list()
all_areas = data_loader.df_locations["location"].to_list()


Y = data_loader.df.values.T
Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)

loss_fns_dict = {
    "-0.5*Mean+0.5*Var": lambda x: loss_mean_var(x, Y, alp=0.5),
    "-0.3*Mean+0.7*Var": lambda x: loss_mean_var(x, Y, alp=0.3),
    "-0.1*Mean+0.9*Var": lambda x: loss_mean_var(x, Y, alp=0.1),
    "-0.0*Mean+1.0*Var": lambda x: loss_mean_var(x, Y, alp=0.0),
}

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

weights = []
models = {}
for loss_fn in tqdm(loss_fns_dict):
    writer = SummaryWriter(log_dir=f"logs/{loss_fn}/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    models[loss_fn] = AugmentedLagrangian(
        df=data_loader.df, hyperparameters=hparams, loss_fn=loss_fns_dict[loss_fn], writer=writer, device=device
    )
    models[loss_fn].train()
    weights.append(models[loss_fn].xs[-1].numpy())

# %%
df_weights = pd.DataFrame(np.array(weights).T * 100, index=data_loader.df.columns, columns=loss_fns_dict.keys()).iloc[
    ::-1
]

properties = {}
for loss_fn in loss_fns_dict:
    properties[loss_fn] = {}

    y = torch.matmul(models[loss_fn].xs[-1], Y) * 100
    properties[loss_fn]["Mean"] = y.mean().numpy()
    properties[loss_fn]["Variance"] = y.std().numpy()

df_properties = pd.DataFrame(properties, dtype=np.float32)
df_properties.info()

df_table = pd.concat((df_weights, df_properties))

# store_path = Path.cwd()/"data/processed/augmented_lagrangian"
# store_path.mkdir(exist_ok=True)

models_output_file = Path(args.model_output)
with open(models_output_file, "wb") as f:
    pickle.dump(models, f)

table_output_file = Path(args.table_output)
df_table.to_pickle(table_output_file)
