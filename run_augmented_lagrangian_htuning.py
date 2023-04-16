from Wind.load_data import DataLoader, get_test_dataset
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass, asdict

from Wind.AugmentedLagrangian.augmented_lagrangian import Hyperparameters, AugmentedLagrangian
from Wind.AugmentedLagrangian.loss_functions import loss_var, loss_mean_cov, loss_mean_var

pd.options.plotting.backend = "plotly"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

# Use Apples Metal Performance Shaders (MPS) if available
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     device = torch.device("mps")

hparams_dict = {
    "n_steps": 250,
    "n_iterations": 1000,
    "rho": 2.0,
    "rho_scaling": 1.0,
    "step_size": 5e-3,
    "init_lmbda": -0.05,
    "init_mu": 0.001,
}

# Use test dataset
# Y = torch.from_numpy(get_test_dataset()).to(device=device)

# Use Wind dataset
data_loader = DataLoader()

Y = data_loader.df.values.T
Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)

hparams = Hyperparameters(**hparams_dict)

step_sizes = [2e-2, 1.5e-2]
rhos = [2]
rho_scalings = [1]
hparams.n_iterations = 200

loss_fns_dict = {"Var": lambda x: loss_var(x, Y), "-Mean+Var": lambda x: loss_mean_var(x, Y, alp=0.2)}
loss = "-Mean+Var"

for step_size in step_sizes:
    hparams.step_size = step_size
    for rho in rhos:
        hparams.rho = rho
        for rho_scaling in rho_scalings:
            try:
                hparams_as_str = "|".join([f"{key}={value}" for key, value in asdict(hparams).items()])
                print(f"{hparams_as_str}")
                hparams.rho_scaling = rho_scaling
                model = AugmentedLagrangian(
                    Y,
                    hyperparameters=hparams,
                    loss_fn=loss_fns_dict[loss],
                    writer=SummaryWriter(log_dir=f"logs/{hparams_as_str}/{datetime.now().isoformat()}"),
                )
                model.train()
            except Exception as e:
                print(f"Failed with exception {e}")
