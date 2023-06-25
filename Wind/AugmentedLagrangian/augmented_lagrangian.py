import logging
from typing import Callable, Optional, List
from tensorboard.backend.event_processing import event_accumulator
from Wind.load_data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.base import BaseEstimator
import time

from .loss_functions import lf, lf_rho, loss_var, loss_mean_var, loss_mean_cov


@dataclass
class Hyperparameters:
    n_steps: int
    n_iterations: int
    rho: float
    rho_scaling: float
    step_size: float
    init_lmbda: float
    init_mu: float


class AugmentedLagrangian(BaseEstimator):
    def __init__(
        self,
        df: pd.DataFrame,
        trainable_cols: List[str],
        use_bias: bool,
        hyperparameters: Hyperparameters,
        loss_fn: Callable,
        writer: SummaryWriter,
        alpha: float,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger("__name__")
        self.hparams = hyperparameters
        self.device = device
        self.writer = writer
        self.df = df

        self.df_trainable = df.loc[:,trainable_cols]
        self.df_not_trainable = df.loc[:,~trainable_cols]

        bias = np.zeros(len(self.df_trainable))
        n_not_trainable = (~trainable_cols).sum()
        if n_not_trainable > 0 and use_bias:
            weighting = n_not_trainable/len(df.columns)
            bias = self.df_not_trainable.mean(axis=1).values*weighting # Weight so that sum is still 1
            self.logger.info(f"{n_not_trainable} not trainable weights.")
            
        self.bias = torch.from_numpy(bias).to(dtype=torch.float32,device=device)
        self.logger.info(f"{bias.mean()} mean bias.")

        if use_bias: 
            self.sum_weight = len(self.df_trainable.columns)/len(df.columns)
        else:
            self.sum_weight = 1.0
        self.sum_weight = torch.tensor(self.sum_weight, dtype=torch.float32, device=device)

        self.Y = torch.from_numpy(self.df_trainable.values.T).to(dtype=torch.float32, device=device)
        self.alpha = alpha
        self.loss_fn_ = loss_fn

        # Initialize x evenly
        self.n_weights = self.df_trainable.values.T.shape[0]
        x = np.ones(self.n_weights, dtype=np.float32, )
        x = x / x.sum(axis=-1)
        x = torch.from_numpy(x)
        x.requires_grad = True
        self.x = x.to(self.device) * self.sum_weight # Scale so all weights sum to 1 initially

        # Langrange multipliers
        self.lmbda = torch.tensor(self.hparams.init_lmbda, requires_grad=True, device=self.device)
        self.mu = torch.tensor(
            [self.hparams.init_mu for i in range(self.n_weights)],
            requires_grad=True,
            dtype=torch.float32,
            device=self.device,
        )

        # Penalty variables
        self.rho_scaling = torch.tensor(self.hparams.rho_scaling, dtype=torch.float32)
        self.rho = torch.tensor(self.hparams.rho, dtype=torch.float32)
        I_rho = np.eye(self.n_weights, dtype=np.float32)
        I_rho = torch.from_numpy(I_rho) * self.hparams.rho
        self.I_rho = I_rho.to(device=self.device)

        # Log x and multipliers
        xs = torch.tensor([], device=self.device)
        self.xs = torch.cat((xs, self.x.detach().view(1, -1)))

        mus = torch.tensor([], device=self.device)
        self.mus = torch.cat((mus, self.mu.detach().view(1, -1)))

    def loss_fn(self, x):
        return self.loss_fn_(x, self.Y, alp=self.alpha, bias=self.bias)

    def _update_penalty(self):
        with torch.no_grad():
            self.rho = self.rho * self.hparams.rho_scaling
            zero = torch.tensor(0.0, dtype=torch.float32)
            for i in range(self.n_weights):
                if (-self.x[i]) < zero and torch.isclose(self.mu[i], zero):
                    self.I_rho[i, i] = 0
                else:
                    self.I_rho[i, i] = self.rho

    def _update_multipliers(self):
        with torch.no_grad():
            self.lmbda = self.lmbda + self.rho * (self.x.sum() - self.sum_weight)
            self.mu = torch.maximum(torch.zeros_like(self.x), self.mu + self.rho * (-self.x))

    def _are_kkt_conditions_verified(self, atol=1e-4):
        # dx L = 0
        dx = torch.autograd.grad(lf(self.x, self.lmbda, self.mu, self.loss_fn), self.x)[0]
        if torch.isclose(dx, torch.zeros_like(dx), atol=atol).all():
            # c(x) = 0 | x.sum()-1 = 0
            if torch.isclose((self.x.sum() - self.sum_weight), torch.tensor(0.0, dtype=torch.float32), atol=atol):
                # h(x) <= 0 | (-x) <= 0
                if ((-self.x) <= 0.0).all():
                    # mu >= 0
                    if (self.mu >= 0.0).all():
                        # mu*.h(x) = 0
                        if torch.isclose((-self.x) * self.mu, torch.zeros_like(self.mu), atol=atol).all():
                            return True

        return False

    def train(self, convergence_tolerance=1e-4):
        start_time = time.perf_counter()
        # Hyperparamters
        step_size = torch.tensor(self.hparams.step_size, dtype=torch.float32)

        with self.writer:
            # Training loop
            for it in range(self.hparams.n_iterations):
                # Solve gradient descent for current lagrangian multipliers
                for i in range(self.hparams.n_steps):
                    obj = lf_rho(self.x, self.lmbda, self.mu, self.rho, self.I_rho, loss_fn=self.loss_fn)
                    dx = torch.autograd.grad(obj, self.x)
                    with torch.no_grad():
                        self.x -= step_size * dx[0]

                    self.writer.add_histogram("x", self.x.detach().clone().cpu().numpy(), it * self.hparams.n_steps + i)

                    # Log scalar values for each step
                    self.writer.add_scalars(
                        "Function values",
                        {
                            "f": self.loss_fn(self.x).item(),
                            "Lf": lf(self.x, self.lmbda, self.mu, loss_fn=self.loss_fn).item(),
                            "Lf_rho": obj.item(),
                        },
                        it * self.hparams.n_steps + i,
                    )

                # Log multipliers and rho for each iteration
                self.writer.add_histogram("lmbda", self.lmbda.detach().clone().cpu().numpy(), it)
                self.writer.add_histogram("mu", self.mu.detach().clone().cpu().numpy(), it)
                self.writer.add_histogram("rho", self.rho.detach().clone().cpu().numpy(), it)

                # Log x and mu
                self.xs = torch.cat((self.xs, self.x.detach().view(1, -1)))
                self.mus = torch.cat((self.mus, self.mu.detach().view(1, -1)))

                # Update lagrangian multipliers and rho
                self._update_multipliers()
                self._update_penalty()

                # Assert KKT Conditions
                converged = self._are_kkt_conditions_verified(atol=convergence_tolerance)

                if it % 10 == 0:
                    print(
                        f"Iteration: {it} - objective value lagrangian: {lf(self.x, self.lmbda, self.mu, loss_fn=self.loss_fn).item():.5f}"
                    )

                if converged:
                    print(
                        f"Iteration: {it} - objective value lagrangian: {lf(self.x, self.lmbda, self.mu, loss_fn=self.loss_fn).item():.5f}"
                    )
                    print("KKT conditions met")
                    break

            self.writer.add_hparams(asdict(self.hparams), {"loss": obj.item()})
        print(f"Computation time {time.perf_counter()-start_time:.4f} s")

    def get_scalar_data_from_summary_writer(self, scalar="f"):
        summary_dir = self.writer.log_dir + "/Function values_" + scalar
        # Create an EventAccumulator object
        event_acc = event_accumulator.EventAccumulator(summary_dir)

        # Load the data from the event files
        event_acc.Reload()

        # event_acc.Tags()['scalars']
        # Extract the "scalar" values and their timestamps
        scalar_data = event_acc.Scalars("Function values")

        # Extract the values and timestamps for each scalar
        values = [scalar.value for scalar in scalar_data]
        timestamps = [scalar.wall_time for scalar in scalar_data]

        return values
