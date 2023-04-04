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

pd.options.plotting.backend = "plotly"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

# Use Apples Metal Performance Shaders (MPS) if available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")


def update_identity_rho(x, I, rho, mu, n_weights):
    zero = torch.tensor(0.0, dtype=torch.float32)
    for i in range(n_weights):
        if (-x[i]) < zero and torch.isclose(mu[i], zero):
            I[i, i] = 0
        else:
            I[i, i] = rho


def update_dual_variables(lmbda, mu, rho, x):
    with torch.no_grad():
        lmbda = lmbda + rho * (x.sum() - 1)
        mu = torch.maximum(torch.zeros_like(x), mu + rho * (-x))


def are_kkt_conditions_verified(x, lmbda, mu, loss_fn, atol=1e-4):
    # dx L = 0
    dx = torch.autograd.grad(lf(x, lmbda, mu, loss_fn), x)[0]
    if torch.isclose(dx, torch.zeros_like(dx), atol=atol).all():
        # c(x) = 0 | x.sum()-1 = 0
        if torch.isclose((x.sum() - 1), torch.tensor(0.0, dtype=torch.float32), atol=atol):
            # h(x) <= 0 | (-x) <= 0
            if ((-x) <= 0.0).all():
                # mu >= 0
                if (mu >= 0.0).all():
                    # mu*.h(x) = 0
                    if torch.isclose((-x) * mu, torch.zeros_like(mu), atol=atol).all():
                        return True

    return False


def loss_var(x, Y, **kwargs):
    y = torch.matmul(x, Y)
    return y.var()


def loss_cov(x, Y, **kwargs):
    y = x[:, None] * Y
    cov_matrix = y.cov()
    return torch.triu(cov_matrix, diagonal=1).sum()


def loss_mean_var(x, Y, alp):
    y = x[:, None] * Y
    return -alp * y.mean() + (1 - alp) * y.var()


def loss_mean_cov(x, Y, alp):
    y = x[:, None] * Y
    cov_matrix = y.cov()
    return -alp * y.mean() + (1 - alp) * torch.triu(cov_matrix, diagonal=1).sum()


def lf(x, lmbda, mu, loss_fn):
    return loss_fn(x) + lmbda * (x.sum() - 1) + torch.matmul(-x, mu)


def lf_rho(x, lmbda, mu, rho, I_rho, loss_fn):
    return (
        lf(x, lmbda, mu, loss_fn) + rho / 2 * (x.sum() - 1) ** 2 + 1 / 2 * torch.matmul(torch.matmul(-x, I_rho), (-x))
    )


# Generate test dataset

N = 1000
s = np.linspace(0, 10, N, dtype=np.float32)
y1 = np.sin(s)
y2 = np.cos(s)
plt.plot(y1)
plt.plot(y2)

Y = np.stack((y1, y2), axis=-1).T

Y = torch.from_numpy(Y)
Y = Y.to(device)

# Use Wind dataset

data_loader = DataLoader()

Y = data_loader.df.values.T
Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)

# %%%


@dataclass
class Hyperparameters:
    n_steps: int
    n_iterations: int
    rho: float
    rho_scaling: float
    step_size: float
    init_lmbda: float
    init_mu: float


def train(Y, loss_fn, hparams: Hyperparameters, writer):
    n_weights = Y.shape[0]
    x = np.ones(n_weights, dtype=np.float32)
    x = x / x.sum(axis=-1)
    x = torch.from_numpy(x)
    x.requires_grad = True
    x = x.to(device)

    # Hyperparamters
    rho = torch.tensor(hparams.rho, dtype=torch.float32)  # Testdata = 1.1
    rho_scaling = torch.tensor(hparams.rho_scaling, dtype=torch.float32)  # Testdata= 1.1
    step_size = torch.tensor(hparams.step_size, dtype=torch.float32)
    n_steps = hparams.n_steps
    n_iterations = hparams.n_iterations

    # Langrange multipliers
    lmbda = torch.tensor(hparams.init_lmbda, requires_grad=True, device=device)
    mu = torch.tensor(
        [hparams.init_mu for i in range(n_weights)], requires_grad=True, dtype=torch.float32, device=device
    )

    I_rho = np.eye(n_weights, dtype=np.float32)
    I_rho = torch.from_numpy(I_rho) * rho
    I_rho = I_rho.to(device)

    xs = torch.tensor([], device=device)
    xs = torch.cat((xs, x.detach().view(1, -1)))

    # Training loop
    for it in range(n_iterations):
        # solve for current lagrangian multipliers
        for i in range(n_steps):
            obj = lf_rho(x, lmbda, mu, rho, I_rho, loss_fn=loss_fn)
            dx = torch.autograd.grad(obj, x)
            with torch.no_grad():
                x -= step_size * dx[0]

            writer.add_histogram("x", x.detach().clone().cpu().numpy(), it * n_steps + i)

            # Log scalar values for each step
            writer.add_scalars(
                "Function values",
                {"f": loss_fn(x).item(), "Lf": lf(x, lmbda, mu, loss_fn=loss_fn).item(), "Lf_rho": obj.item()},
                it * n_steps + i,
            )

        # Log multipliers and rho for each iteration
        writer.add_histogram("lmbda", lmbda.detach().clone().cpu().numpy(), it)
        writer.add_histogram("mu", mu.detach().clone().cpu().numpy(), it)
        writer.add_histogram("rho", rho.detach().clone().cpu().numpy(), it)

        # Log x
        xs = torch.cat((xs, x.detach().view(1, -1)))

        # Update lagrangian multipliers and rho
        with torch.no_grad():
            lmbda = lmbda + rho * (x.sum() - 1)
            mu = torch.maximum(torch.zeros_like(x), mu + rho * (-x))
        update_identity_rho(x, I_rho, rho, mu, n_weights)

        rho = rho * rho_scaling

        # Assert KKT Conditions
        converged = are_kkt_conditions_verified(x, lmbda, mu, loss_fn, atol=1e-4)

        print(f"Iteration: {it} - objective value lagrangian: {lf(x, lmbda, mu, loss_fn=loss_fn).item():.5f}")
        if converged:
            print("KKT conditions met")
            break

    writer.add_hparams(asdict(hparams), {"loss": obj.item()})
    writer.close()
    return writer.log_dir, xs.cpu().numpy()


hparams_dict = {
    "n_steps": 250,
    "n_iterations": 100,
    "rho": 2.0,
    "rho_scaling": 1.0,
    "step_size": 1e-3,
    "init_lmbda": -0.05,
    "init_mu": 0.5,
}

hparams_dict = {
    "n_steps": 25,
    "n_iterations": 100,
    "rho": 2.0,
    "rho_scaling": 1.0,
    "step_size": 1e-3,
    "init_lmbda": -0.05,
    "init_mu": 0.5,
}

hparams = Hyperparameters(**hparams_dict)
writer = SummaryWriter(log_dir=f"logs/{datetime.now().isoformat()}")


log_dir, xs = train(Y, loss_fn=lambda x: loss_var(x, Y), hparams=hparams, writer=writer)

step_sizes = [1e-2, 5e-3, 1e-3, 5e-4]
rhos = [1.5, 2, 2.5]
rho_scalings = [0.9, 0.99]


for step_size in step_sizes:
    hparams.step_size = step_size
    for rho in rhos:
        hparams.rho = rho
        for rho_scaling in rho_scalings:
            print(f"{hparams_as_str}")
            hparams.rho_scaling = rho_scaling
            hparams_as_str = "|".join([f"{key}={value}" for key, value in asdict(hparams).items()])
            log_dir, xs = train(
                Y,
                loss_fn=lambda x: loss_var(x, Y),
                hparams=hparams,
                writer=SummaryWriter(log_dir=f"logs/{hparams_as_str}/{datetime.now().isoformat()}"),
            )


def get_scalar_data_from_summary_writer(log_dir, scalar="f"):
    summary_dir = log_dir + "/Function values_" + scalar
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


vals = get_scalar_data_from_summary_writer(log_dir, "f")

df = pd.DataFrame(
    {
        "f": get_scalar_data_from_summary_writer(log_dir, "f"),
        "Lf": get_scalar_data_from_summary_writer(log_dir, "Lf"),
        "Lf_rho": get_scalar_data_from_summary_writer(log_dir, "Lf_rho"),
    }
)
df.plot()

px.line(xs)

# %%
weights = []
loss_fns = ["Var", "Cov", "-Mean+Var", "-Mean+Cov"]
for loss_fn in tqdm(loss_fns):
    objs, lmbdas, mus, rhos, xs = train(Y, loss_fn=loss_fn)
    weights.append(xs[-1])

# %%
df_weights = pd.DataFrame(np.around(100 * np.array(weights).T, 2), index=data_loader.df.columns, columns=loss_fns).iloc[
    ::-1
]

print(
    df_weights.style.to_latex(
        hrules=True,
        label="tab:developments-by-loss-fn",
        # escape=False,
        column_format="|" + "|".join(["l" for _ in df_weights.columns]) + "|",
        caption="Overview of how the model would weight the different wind farms given the different loss functions.",
    )
)
