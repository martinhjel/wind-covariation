import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import torch

torch.cuda.is_available()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")


def update_identity_rho(x, I, rho, mu, n_weights):
    zero = torch.tensor(0.0, dtype=torch.float64)
    for i in range(n_weights):
        if (-x[i]) < zero and torch.isclose(mu[i], zero):
            I[i, i] = 0
        else:
            I[i, i] = rho


def update_dual_variables(lmbda, mu, rho, x):
    with torch.no_grad():
        lmbda = lmbda + rho * (x.sum() - 1)
        mu = torch.maximum(torch.zeros_like(x), mu + rho * (-x))


def are_kkt_conditions_verified(x, lmbda, mu, atol=1e-4):
    # dx L = 0
    dx = torch.autograd.grad(lf(x, lmbda, mu), x)[0]
    if torch.isclose(dx, torch.zeros_like(dx), atol=atol).all():
        # c(x) = 0 | x.sum()-1 = 0
        if torch.isclose((x.sum() - 1), torch.tensor(0.0, dtype=torch.float64), atol=atol):
            # h(x) <= 0 | (-x) <= 0
            if ((-x) <= 0.0).all():
                # mu >= 0
                if (mu >= 0.0).all():
                    # mu*.h(x) = 0
                    if torch.isclose((-x) * mu, torch.zeros_like(mu), atol=atol).all():
                        return True

    return False


def f(x, loss_fn="Var", alp=0.5):
    if loss_fn == "Var":
        y = torch.matmul(x, Y)
        return y.var()
    elif loss_fn == "Cov":
        y = x[:, None] * Y
        cov_matrix = y.cov()
        return torch.triu(cov_matrix, diagonal=1).sum()
    elif loss_fn == "-Mean+Var":
        y = x[:, None] * Y
        return -alp * y.mean() + (1 - alp) * y.var()
    elif loss_fn == "-Mean+Cov":
        y = x[:, None] * Y
        cov_matrix = y.cov()
        return -alp * y.mean() + (1 - alp) * torch.triu(cov_matrix, diagonal=1).sum()


def lf(x, lmbda, mu, loss_fn="Var"):
    return f(x, loss_fn) + lmbda * (x.sum() - 1) + torch.matmul(-x, mu)


def lf_rho(x, lmbda, mu, rho, I_rho, loss_fn="Var"):
    return lf(x, lmbda, mu, loss_fn) + rho / 2 * (x.sum() - 1) ** 2 + 1 / 2 * torch.matmul(torch.matmul(-x, I_rho), (-x))


### Generate test dataset

N = 1000
s = np.linspace(0, 10, N)
y1 = np.sin(s)
y2 = np.cos(s)
plt.plot(y1)
plt.plot(y2)

Y = np.stack((y1, y2), axis=-1).T

Y = torch.from_numpy(Y)
Y = Y.to(device)

### Use Wind dataset
from pathlib import Path
from Wind.load_data import DataLoaderFileShare

data_loader = DataLoaderFileShare(Path("config.ini"))

Y = data_loader.df.values.T
Y = torch.from_numpy(Y)
Y = Y.to(device)

# %%%


def train(Y, loss_fn):
    n_weights = Y.shape[0]
    x = np.ones(n_weights)
    x = x / x.sum(axis=-1)
    x = torch.from_numpy(x)
    x.requires_grad = True
    x = x.to(device)

    # Hyperparamters
    rho = torch.tensor(2.0, dtype=torch.float64)  # Testdata = 1.1
    rho_scaling = torch.tensor(1.0, dtype=torch.float64)  # Testdata= 1.1
    step_size = torch.tensor(1e-3, dtype=torch.float64)
    n_steps = 250
    n_iterations = 100

    # Langrange multipliers
    lmbda = torch.tensor(-0.05, requires_grad=True, device=device)
    mu = torch.tensor([0.5 for i in range(n_weights)], requires_grad=True, dtype=torch.float64, device=device)

    I_rho = np.eye(Y.shape[0])
    I_rho = torch.from_numpy(I_rho) * rho
    I_rho = I_rho.to(device)

    objs = []
    xs = [x.cpu().detach().numpy().copy()]
    lmbdas = [lmbda.item()]
    mus = [mu.cpu().detach().numpy().copy()]
    rhos = [rho.item()]

    # Training loop
    for it in range(n_iterations):
        # solve for current lagrangian multipliers
        for i in range(n_steps):
            obj = lf_rho(x, lmbda, mu, rho, I_rho, loss_fn=loss_fn)
            dx = torch.autograd.grad(obj, x)
            with torch.no_grad():
                x -= step_size * dx[0]
            xs.append(x.cpu().detach().numpy().copy())
            objs.append([obj.item(), lf(x, lmbda, mu, loss_fn=loss_fn).item(), f(x, loss_fn=loss_fn).item()])
        objs.append(
            [
                lf_rho(x, lmbda, mu, rho, I_rho,loss_fn=loss_fn).item(),
                lf(x, lmbda, mu, loss_fn=loss_fn).item(),
                f(x, loss_fn=loss_fn).item(),
            ]
        )

        mus.append(mu.cpu().detach().numpy().copy())
        lmbdas.append(lmbda.item())
        # Update lagrangian multipliers and rho
        with torch.no_grad():
            lmbda = lmbda + rho * (x.sum() - 1)
            mu = torch.maximum(torch.zeros_like(x), mu + rho * (-x))
        update_identity_rho(x, I_rho, rho, mu, n_weights)

        rho = rho * rho_scaling
        rhos.append(rho.item())

        # Assert KKT Conditions
        converged = are_kkt_conditions_verified(x, lmbda, mu, atol=1e-4)

        if converged:
            print("KKT conditions met")
            break

    return objs, lmbdas, mus, rhos, xs


objs, lmbdas, mus, rhos, xs = train(Y, loss_fn="Var")

data = [
    go.Scatter(y=[i[0] for i in objs], name="lf_rho"),
    go.Scatter(y=[i[1] for i in objs], name="lf"),
    go.Scatter(y=[i[2] for i in objs], name="f"),
]
go.Figure(data=data).show()
px.line(xs)
px.line(rhos)

px.line(mus)
px.line(lmbdas)


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
