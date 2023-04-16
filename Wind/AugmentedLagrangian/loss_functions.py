import torch


def loss_var(x, Y):
    y = torch.matmul(x, Y)
    return y.var()


def loss_mean_var(x, Y, alp):
    y = torch.matmul(x, Y)
    return -alp * y.mean() + (1 - alp) * y.var()


def loss_mean_cov(x, Y, alp):
    y = x[:, None] * Y
    cov_matrix = y.cov()
    return -alp * y.sum(axis=0).mean() + (1 - alp) * torch.triu(cov_matrix, diagonal=1).sum()


def lf(x, lmbda, mu, loss_fn):
    return loss_fn(x) + lmbda * (x.sum() - 1) + torch.matmul(-x, mu)


def lf_rho(x, lmbda, mu, rho, I_rho, loss_fn):
    return (
        lf(x, lmbda, mu, loss_fn) + rho / 2 * (x.sum() - 1) ** 2 + 1 / 2 * torch.matmul(torch.matmul(-x, I_rho), (-x))
    )
