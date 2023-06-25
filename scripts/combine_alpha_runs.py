import logging
import sys
import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from _helpers import configure_logging

sys.path.insert(1, "/Users/martihj/gitsource/wind-covariation") # Absolute path due to issues with snakemake
from Wind.load_data import DataLoader, get_test_dataset
from Wind.AugmentedLagrangian.augmented_lagrangian import Hyperparameters, AugmentedLagrangian
from Wind.AugmentedLagrangian.loss_functions import loss_var, loss_mean_var

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("combine_agumented_lagrangian_runs", area="norwegian", bias="bias_false", alpha=[0.5, 0.3, 0.1, 0.0])

    configure_logging(snakemake)

    model = snakemake.input.model
    alphas = snakemake.params.alpha_list
    table_output_file = Path(snakemake.output.table)
    bias = snakemake.wildcards.bias

    weights = []
    models = {}
    for alp, file in zip(alphas, model):
        with open(file, "rb") as f:
            models[alp] = pickle.load(f)
        weights.append(models[alp].xs[-1].numpy())

    index = models[alp].df_trainable.columns
    df_trainable_weights = pd.DataFrame(np.array(weights).T * 100, index=index, columns=alphas)
    
    index = models[alp].df_not_trainable.columns
    if bias == "bias_true":
        weights = [[1/len(models[alp].df.columns) for j in index] for i in alphas]
    else:
        weights = [[0.0 for j in index] for i in alphas]
    df_nontrainable_weights = pd.DataFrame(np.array(weights).T * 100, index=index, columns=alphas)

    df = pd.concat([df_trainable_weights, df_nontrainable_weights],axis=0)

    df_weights = pd.concat([df_trainable_weights, df_nontrainable_weights], axis=0)

    Y = models[alp].df_trainable.values.T

    properties = {}
    for alp in alphas:
        properties[alp] = {}

        y = np.matmul(models[alp].xs[-1], Y) * 100 + np.matmul(models[alp].df_not_trainable.values, df_nontrainable_weights[alp].values)
        properties[alp]["Mean"] = y.mean().numpy()
        properties[alp]["Std. dev."] = y.std().numpy()

    df_properties = pd.DataFrame(properties, dtype=np.float32)
    
    df_table = pd.concat((df_weights, df_properties))

    df_table.to_csv(table_output_file)

    # df = pd.DataFrame(
    #     {
    #         "f": model.get_scalar_data_from_summary_writer("f"),
    #         "Lf": model.get_scalar_data_from_summary_writer("Lf"),
    #         "Lf_rho": model.get_scalar_data_from_summary_writer("Lf_rho"),
    #     }
    # )
    # df.plot()

    # df_xs = pd.DataFrame(model.xs, columns=data_loader.df.columns)
    # px.line(df_xs)

    # df_mus = pd.DataFrame(model.mus, columns=data_loader.df.columns)
    # px.line(df_mus)
