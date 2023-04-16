import sys
import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from Wind.load_data import DataLoader, get_test_dataset
from Wind.AugmentedLagrangian.augmented_lagrangian import Hyperparameters, AugmentedLagrangian
from Wind.AugmentedLagrangian.loss_functions import loss_var, loss_mean_var


# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input-model", nargs="+", type=str, required=True, help="")
parser.add_argument("--alpha", nargs="+", type=float, required=True, help="")
parser.add_argument("--output-table", required=True, help="")
args = parser.parse_args()

weights = []
models = {}
for alp, file in zip(args.alpha, args.input_model):
    with open(file, "rb") as f:
        models[alp] = pickle.load(f)
    weights.append(models[alp].xs[-1].numpy())

index = models[alp].df.columns

df_weights = pd.DataFrame(np.array(weights).T * 100, index=index, columns=args.alpha).iloc[::-1]

Y = models[alp].df.values.T

properties = {}
for alp in args.alpha:
    properties[alp] = {}

    y = np.matmul(models[alp].xs[-1], Y) * 100
    properties[alp]["Mean"] = y.mean().numpy()
    properties[alp]["Variance"] = y.std().numpy()

df_properties = pd.DataFrame(properties, dtype=np.float32)
df_properties.info()

df_table = pd.concat((df_weights, df_properties))

table_output_file = Path(args.output_table)
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
