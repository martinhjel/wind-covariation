import time
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

sys.path.insert(1, "/Users/martihj/gitsource/wind-covariation")  # Absolute path due to issues with snakemake

from scripts._helpers import configure_logging

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("run_analysis")

    configure_logging(snakemake)
    # %%
    logger.info("Reading snakemake input")
    wind_locations_file = Path(snakemake.input.locations)
    wind_nve_locations_file = Path(snakemake.input.nve_locations)
    wind_data_file = Path(snakemake.input.combined)
    models = [Path(i) for i in snakemake.input.models]

    # %% [markdown]
    #
    # # Read data

    outputs = snakemake.output

    # %%
    df_wind_locations = pd.read_csv(wind_locations_file)
    df_nve_wind_locations = pd.read_csv(wind_nve_locations_file, index_col=0)
    df_nve_wind_locations = df_nve_wind_locations.sort_values(by="lat")  # Sort by south to north

    df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
    df_locations = df_locations.reset_index(drop=True)
    df_locations = df_locations.sort_values(by="lat")  # Sort by south to north

    df = pd.read_csv(wind_data_file, index_col=0)
    df.index = pd.to_datetime(df.index)

    # %%
    df.head()

    df.columns

    nor_location = df_nve_wind_locations["location"].to_list()

    df_nor = df[nor_location[::-1]]
    
    sintef_weights = np.array([
        3.0204,
        1.7831,
        1.2009,
        3.8210,
        2.8384,
        2.2562,
        8.9520,
        1.6376,
        9.4614,
        6.0044,
        0.8734,
        0.6550,
        11.6812,
        15.8661,
        29.9491])
    
    data = df_nor.values
    data.shape
    sintef_weights.shape
    weighted = data*sintef_weights

    print(f" mean: {weighted.sum(axis=1).mean()}")
    print(f"std. dev: {weighted.sum(axis=1).std()}")
    

