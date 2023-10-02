import time
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from Wind.plotly_template import my_template

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

    df.head()

    # %% DTW
    

    from tslearn.metrics import dtw

    distance = dtw(df.iloc[:10000,0], df.iloc[:10000,1])

    dtw(df.iloc[:10000,0], df.iloc[:10000,-1])
# %% PCA

    from sklearn.decomposition import PCA

    pca = PCA()

    pca.fit(df)

    fig = px.bar(pca.explained_variance_ratio_*100)

    np.sum(pca.explained_variance_ratio_[:2])

    fig.update_layout(
        template=my_template,
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Ratio [%]",
    )
    
    fig.write_image("images/pca.pdf")


# %% Tests

    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    from factor_analyzer.factor_analyzer import calculate_kmo

    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    kmo_all, kmo_model = calculate_kmo(df)

    print("Bartlett's Test p-value:", p_value)
    print("KMO Test:", kmo_model)

