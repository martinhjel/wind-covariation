import sys
import pickle

import pandas as pd
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

import logging
from _helpers import configure_logging

sys.path.insert(1, "/Users/martihj/gitsource/wind-covariation")  # Absolute path due to issues with snakemake
from Wind.plotly_template import my_template

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pd.set_option("plotting.backend", "plotly")
show_figure = True


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_weights", area="all", bias="bias_false", alpha="0.5")

    configure_logging(snakemake)

    model_file = snakemake.input.model
    
    alpha = snakemake.wildcards.alpha
   
    with open(model_file, "rb") as f:
            model = pickle.load(f)
    df_weights = pd.DataFrame(model.xs.numpy(), columns=model.df_trainable.columns).iloc[::-1]

    n_cols = len(df_weights.columns)
    colormap = sns.color_palette("Set3", n_cols).as_hex()
    colors = {col: color for col, color in zip(df_weights.columns, colormap)}

    fig = df_weights.plot(
        title=f"With alpha={alpha}",
        template=my_template,
        labels=dict(value="Weight", index="Step"),
        color_discrete_map=colors,
    )
    fig.update_layout(
        xaxis=dict(range=[0, 30]),
        legend=dict(
            x=1,  # 1 is the far right, so the legend is placed just right of the plot
            y=1,  # 1 is the top, so the legend's top is aligned with the plot's top
            xanchor="left",  # The left side of the legend is at position `x`
            yanchor="top"    # The top of the legend is at position `y`
        )
    )
    if show_figure:
        fig.show()
    fig.write_image(f"images/weights_training/{snakemake.wildcards.area}/{snakemake.wildcards.bias}/{alpha}_alpha.pdf")

