import time
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

sys.path.insert(1, "/Users/martihj/gitsource/wind-covariation")  # Absolute path due to issues with snakemake
from Wind.analyze import (
    get_corr_figure,
    get_hours_shift_figure,
    get_mean_std_wind_figure,
    get_corr_distance_figure,
    get_line_plot_with_mean,
    get_histogram_2d_figure,
    get_scatter_2d_figure,
    get_scatter_with_kernel_density_2d_figure,
    get_scatter_density_2d_figure,
    get_multiple_corr_distance_figure,
)
from _helpers import configure_logging

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

show_figure = False

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

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

    # %%
    df_locations.head()

    # %%
    df_nve_wind_locations.head()

    # %%
    # Plot locations on map
    fig = px.scatter_mapbox(
        df_locations,
        lat="lat",
        lon="lon",
        color="location",
        zoom=3,
        size_max=10,
        height=600,
        size=[3 for _ in df_locations.iterrows()],
    )
    fig.update_layout(
        mapbox_style="open-street-map",
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    if show_figure:
        fig.show(config=dict(editable=True))

    # %%

    df_mean_wind_nve = df[df_nve_wind_locations["location"]].mean()
    # df_mean_wind_nve.to_csv("data/mean_wind.csv")
    fig = px.bar(df_mean_wind_nve, text_auto=".2", title="")
    fig.update_traces(textfont_size=12, textangle=0, textposition="inside", cliponaxis=False)
    fig.update_layout(
        xaxis_title="Wind farm",
        yaxis_title="Mean wind power output",
        showlegend=False,
        template="plotly_white",
        width=1000,
    )
    if show_figure:
        fig.show()

    # %%

    # Plot correlation
    fig = get_corr_figure(df.resample("1D").mean())
    if show_figure:
        fig.show()
    fig.write_image("images/corr-matrix-1day.pdf")

    # %%
    froya_lat = df_nve_wind_locations[df_nve_wind_locations["location"] == "Frøyabanken"]["lat"].values[0]

    cols_south = df_nve_wind_locations[df_nve_wind_locations["lat"] < froya_lat]["location"].to_list()
    cols_north = df_nve_wind_locations[df_nve_wind_locations["lat"] >= froya_lat]["location"].to_list()
    nl_cols = [i for i in df.columns if "NL" in i]
    uk_cols = [i for i in df.columns if "UK" in i]
    de_cols = [i for i in df.columns if "DE" in i]
    dk_cols = [i for i in df.columns if "DK" in i]

    df_agg = pd.DataFrame(
        {
            "Farms north of Stadt": df[cols_north].mean(axis=1),
            "All 15 wind farms": df.mean(axis=1),
            "Farms south of Stadt": df[cols_south].mean(axis=1),
            "DK": df[dk_cols].mean(axis=1),
            "UK": df[uk_cols].mean(axis=1),
            "DE": df[de_cols].mean(axis=1),
            "NL": df[nl_cols].mean(axis=1),
        },
        index=df.index,
    )

    fig = get_corr_figure(df_agg.resample("1D").mean(), scale_size=1 / 2.5)
    if show_figure:
        fig.show()
    fig.write_image("images/corr-matrix-1day-aggregated.pdf")

    # %%
    resolutions = ["1H", "1D", "7D", "30D"]
    colors = ["red", "black", "green", "blue"]
    fig = get_multiple_corr_distance_figure(df, df_locations, resolutions=resolutions, colors=colors)
    if show_figure:
        fig.show()
    fig.write_image("images/corr-distance.pdf")

    # %%
    fig = get_corr_distance_figure(df, df_locations)
    if show_figure:
        fig.show()

    # %%

    # Plot short-term variation
    n_shifts = 25
    quantile = 0.9

    fig = get_hours_shift_figure(df, df_nve_wind_locations, n_shifts, quantile)
    fig.update_layout(width=900)
    if show_figure:
        fig.show()
    fig.write_image("images/shift-quantile0.9.pdf")

    # %%

    fig = get_hours_shift_figure(df, df_nve_wind_locations, n_shifts, quantile=0.9999)
    fig.update_layout(width=900)
    if show_figure:
        fig.show()
    fig.write_image("images/shift-quantile0.9999.pdf")

    # %%

    # fig = px.area(df.resample("1D").mean())
    # if show_figure:
        # fig.show()

    # %%
    resample_period = "7D"
    fig = get_mean_std_wind_figure(df, resample_period)
    if show_figure:
        fig.show()

    # %%
    df["BE"].sort_values().plot()
    vals = df["BE"].values
    x = np.linspace(0, 1, len(vals))
    px.line(x=x, y=np.sort(vals))

    # %%

    px.line(df["BE"].sample(10000).sort_values().values)

    # %%

    ### Line plots
    area = "Utsira nord"
    resample_period = "7D"
    fig = get_line_plot_with_mean(df, area, resample_period)
    if show_figure:
        fig.show()
    fig.write_image("images/utsira-nord-std-wind-7D.pdf")

    # %%
    resample_period = "1H"
    fig = get_line_plot_with_mean(df, area, resample_period)
    if show_figure:
        fig.show()
    fig.write_image("images/utsira-nord-std-wind-1H.pdf")

    # %%
    # resample_period = "1H"
    # fig = get_mean_std_wind_yearly_figure(df, resample_period)

    # %%
    ## Scatter plots
    df.columns
    area_a = "Sørlige Nordsjø II"  # "Utsira nord"
    area_b = "DE West"  # "Auvær"

    fig = get_scatter_2d_figure(df.sample(10000), area_a, area_b)
    if show_figure:
        fig.show()

    # %%

    fig = get_histogram_2d_figure(df, area_a, area_b)
    if show_figure:
        fig.show()

    # %%

    ## Kernel density plots
    N = 50
    fig = get_scatter_with_kernel_density_2d_figure(
        df, area_a, area_b, N, z_max=4, n_scatter_samples=500, bandwidth=0.1, rtol=0.01, kernel="epanechnikov"
    )
    if show_figure:
        fig.show()

    # %%

    fig = get_scatter_with_kernel_density_2d_figure(
        df, area_a, area_b, N, z_max=3, n_scatter_samples=0, bandwidth=0.1, rtol=0.01, kernel="gaussian"
    )
    if show_figure:
        fig.show()
    fig.write_image("images/scatter-soerlige-nordsjoe-ii-de-west.pdf")

    # %%
    area_a = "Sørlige Nordsjø II"
    area_b = "Nordmela"  # "Utsira nord"
    fig = get_scatter_with_kernel_density_2d_figure(
        df, area_a, area_b, N, z_max=3, n_scatter_samples=0, bandwidth=0.1, rtol=0.01, kernel="gaussian"
    )
    if show_figure:
        fig.show()
    fig.write_image("images/scatter-soerlige-nordsjoe-ii-nordmela.pdf")

    # %%

    # fig1 = get_scatter_density_2d_figure(df.sample(10000), area_a, area_b)
    # fig1.show()

    # %%
    # %%
    import plotly.graph_objects as go

    df["Sum"] = df.mean(axis=1)
    df_diff = df.diff(periods=1)

    data = []
    for col in ["BE", "Sørlige Nordsjø II", "Sum"]:
        bins = np.arange(-0.7, 0.7, 0.001)
        hist, bin_edges = np.histogram(df_diff[col], bins=bins, density=True)
        data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name=col))

    layout = go.Layout(
        title_text="",
        xaxis_title_text="Hourly changes of wind outputs",
        yaxis_title_text="Probability",
        barmode="overlay",
    )

    fig = go.Figure(data=data, layout=layout)
    if show_figure:
        fig.show()

    # %%
    df_diff = df.resample("1D").mean().diff(periods=1)

    data = []
    for col in ["BE", "Sørlige Nordsjø II", "Sum"]:
        bins = np.arange(-0.7, 0.7, 0.008)
        hist, bin_edges = np.histogram(df_diff[col], bins=bins, density=True)
        data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name=col))

    layout = go.Layout(
        title_text="",
        xaxis_title_text="Daily changes of wind outputs",
        yaxis_title_text="Probability",
        barmode="overlay",
    )

    fig = go.Figure(data=data, layout=layout)
    if show_figure:
        fig.show()

    # %%

    # %%
    df_diff = df.resample("7D").mean().diff(periods=1)

    data = []
    for col in ["BE", "Sørlige Nordsjø II", "Sum"]:
        bins = np.arange(-0.7, 0.7, 0.01)
        hist, bin_edges = np.histogram(df_diff[col], bins=bins, density=True)
        data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name=col))

    layout = go.Layout(
        title_text="",
        xaxis_title_text="Weekly changes of wind outputs",
        yaxis_title_text="Probability",
        barmode="overlay",
    )

    fig = go.Figure(data=data, layout=layout)
    if show_figure:
        fig.show()
    # %%

    # %%

    from sklearn.neighbors import KernelDensity
    import plotly.graph_objects as go

    data = []
    col = "Sørlige Nordsjø II"
    diff = df[col].resample("1H").mean().diff(periods=1)
    bins = np.arange(-0.7, 0.7, 0.001)
    hist, bin_edges = np.histogram(diff, bins=bins, density=True)
    #data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="hourly"))
    kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(diff.dropna().values[:, np.newaxis])
    data.append(go.Scatter(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis])), name="hourly", mode="lines"))

    diff = df[col].resample("2H").mean().diff(periods=1)
    bins = np.arange(-0.7, 0.7, 0.001)
    hist, bin_edges = np.histogram(diff, bins=bins, density=True)
    #data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="bihourly"))
    kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(diff.dropna().values[:, np.newaxis])
    data.append(go.Scatter(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis])), name="bihourly", mode="lines"))

    diff = df[col].resample("1D").mean().diff(periods=1)
    bins = np.arange(-0.7, 0.7, 0.008)
    hist, bin_edges = np.histogram(diff, bins=bins, density=True)
    #data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="daily"))
    kde = KernelDensity(kernel="gaussian", bandwidth=0.03).fit(diff.dropna().values[:, np.newaxis])
    data.append(go.Scatter(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis])), name="daily", mode="lines"))

    diff = df[col].resample("7D").mean().diff(periods=1)
    bins = np.arange(-0.7, 0.7, 0.01)
    hist, bin_edges = np.histogram(diff, bins=bins, density=True)
    #data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="weekly"))
    kde = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(diff.dropna().values[:, np.newaxis])
    data.append(go.Scatter(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis])), name="weekly", mode="lines"))

    diff = df[col].resample("1M").mean().diff(periods=1)
    bins = np.arange(-0.7, 0.7, 0.02)
    hist, bin_edges = np.histogram(diff, bins=bins, density=True)
    #data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="monthly"))
    kde = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(diff.dropna().values[:, np.newaxis])
    data.append(go.Scatter(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis])), name="monthly", mode="lines"))
    # %%

    # %%
    from Wind.plotly_template import my_template

    layout = go.Layout(
        title_text="",
        xaxis_title_text="Changes in wind outputs",
        yaxis_title_text="Probability density function",
        barmode="overlay",
        template=my_template,
        width=600,
        height=400,
        legend=dict(x=0.8, y=0.98),
    )

    # for i, trace in enumerate(data):
    #     if i // 2 or i == 0:
    #         trace.visible = "legendonly"

    fig = go.Figure(data=data, layout=layout)
    if show_figure:
        fig.show()
    fig.write_image("images/deviation-soerlige-nordsjoe-ii.pdf")
    logger.info("writing: images/deviation-soerlige-nordsjoe-ii.pdf")
  