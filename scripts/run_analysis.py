import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

sys.path.insert(1, os.path.join(sys.path[0], ".."))
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
)


# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input-locations", type=str, required=True, help="")
parser.add_argument("--input-nve-locations", type=str, required=True, help="")
parser.add_argument("--wind-data", required=True, help="")
args = parser.parse_args()

wind_locations_file = Path(args.input_locations)
wind_nve_locations_file = Path(args.input_nve_locations)
wind_data_file = Path(args.wind_data)

### Read data
df_wind_locations = pd.read_csv(wind_locations_file)
df_nve_wind_locations = pd.read_csv(wind_nve_locations_file, index_col=0)
df_nve_wind_locations = df_nve_wind_locations.sort_values(by="lat")  # Sort by south to north

df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
df_locations = df_locations.reset_index(drop=True)
df_locations = df_locations.sort_values(by="lat")  # Sort by south to north

df = pd.read_csv(wind_data_file, index_col=0)
df.index = pd.to_datetime(df.index)

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
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show(config=dict(editable=True))


df_mean_wind_nve = df[df_nve_wind_locations["location"]].mean()
# df_mean_wind_nve.to_csv("data/mean_wind.csv")
fig = px.bar(df_mean_wind_nve, text_auto=".2", title="")
fig.update_traces(textfont_size=12, textangle=0, textposition="inside", cliponaxis=False)
fig.update_layout(
    xaxis_title="Wind farm", yaxis_title="Mean wind power output", showlegend=False, template="plotly_white", width=1000
)
fig.show()

# Plot correlation
fig = get_corr_figure(df)
fig.show()

fig = get_corr_distance_figure(df, df_locations)
fig.show()


# Plot short-term variation
n_shifts = 25
quantile = 0.9

fig = get_hours_shift_figure(df, df_nve_wind_locations, n_shifts, quantile)
fig.update_layout(width=900)
fig.show()

fig = get_hours_shift_figure(df, df_nve_wind_locations, n_shifts, quantile=0.9999)
fig.update_layout(width=900)
fig.show()

fig = px.area(df.resample("1D").mean())
fig.show()

resample_period = "7D"
fig = get_mean_std_wind_figure(df, resample_period)
fig.show()

##

df["BE"].sort_values().plot()

vals = df["BE"].values

x = np.linspace(0, 1, len(vals))
px.line(x=x, y=np.sort(vals))

px.line(df["BE"].sample(10000).sort_values().values)


### Line plots


area = "Utsira nord"
resample_period = "7D"
fig = get_line_plot_with_mean(df, area, resample_period)
fig.show()

resample_period = "1H"
fig = get_line_plot_with_mean(df, area, resample_period)
fig.show()


# resample_period = "1H"
# fig = get_mean_std_wind_yearly_figure(df, resample_period)

## Scatter plots
df.columns
area_a = "Sørlige Nordsjø II"  # "Utsira nord"
area_b = "DE West"  # "Auvær"

fig = get_scatter_2d_figure(df.sample(10000), area_a, area_b)
fig.show()


fig = get_histogram_2d_figure(df, area_a, area_b)
fig.show()

## Kernel density plots
N = 50
fig = get_scatter_with_kernel_density_2d_figure(
    df, area_a, area_b, N, n_scatter_samples=500, bandwidth=0.1, rtol=0.01, kernel="epanechnikov"
)
fig.show()

fig = get_scatter_with_kernel_density_2d_figure(
    df, area_a, area_b, N, n_scatter_samples=500, bandwidth=0.1, rtol=0.01, kernel="gaussian"
)
fig.show()

area_a = "Sørlige Nordsjø II"
area_b = "Nordmela"  # "Utsira nord"
fig = get_scatter_with_kernel_density_2d_figure(
    df, area_a, area_b, N, n_scatter_samples=500, bandwidth=0.1, rtol=0.01, kernel="gaussian"
)
fig.show()

# fig1 = get_scatter_density_2d_figure(df.sample(10000), area_a, area_b)
# fig1.show()


# %%

df.head()
import plotly.graph_objects as go

df_diff = df.diff(periods=1)

data = []
for col in ["BE", "Sørlige Nordsjø II", "Sum"]:
    bins = np.arange(-0.7, 0.7, 0.001)
    hist, bin_edges = np.histogram(df_diff[col], bins=bins, density=True)
    data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name=col))

layout = go.Layout(
    title_text="", xaxis_title_text="Hourly changes of wind outputs", yaxis_title_text="Probability", barmode="overlay"
)

fig = go.Figure(data=data, layout=layout)
fig.show()

# %%

df_diff = df.resample("1D").mean().diff(periods=1)

data = []
for col in ["BE", "Sørlige Nordsjø II", "Sum"]:
    bins = np.arange(-0.7, 0.7, 0.008)
    hist, bin_edges = np.histogram(df_diff[col], bins=bins, density=True)
    data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name=col))

layout = go.Layout(
    title_text="", xaxis_title_text="Daily changes of wind outputs", yaxis_title_text="Probability", barmode="overlay"
)

fig = go.Figure(data=data, layout=layout)
fig.show()

# %%
df_diff = df.resample("7D").mean().diff(periods=1)

data = []
for col in ["BE", "Sørlige Nordsjø II", "Sum"]:
    bins = np.arange(-0.7, 0.7, 0.01)
    hist, bin_edges = np.histogram(df_diff[col], bins=bins, density=True)
    data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name=col))

layout = go.Layout(
    title_text="", xaxis_title_text="Weekly changes of wind outputs", yaxis_title_text="Probability", barmode="overlay"
)

fig = go.Figure(data=data, layout=layout)
fig.show()
# %%

from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go

data = []
col = "BE"
diff = df[col].resample("1H").mean().diff(periods=1)
bins = np.arange(-0.7, 0.7, 0.001)
hist, bin_edges = np.histogram(diff, bins=bins, density=True)
data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="hourly"))
kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(diff.dropna().values[:, np.newaxis])
data.append(go.Line(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis])), name="hourly"))

diff = df[col].resample("1H").mean().diff(periods=2)
bins = np.arange(-0.7, 0.7, 0.001)
hist, bin_edges = np.histogram(diff, bins=bins, density=True)
data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="bihourly"))
kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(diff.dropna().values[:, np.newaxis])
data.append(go.Line(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis])), name="bihourly"))


diff = df[col].resample("1D").mean().diff(periods=1)
bins = np.arange(-0.7, 0.7, 0.008)
hist, bin_edges = np.histogram(diff, bins=bins, density=True)
data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="daily"))
kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(diff.dropna().values[:, np.newaxis])
data.append(go.Line(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis]))))


diff = df[col].resample("7D").mean().diff(periods=1)
bins = np.arange(-0.7, 0.7, 0.01)
hist, bin_edges = np.histogram(diff, bins=bins, density=True)
data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="weekly"))
kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(diff.dropna().values[:, np.newaxis])
data.append(go.Line(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis]))))


diff = df[col].resample("1M").mean().diff(periods=1)
bins = np.arange(-0.7, 0.7, 0.02)
hist, bin_edges = np.histogram(diff, bins=bins, density=True)
data.append(go.Bar(x=bin_edges, y=hist, opacity=0.8, name="monthly"))
kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(diff.dropna().values[:, np.newaxis])
data.append(go.Line(x=bins, y=np.exp(kde.score_samples(bins[:, np.newaxis]))))


layout = go.Layout(
    title_text="", xaxis_title_text="Weekly changes of wind outputs", yaxis_title_text="Probability", barmode="overlay"
)

fig = go.Figure(data=data, layout=layout)
fig.show()

# %%
