import numpy as np
import pandas as pd
import plotly.express as px

df_wind_locations = pd.read_csv("data/offshore_wind_locations.csv")
df_nve_wind_locations = pd.read_csv("data/nve_offshore_wind_areas.csv", index_col=0)

df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
df_locations = df_locations.reset_index(drop=True)
df_locations = df_locations.sort_values(by="lat")  # Sort by south to north

area_nor = df_nve_wind_locations["location"].values
areas_all = df_locations["location"].values

areas = areas_all
# areas = area_nor

# Plot locations on map
fig = px.scatter_mapbox(
    df_locations[df_locations["location"].isin(areas)],
    lat="lat",
    lon="lon",
    color="location",
    zoom=3,
    size_max=15,
    height=600,
    size=weights,  # norwegian water resources and energy directorate
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show(config=dict(editable=True))

# Load data
data = []
for l in df_locations["location"].values:
    data.append(pd.read_csv(f"data/{l}.csv", index_col=0, parse_dates=True))

df = pd.concat(data, axis=1)
df = df[df_locations["location"]]  # Sort by south to north

df.head()

vals = df[areas].values
weights = np.zeros((vals.shape[1], 1)) + 1 / vals.shape[1]

n_rands = 1000
rand_weights = np.random.rand(weights.shape[0], n_rands)

rand_weights.shape
rand_weights /= rand_weights.sum(axis=0)  # normalize so sum = 1

rand_weights.shape

y_total = np.matmul(vals, rand_weights)

stds = y_total.std(axis=0)

px.scatter(x=rand_weights[0, :], y=stds)

stds.argmin()
stds.min()

weights = rand_weights[:, stds.argmin()]
