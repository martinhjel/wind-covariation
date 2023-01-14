import numpy as np
import pandas as pd
import plotly.express as px


from Wind.analyze import get_corr_figure, get_hours_shift_figure, get_mean_std_wind_figure, get_corr_distance_figure

df_wind_locations = pd.read_csv("data/offshore_wind_locations.csv")
df_nve_wind_locations = pd.read_csv("data/nve_offshore_wind_areas.csv", index_col=0)

df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
df_locations = df_locations.reset_index(drop=True)
df_locations = df_locations.sort_values(by="lat")  # Sort by south to north

# Plot locations on map
fig = px.scatter_mapbox(df_locations, lat="lat", lon="lon", color="location", zoom=3, height=600)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show(config=dict(editable=True))

# Load data
data = []
for l in df_locations["location"].values:
    data.append(pd.read_csv(f"data/{l}.csv", index_col=0, parse_dates=True))

df = pd.concat(data, axis=1)
df = df[df_locations["location"]]  # Sort by south to north

df.info()
df.describe()

# Plot correlation
fig = get_corr_figure(df)
fig.show()

fig = get_corr_distance_figure(df, df_locations)
fig.show()


#
n_shifts = 25
quantile = 0.99

fig = get_hours_shift_figure(df, n_shifts, quantile)
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

x=np.linspace(0,1,len(vals))
px.line(
    x=x,
    y=np.sort(vals)
)

px.line(df["BE"].sample(10000).sort_values().values)

