import pandas as pd
import plotly.express as px
from arcgis.gis import GIS
from arcgis.features import FeatureLayer, GeoAccessor, GeoSeriesAccessor
from arcgis.geometry import SpatialReference

gis = GIS()

nve_havvind_uri = "https://nve.geodataonline.no/arcgis/rest/services/Havvind/MapServer/0"

dc_fl = FeatureLayer(nve_havvind_uri)
dc_df = GeoAccessor.from_layer(dc_fl)
dc_df.head()

dc_df["latlon"] = dc_df["SHAPE"].apply(lambda x: x.project_as(spatial_reference=SpatialReference(wkid=4326)))

dc_df["latlon_centroid"] = dc_df["latlon"].apply(lambda x: x.centroid)

ll_list = []
for i in range(len(dc_df)):
    js = json.loads(dc_df["latlon"][i].JSON)
    latlon_list = [p for mp in js["rings"] for p in mp]
    latlon_list.append(list(dc_df["latlon_centroid"][i]))
    df = pd.DataFrame(latlon_list)
    df["name"] = dc_df["NAVN"][i]
    ll_list.append(df)
dff = pd.concat(ll_list, axis=0, ignore_index=True)
dff = dff.rename(columns={0: "lon", 1: "lat"})


fig = px.scatter_mapbox(dff, lat="lat", lon="lon", color="name", zoom=3, height=600)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show(config=dict(editable=True))
