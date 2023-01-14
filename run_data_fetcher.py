from datetime import date, datetime, timedelta
import pandas as pd
from configparser import ConfigParser
from Wind.rninja_client import NinjaClient
from shapely.geometry import Point

config = ConfigParser()
config.read("config.ini")
web_token = config["Renewables Ninja"]["token"]

ninja_client = NinjaClient(web_token)

df_wind_locations = pd.read_csv("data/offshore_wind_locations.csv")
df_nve_wind_locations = pd.read_csv("data/nve_offshore_wind_areas.csv", index_col=0)

df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
df_locations = df_locations.reset_index(drop=True)

for i in df_locations.index:
    print(i)
    lat, lon = df_locations["lat"][i], df_locations["lon"][i]
    df_wind, meta = ninja_client.get_wind_dataframe(lat=lat, long=lon, date_from="2000-01-01", date_to="2019-12-31")
    df_wind = df_wind.rename(columns={"electricity": df_locations["location"][i]})
    df_wind.to_csv(f"data/{df_locations['location'][i]}.csv")
