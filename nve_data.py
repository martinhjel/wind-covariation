import pandas as pd

df_overview = pd.read_excel("data/nve/oversikt-over-data.xlsx", sheet_name="Oversikt over alle profiler")

df_overview

files = [i for i in (Path.cwd() / "data/Profiler").glob("*/*")]

files_NO2 = [i.name for i in files if "offshore_new" in str(i)]

ts = {
    "NO2_wind_offshore_SorligeNordsjo2.csv": "NO2_SorligeNordsjo2",
    "NO2_wind_offshore_UtsiraNord.csv": "NO2_UtsiraNord",
    "BEL_wind_offshore_new.csv": "BEL",
    "DEU_wind_offshore_new.csv": "DEU",
    "DK1_wind_offshore_new.csv": "DK1",
    "DK2_wind_offshore_new.csv": "DK2",
    "GBR_wind_offshore_new.csv": "GBR",
    "NLD_wind_offshore_new.csv": "NLD",
    "SE2_wind_offshore_new.csv": "SE2",
    "SE3_wind_offshore_new.csv": "SE3",
    "SE4_wind_offshore_new.csv": "SE4",
    "EST_wind_offshore_new.csv": "EST",
    "FIN_wind_offshore_new.csv": "FIN",
    "LTU_wind_offshore_new.csv": "LTU",
    "LVA_wind_offshore_new.csv": "LVA",
}

files_offshore = [i for i in files if str(i.name) in ts.keys()]
