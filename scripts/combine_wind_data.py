import argparse
from pathlib import Path
import pandas as pd


# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input-locations", type=str, required=True, help="")
parser.add_argument("--input-nve-locations", type=str, required=True, help="")
parser.add_argument("--output-combined", required=True, help="")
args = parser.parse_args()

wind_locations_file = Path(args.input_locations)
wind_nve_locations_file = Path(args.input_nve_locations)
output_combined_file = Path(args.output_combined)

### Load from local store
df_wind_locations = pd.read_csv(wind_locations_file)
df_nve_wind_locations = pd.read_csv(wind_nve_locations_file, index_col=0)
df_nve_wind_locations = df_nve_wind_locations.sort_values(by="lat")  # Sort by south to north

df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
df_locations = df_locations.reset_index(drop=True)
df_locations = df_locations.sort_values(by="lat")  # Sort by south to north

# Load data
data = []
for l in df_locations["location"].values:
    data.append(pd.read_csv(f"data/{l}.csv", index_col=0, parse_dates=True))

df = pd.concat(data, axis=1)
df = df[df_locations["location"]]  # Sort by south to north


df.to_csv(output_combined_file)
