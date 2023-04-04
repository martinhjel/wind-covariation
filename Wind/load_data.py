from configparser import ConfigParser
from typing import Optional
from pathlib import Path
from getpass import getpass
from io import StringIO
from azure.storage.fileshare import ShareServiceClient, ShareDirectoryClient
from configparser import RawConfigParser
import pandas as pd


class DataLoader:
    """
    DataLoader loads offshore wind energy data from local or Azure file share storage.

    Args:
        data_folder_path (str, optional): Local path to data folder. Defaults to None, in which case
        data is loaded from Azure file storage, read from the 'config.ini' file.

    Attributes:
        df (pandas.DataFrame): Concatenated dataframe of offshore wind energy data.
        df_locations(pandas.DataFrame): Locations' dataframe with columns: latitude, longitude, and location name.
        df_nve_wind_locations (pandas.DataFrame): NVE's offshore wind area dataframe with columns: latitude, longitude, and location ids.

    """

    def __init__(self, data_folder_path: Optional[Path] = None):
        if data_folder_path:
            self._load_locally(data_folder_path=data_folder_path)
        else:
            self._load_from_file_share()

    def _load_from_file_share(self):
        try:
            config = RawConfigParser()
            config.read("config.ini")
            sas_token_url = config["File Storage"]["sas_token"]
        except:
            # url to the root file share folder ("data")
            sas_token_url = getpass("sas taken and url: ")

        dir_client = ShareDirectoryClient(
            account_url=sas_token_url, directory_path="data", share_name="wind-covariation"
        )
        file_client = dir_client.get_file_client("offshore_wind_locations.csv")

        dir_client = ShareDirectoryClient(
            account_url=sas_token_url, directory_path="data/nve/profiler/Wind and solar", share_name="wind-covariation"
        )

        dir_client = ShareDirectoryClient(
            account_url=sas_token_url, directory_path="data", share_name="wind-covariation"
        )

        file_client = dir_client.get_file_client("offshore_wind_locations.csv")
        df_wind_locations = pd.read_csv(StringIO(file_client.download_file().content_as_text()))

        file_client = dir_client.get_file_client("nve_offshore_wind_areas.csv")
        df_nve_wind_locations = pd.read_csv(StringIO(file_client.download_file().content_as_text()))

        df_nve_wind_locations = df_nve_wind_locations.sort_values(by="lat")  # Sort by south to north

        df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
        df_locations = df_locations.reset_index(drop=True)
        df_locations = df_locations.sort_values(by="lat")  # Sort by south to north

        # Load data
        data = []
        for l in df_locations["location"].values:
            file_client = dir_client.get_file_client(f"{l}.csv")
            df_temp = pd.read_csv(
                StringIO(file_client.download_file().content_as_text()), index_col=0, parse_dates=True
            )
            data.append(df_temp)

        df = pd.concat(data, axis=1)
        df = df[df_locations["location"]]  # Sort by south to north

        self.df = df
        self.df_locations = df_locations
        self.df_nve_wind_locations = df_nve_wind_locations

    def _load_locally(self, data_folder_path: Path):
        df_wind_locations = pd.read_csv(data_folder_path / "offshore_wind_locations.csv")
        df_nve_wind_locations = pd.read_csv(data_folder_path / "nve_offshore_wind_areas.csv", index_col=0)
        df_nve_wind_locations = df_nve_wind_locations.sort_values(by="lat")  # Sort by south to north

        df_locations = pd.concat([df_wind_locations, df_nve_wind_locations], axis=0)
        df_locations = df_locations.reset_index(drop=True)
        df_locations = df_locations.sort_values(by="lat")  # Sort by south to north

        # Load data
        data = []
        for l in df_locations["location"].values:
            data.append(pd.read_csv(data_folder_path / f"{l}.csv", index_col=0, parse_dates=True))

        df = pd.concat(data, axis=1)
        df = df[df_locations["location"]]  # Sort by south to north

        self.df = df
        self.df_locations = df_locations
        self.df_nve_wind_locations = df_nve_wind_locations
