import json
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests


class NinjaClient:
    BASE_URI = "https://www.renewables.ninja/api/"
    PV_URI = BASE_URI + "data/pv"
    WIND_URI = BASE_URI + "data/wind"
    COUNTRIES_URI = BASE_URI + "countries"
    LIMITS_URI = BASE_URI + "limits"

    def __init__(self, web_token: str):
        self.headers = {"Authorization": f"Token {web_token}"}

        self.last_query_time = pd.Timestamp("2020-01-01T00:00:00")
        self.burst_time_limit, self.max_queries_per_hour = self._compute_limits()

    def _compute_limits(self):
        limits = self.get_limits()
        freq, time = limits["burst"].split("/")
        burst_time = pd.Timedelta(f"{1/int(freq)} {time}")

        max_per_hour = int(limits["sustained"].split("/")[0])

        return burst_time, max_per_hour

    def wait_for_burst(self):
        while pd.Timestamp.now() + self.burst_time_limit < self.last_query_time:
            time.sleep(1)

    def _multiple_dates_queries(self, uri, args):
        date_froms, date_tos = self._get_periods(date_from=args["date_from"], date_to=args["date_to"])
        df = pd.DataFrame()
        metadata = []
        for date_from, date_to in zip(date_froms, date_tos):
            args["date_from"] = date_from
            args["date_to"] = date_to
            df_i, meta = self._query(uri, args)
            df = pd.concat((df, df_i), axis=0)
            metadata.append(meta)

        return df, metadata

    def _query(self, uri, args):

        self.wait_for_burst()
        res = requests.get(uri, params=args, headers=self.headers)

        # Check if sustained limit is reached
        if res.status_code == 429:
            print(res.text)
            available_in_seconds = int(res.text.split(" ")[-2])

            # Wait.
            for i in range(available_in_seconds):
                time.sleep(1)
                if i % (60 * 5) == 0:
                    print(f"available in {(available_in_seconds-i)/60:.2f} min.")

            res = requests.get(uri, params=args, headers=self.headers, max_retries=5, delay_between_retries=3)

        try:
            res.raise_for_status()
        except Exception as e:
            print(res.text)
            raise e

        parsed_response = res.json()
        df = pd.read_json(json.dumps(parsed_response["data"]), orient="index")
        metadata = parsed_response["metadata"]

        return df, metadata

    def _get_periods(self, date_from: str, date_to: str):
        """
        
        Chunks date period if it is larger than 1 year.
        """

        date_from = datetime.strptime(date_from, "%Y-%m-%d").date()
        date_froms = [date_from]

        date_to = datetime.strptime(date_to, "%Y-%m-%d").date()
        date_tos = []

        max_current_period = date(year=date_froms[-1].year + 1, month=date_froms[-1].month, day=date_froms[-1].day)

        while max_current_period < date_to:
            date_tos.append(max_current_period)
            date_froms.append(max_current_period + timedelta(days=1))
            max_current_period = date(year=date_froms[-1].year + 1, month=date_froms[-1].month, day=date_froms[-1].day)

        date_tos.append(date_to)

        return date_froms, date_tos

    def get_wind_dataframe(
        self,
        lat,
        long,
        date_from: str,
        date_to: str,
        dataset="merra2",
        capacity=1.0,
        height=100,
        turbine="Vestas V80 2000",
        interpolate=False,
    ):
        args = {
            "lat": lat,
            "lon": long,
            "date_from": date_from,
            "date_to": date_to,
            "dataset": dataset,
            "capacity": capacity,
            "height": height,
            "turbine": turbine,
            "interpolate": interpolate,
            "format": "json",
        }

        return self._multiple_dates_queries(NinjaClient.WIND_URI, args)

    def get_solar_dataframe(
        self,
        lat,
        long,
        date_from: str,
        date_to: str,
        dataset="merra2",
        capacity=1.0,
        system_loss=0.1,
        tracking=0,
        tilt=35,
        azim=180,
        interpolate=False,
    ):
        args = {
            "lat": lat,
            "lon": long,
            "date_from": date_from,
            "date_to": date_to,
            "dataset": dataset,
            "capacity": capacity,
            "system_loss": system_loss,
            "tracking": tracking,
            "tilt": tilt,
            "azim": azim,
            "format": "json",
        }

        return self._multiple_dates_queries(NinjaClient.PV_URI, args)

    def get_countries(self):
        res = requests.get(NinjaClient.COUNTRIES_URI, headers=self.headers)
        return pd.DataFrame(res.json()["countries"])

    def get_limits(self):
        res = requests.get(NinjaClient.LIMITS_URI, headers=self.headers)
        return res.json()
