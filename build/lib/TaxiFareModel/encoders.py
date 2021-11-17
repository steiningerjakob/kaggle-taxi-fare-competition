import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
import pygeohash as gh

# Custom DistanceTransformer
class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(X_,
                                              start_lat=self.start_lat,
                                              start_lon=self.start_lon,
                                              end_lat=self.end_lat,
                                              end_lon=self.end_lon)
        return X_[['distance']]


class AddGeohash(BaseEstimator, TransformerMixin):
    def __init__(self, precision=6):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['geohash_pickup'] = X.apply(lambda x: gh.encode(
            x.pickup_latitude, x.pickup_longitude, precision=self.precision),
                                      axis=1)
        X['geohash_dropoff'] = X.apply(lambda x: gh.encode(
            x.dropoff_latitude, x.dropoff_longitude, precision=self.precision),
                                       axis=1)
        return X[['geohash_pickup', 'geohash_dropoff']]


class DistanceToCenter(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, X, y=None):
        nyc_center = (40.7141667, -74.0063889)
        X["nyc_lat"], X["nyc_lng"] = nyc_center[0], nyc_center[1]
        args_pickup = dict(start_lat="nyc_lat",
                           start_lon="nyc_lng",
                           end_lat="pickup_latitude",
                           end_lon="pickup_longitude")
        args_dropoff = dict(start_lat="nyc_lat",
                            start_lon="nyc_lng",
                            end_lat="dropoff_latitude",
                            end_lon="dropoff_longitude")
        X['pickup_distance_to_center'] = haversine_vectorized(X, **args_pickup)
        X['dropoff_distance_to_center'] = haversine_vectorized(
            X, **args_dropoff)
        return X[["pickup_distance_to_center", "dropoff_distance_to_center"]]

    def fit(self, X, y=None):
        return self


class Direction(BaseEstimator, TransformerMixin):
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def transform(self, X, y=None):
        def calculate_direction(d_lon, d_lat):
            result = np.zeros(len(d_lon))
            l = np.sqrt(d_lon**2 + d_lat**2)
            result[d_lon > 0] = (180 / np.pi) * np.arcsin(
                d_lat[d_lon > 0] / l[d_lon > 0])
            idx = (d_lon < 0) & (d_lat > 0)
            result[idx] = 180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
            idx = (d_lon < 0) & (d_lat < 0)
            result[idx] = -180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
            return result

        X['delta_lon'] = X[self.start_lon] - X[self.end_lon]
        X['delta_lat'] = X[self.start_lat] - X[self.end_lat]
        X['direction'] = calculate_direction(X.delta_lon, X.delta_lat)
        return X[["delta_lon", "delta_lat", "direction"]]

    def fit(self, X, y=None):
        return self

# Custom TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]

# Custom dataset size optimizer
class DatasetOptimizer(BaseEstimator, TransformerMixin):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    def __init__(self, **kwargs):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, verbose=True, **kwargs):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()

        in_size = X_.memory_usage(index=True).sum()

        # Optimized size here
        for type in ["float", "integer"]:
            l_cols = list(X_.select_dtypes(include=type))
            for col in l_cols:
                X_[col] = pd.to_numeric(X_[col], downcast=type)
                if type == "float":
                    X_[col] = pd.to_numeric(X_[col], downcast="integer")
        out_size = X_.memory_usage(index=True).sum()
        ratio = (1 - round(out_size / in_size, 2)) * 100
        GB = out_size / 1000000000
        if verbose:
            print("optimized size by {} % | {} GB".format(ratio, GB))
        return X_
