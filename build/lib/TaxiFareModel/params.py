# GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-745-steininger'

STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

# path to local model for predict
PATH_TO_LOCAL_MODEL = 'model.joblib'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Training params (CURRENTLY NOT USED)

params = dict(
    nrows=10000,
    upload=True,
    local=False,  # set to False to get data from GCP (Storage or BigQuery)
    gridsearch=False,
    optimize=True,
    estimator="xgboost",
    mlflow=True,  # set to True to log params to mlflow
    mlflow_URI='https://mlflow.lewagon.co/',
    experiment_name='[SW] [Stockholm] [steiningerjakob] taxifare',
    pipeline_memory=None,  # None if no caching and True if caching expected
    distance_type="manhattan",
    feateng=[
        "distance_to_center", "direction", "distance", "time_features",
        "geohash"
    ],
    n_jobs=-1)  # Try with njobs=1 and njobs = -1


DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")
