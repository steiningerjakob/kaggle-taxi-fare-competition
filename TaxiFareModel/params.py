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


# MLFlow configuration

# for mlflow
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_BASE_NAME = '[SW] [Stockholm] [steiningerjakob] taxifare'

if __name__ == '__main__':
    pass
