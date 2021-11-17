### TODO: mlflow not working because of lewagon environment conflict

from google.cloud import storage
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import linear_model
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data, get_train_test_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.params import *
import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from termcolor import colored



class Trainer():

    def __init__(self):
        # for mlflow
        self.mlflow_URI = MLFLOW_URI
        self.experiment_name = EXPERIMENT_BASE_NAME

    # change default mlflow experiment name
    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name


    # define  pipeline
    def set_pipeline(self):
        '''defines the pipeline as a class attribute and returns it'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('model', linear_model.Lasso(alpha=0.1))
        ])

        self.pipeline = pipe

    def train(self, X_train, y_train):
        '''set and train the pipeline'''
        self.set_pipeline()
        # for mlflow
        #self.mlflow_log_param("model", self.pipeline['model'])
        # train the pipelined model on X_train and y_train
        trained_pipeline = self.pipeline.fit(X_train, y_train)

        self.trained_pipeline = trained_pipeline

    def evaluate(self, X_test, y_test):
        '''evaluates model performance on X_test and returns RMSE'''
        # compute y_pred on X_test
        y_pred = self.trained_pipeline.predict(X_test)
        self.y_pred = y_pred

        # evaluate model performance based on RMSE
        rmse = compute_rmse(y_pred, y_test)

        # for mlflow
        #self.mlflow_log_metric("rmse", rmse)

        self.rmse = rmse

    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

    def save_model(self):
        """Save the model into a .joblib format and upload to Google Storage"""
        # save model locally, i.e. to disk
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

        # upload model to GCP
        self.upload_model_to_gcp()
        print(
            f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}"
        )


    # mlflow methods:
'''
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
'''


if __name__ == '__main__':
    # fetch, clean and extract relevant data
    df = get_data()
    clean_df = clean_data(df)
    X_train, X_test, y_train, y_test = get_train_test_data(clean_df)

    # train, evaluate and save model, and document on mlflow
    trainer = Trainer()
    trainer.set_pipeline()
    trainer.train(X_train, y_train)
    rmse = trainer.evaluate(X_test, y_test)
    print('rmse: ', rmse)
    trainer.save_model()
