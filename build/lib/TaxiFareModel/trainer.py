### TODO: mlflow not working because of lewagon environment conflict

#import multiprocessing
#import time
#import warnings

from tempfile import mkdtemp
import category_encoders as ce
from google.cloud import storage
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from TaxiFareModel.encoders import *
from TaxiFareModel.data import get_train_test_data
from TaxiFareModel.utils import compute_rmse, simple_time_tracker
from TaxiFareModel.params import *
import joblib
from termcolor import colored
from xgboost import XGBRegressor
#import memoized_property
#import mlflow
#from mlflow.tracking import MlflowClient


class Trainer():
    ESTIMATOR = "Linear"
    EXPERIMENT_NAME = "TaxifareModel"

    def __init__(self, **kwargs):
        self.pipeline = None
        self.kwargs = kwargs
        self.local = kwargs.get("local",
                                False)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.experiment_name = kwargs.get("experiment_name",
                                          self.EXPERIMENT_NAME)  # cf doc above
        self.estimator = kwargs.get("estimator", self.ESTIMATOR)
        self.model_params = None
        self.n_jobs = kwargs.get('n_jobs', -1)
        self.memory = self.kwargs.get("pipeline_memory", None)
        self.dist = self.kwargs.get("distance_type", "euclidian")
        self.feateng_steps = self.kwargs.get("feateng",
                                        ["distance", "time_features"])
        #self.log_kwargs_params()
        #self.log_machine_specs()

    # change default mlflow experiment name
    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def get_estimator(self):
        if self.estimator == "Lasso":
            model = Lasso()
        elif self.estimator == "Ridge":
            model = Ridge()
        elif self.estimator == "Linear":
            model = LinearRegression()
        elif self.estimator == "GBM":
            model = GradientBoostingRegressor()
        elif self.estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']
            }
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif self.estimator == "xgboost":
            model = XGBRegressor(objective='reg:squarederror',
                                 n_jobs=self.n_jobs,
                                 max_depth=10,
                                 learning_rate=0.05,
                                 gamma=3)
            self.model_params = {
                'max_depth': range(10, 20, 2),
                'n_estimators': range(60, 220, 40),
                'learning_rate': [0.1, 0.01, 0.05]
            }
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        #self.mlflow_log_param("estimator", self.estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model


    # define  pipeline
    def set_pipeline(self):
        '''defines the pipeline as a class attribute and returns it'''
        if self.memory:
            memory = mkdtemp()

        # Define feature engineering pipeline blocks here
        pipe_time_features = make_pipeline(
            TimeFeaturesEncoder(time_column='pickup_datetime'),
            OneHotEncoder(handle_unknown='ignore'))
        pipe_distance = make_pipeline(
            DistanceTransformer(distance_type=self.dist, **DIST_ARGS),
            StandardScaler())
        pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())
        pipe_direction = make_pipeline(Direction(), StandardScaler())
        pipe_distance_to_center = make_pipeline(DistanceToCenter(),
                                                StandardScaler())

        # Define default feature engineering blocs
        feateng_blocks = [
            ('distance', pipe_distance, list(DIST_ARGS.values())),
            ('time_features', pipe_time_features, ['pickup_datetime']),
            ('geohash', pipe_geohash, list(DIST_ARGS.values())),
            ('direction', pipe_direction, list(DIST_ARGS.values())),
            ('distance_to_center', pipe_distance_to_center,
             list(DIST_ARGS.values())),
        ]
        # Filter out some blocks according to input parameters
        for bloc in feateng_blocks:
            if bloc[0] not in self.feateng_steps:
                feateng_blocks.remove(bloc)

        features_encoder = ColumnTransformer(feateng_blocks,
                                             n_jobs=None,
                                             remainder="drop")

        self.pipeline = Pipeline(steps=[('features', features_encoder),
                                        ('model', self.get_estimator())],
                                 memory=memory)

    @simple_time_tracker
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

    # get_estimator():

    # gridsearch(): --> as part of pipe



    # mlflow methods (CURRENTLY NOT USED):
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

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)
'''


if __name__ == '__main__':
    # fetch, clean, optimize data and extract relevant data
    nrows = 10_000  # default
    data = "local"  # default
    X_train, X_test, y_train, y_test = get_train_test_data()

    # train, evaluate and save model, and document on mlflow
    trainer = Trainer()
    trainer.set_pipeline()
    trainer.train(X_train, y_train)
    rmse = trainer.evaluate(X_test, y_test)
    print('rmse: ', rmse)
    trainer.save_model()
