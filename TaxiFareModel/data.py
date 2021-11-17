import pandas as pd
from sklearn.model_selection import train_test_split

from TaxiFareModel import params

#BUCKET_NAME = 'wagon-data-745-steininger'
#BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(f"gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}",
                     nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def get_train_test_data(df):
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = get_data()
    print('df shape: ', df.shape)
