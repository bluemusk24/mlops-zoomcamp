#!/usr/bin/env python
# coding: utf-8


import sys
import pickle
import pandas as pd
import os 
import uuid

import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


# function to generate the uuid attached to each row in the dataframe

def generate_uuid(n):
    ride_id = []
    for i in range(n):
        ride_id.append(str(uuid.uuid4()))
    return ride_id

# function to read the dataframe

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df['ride_id'] = generate_uuid(len(df))
    return df


# function to prepare dictionaries

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# function to load model

def load_model(run_id):
    logged_model = f's3://mlops-remote-bucket/3/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


# function to apply the model

def apply_model(input_file, run_id, output_file):
    print(f'reading the data from {input_file}..')
    df = read_dataframe(input_file)
    dict = prepare_dictionaries(df)

    print(f'loading the model with RUN_ID={run_id}..')
    model = load_model(run_id)
    
    print(f'applying the model...')
    y_pred = model.predict(dict)

    print(f'saving the result to {output_file}')
    df_result = pd.DataFrame() 
    df_result['ride_id'] = df['ride_id']                
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_parquet(output_file, index=False)


# function to run the model. We use sys to parameterize the model

def run():
    taxi_type = sys.argv[1]   #green
    year = int(sys.argv[2])        #2021
    month = int(sys.argv[3])     # month 2
    
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'

    run_id = sys.argv[4]          # os.getenv('RUN_ID','729cc0899e22468bbe0bf20321b84225')
    
    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)


if __name__ == '__main__':
    run()
