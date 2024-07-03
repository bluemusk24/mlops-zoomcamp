import os

import pickle

import mlflow
from mlflow.tracking import MlflowClient

from flask import Flask, request, jsonify


# get tracking_uri with MLflowClient and mlflow

#mlflow_tracking_uri = "http://127.0.0.1:5000"
#mlflow.set_tracking_uri(mlflow_tracking_uri)

# Using the model from mlflow ui with the RUN_ID

RUN_ID = os.getenv('RUN_ID')                  #'52871e78853c4a6d9e0ed9a024812f9d'
#logged_model = f'runs:/{RUN_ID}/model'

# model from aws s3 bucket
logged_model = f's3://mlops-remotebucket/2/{RUN_ID}/artifacts/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)