import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://0.0.0.0:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
