import datetime as datetime
import pandas as pd
import requests



from fastapi import FastAPI
from pydantic import BaseModel
from joblib import dump, load
from datetime import datetime
from geopy import distance
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_distance(pick_lat, pick_long, drop_lat, drop_long):
    dist = distance.geodesic((pick_lat, pick_long), (drop_lat, drop_long)).km
    return dist


def get_dataframe(item):
    return pd.DataFrame({
        'vendor_id': [item.vendor_id],
        'pickup_longitude': [item.pickup_longitude],
        'pickup_latitude': [item.pickup_latitude],
        'dropoff_longitude': [item.dropoff_longitude],
        'dropoff_latitude': [item.dropoff_latitude],
        'hour': [datetime.strptime(item.datetime, '%Y-%m-%d %H:%M').hour],
        'minute': [datetime.strptime(item.datetime, '%Y-%m-%d %H:%M').minute],
        'day_week': [datetime.strptime(item.datetime, '%Y-%m-%d %H:%M').weekday()],
        'month': [datetime.strptime(item.datetime, '%Y-%m-%d %H:%M').month],
        'distance': [
            get_distance(item.pickup_latitude, item.pickup_longitude, item.dropoff_latitude, item.dropoff_longitude)]
    })


class Item(BaseModel):
    vendor_id: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    datetime: str


@app.post("/predict-taxi-trip")
async def taxi_trip(item: Item):
    xgb = load('xgb.joblib')

    df = get_dataframe(item)
    prediction = xgb.predict(df)
    result = pow(2, prediction.item(0)) - 1
    return {'result': result}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
