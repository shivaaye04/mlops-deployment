import pickle
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# load model
model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Model API running"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}