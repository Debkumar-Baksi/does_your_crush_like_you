from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# load model once
model = tf.keras.models.load_model("crush_model.keras")

app = FastAPI()

# allow local frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CrushInput(BaseModel):
    looks: int
    height: int
    confidence: int
    talk_frequency: int
    shared_interests: int

@app.post("/predict")
def predict(data: CrushInput):
    x = np.array([[
        data.looks,
        data.height,
        data.confidence,
        data.talk_frequency,
        data.shared_interests
    ]])

    prob = model.predict(x)[0][0]

    return {
        "probability": float(prob),
        "likes_you": bool(prob >= 0.5)
    }
