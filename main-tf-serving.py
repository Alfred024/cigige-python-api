# HTTP deps.
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn

import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000" # APP server
]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TF Server model path 
endpoint = ""

CLASS_NAMES = ["","",""]

@app.get("/ping")
async def ping():
    return "Ping succesfully made it"

# Recibe un archivo de la aplicación que ve el usuario
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = convertFileImgToArray(file=file) 
    image_batch = np.expand_dims(image, 0) # Para prepararlo para su procesamiento, crea un array de múltiples imágenes [[256,256,1]]
    
    json_data = {
        "instances": image_batch.tolist()
    }
    
    # Made an http request to the TF server with the model
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])
    
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

# Format file to numpy array 
def convertFileImgToArray(file) -> np.ndarray:
        arrayImage = np.array(Image.open(BytesIO(file)))
        return arrayImage
    
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=3002)