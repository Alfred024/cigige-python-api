

import random
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://cicuge-app.vercel.app/"
]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("../potatoDisease/potatoDisease/models")
# MODEL = tf.keras.models.load_model("../potatoDisease/potatoDisease/models/1.keras")
PLANTS_STATE = ["Ready"]
# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, ping."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Predict endpoint A
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # base64_bytes = base64_img.split(',')[1]
    # img_bytes = base64.b64decode(base64_bytes)
    # img_bytes = base64.b64decode((base64_bytes)) # Si no funciona el de arriba, usar este
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    response = random.choice(PLANTS_STATE)
    
    # predictions = MODEL.predict(img_batch)
    # predicted_class = PLANTS_STATE[np.argmax(predictions[0])]
    # confidence = np.max(predictions[0])
    # print(predictions)
    # return {
    #     'class': predicted_class,
    #     'confidence': float(confidence)
    # }
    return { "prediction": response }


if __name__ == "__main__": 
    uvicorn.run(app, host='localhost', port=3001)
