import fastapi
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
#potato leaf model  
model = tf.keras.models.load_model("C:/Users/premkumar.m/potato_disease_code/saved_models/output.h5")
class_names = ['Early Blight', 'Late Blight', 'Healthy']

#tomato leaf model
#model = tf.keras.models.load_model("C:/Users/premkumar.m/potato_disease_code/saved_models/tomato1.h5")
#class_names=['Bacterial_spot','Early_blight','healthy','Late_blight','Leaf_Mold','Septoria_leaf_spot','Two-spotted_spider_mite','Target_Spot','mosaic_virus','Yellow_Leaf_Curl_Virus']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    pred = model.predict(img_batch)
    pred_class = class_names[np.argmax(pred[0])]
    conf = np.max(pred[0])
    return {
        'class': pred_class,
        'confidence': float(conf)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
