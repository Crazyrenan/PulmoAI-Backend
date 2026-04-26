import io
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import keras
from keras.applications.densenet import preprocess_input

original_from_config = keras.layers.Dense.from_config

@classmethod
def patched_from_config(cls, config):
    config.pop("quantization_config", None)
    return original_from_config(config)

keras.layers.Dense.from_config = patched_from_config

MODEL_PATH = "model/tbc_model.keras"
CLASS_NAMES = ["health", "sick", "tb"]
TARGET_SIZE = (300, 300)

ml_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_components["model"] = keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input}
        )
    except Exception as e:
        print(f"Failed to load model at startup: {e}")
    yield
    ml_components.clear()

app = FastAPI(lifespan=lifespan)

def process_image(img_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "model" not in ml_components:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        img_bytes = await file.read()
        img_array = process_image(img_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file upload")

    try:
        predictions = ml_components["model"].predict(img_array, verbose=0)
        probabilities = predictions[0].tolist()
        predicted_idx = int(np.argmax(probabilities))
        
        return {
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": float(probabilities[predicted_idx]),
            "probabilities": [float(p) for p in probabilities]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")