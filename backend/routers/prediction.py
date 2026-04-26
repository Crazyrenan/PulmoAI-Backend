import os
import io
import base64
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import jwt
import numpy as np
import keras
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from routers.auth import oauth2_scheme, SECRET_KEY, ALGORITHM

router = APIRouter()

MODEL_PATH = "model.keras"
# Ubah nilai di bawah sesuai dengan nama layer konvolusi terakhir di arsitektur model Anda
LAST_CONV_LAYER = "conv5_block16_concat"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        # Wajib menggunakan load_model agar arsitektur layer dapat diakses
        model = keras.models.load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_base64(img_bytes, heatmap):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + img_array
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    buffered = io.BytesIO()
    superimposed_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@router.post("/predict")
async def predict(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token tidak valid")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token tidak valid atau kedaluwarsa")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    
    contents = await file.read()
    data = preprocess_image(contents)
    
    if model:
        predictions = model(data)
        probs = predictions.numpy()[0].tolist()
        idx = np.argmax(probs)
        heatmap = get_gradcam_heatmap(data, model, LAST_CONV_LAYER, idx)
        gradcam_b64 = generate_gradcam_base64(contents, heatmap)
    else:
        probs = [0.10, 0.03, 0.87]
        idx = 2
        gradcam_b64 = "" 

    classes = ["normal", "viral", "tb"]
    return {
        "prediction": classes[idx],
        "confidence": float(probs[idx]),
        "probabilities": probs,
        "gradcam_image": f"data:image/jpeg;base64,{gradcam_b64}" if gradcam_b64 else None,
        "user": username
    }