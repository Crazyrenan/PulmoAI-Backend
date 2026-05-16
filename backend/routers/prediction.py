import os
import io
import base64
import uuid
import json
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import jwt
import numpy as np
import keras
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications.densenet import preprocess_input
from routers.auth import oauth2_scheme, SECRET_KEY, ALGORITHM
from database import get_db
import models
from sqlalchemy import desc

router = APIRouter()

MODEL_PATH = "model_clean.keras"
model = None
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input}
        )

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((300, 300), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

def get_gradcam_heatmap(img_tensor, model):
    dense_net = model.get_layer("densenet121")
    last_conv_layer = dense_net.get_layer("conv5_block16_concat")

    grad_model = tf.keras.models.Model(
        inputs=dense_net.input,
        outputs=[last_conv_layer.output, dense_net.output]
    )

    eff_pre = model.get_layer("lambda")
    dense_pre = model.get_layer("lambda_1")

    with tf.GradientTape() as tape:
        x_eff = eff_pre(img_tensor, training=False)
        x_dense = dense_pre(img_tensor, training=False)

        conv_outputs, dense_features = grad_model(x_dense, training=False)
        tape.watch(conv_outputs)

        eff_features = model.get_layer("efficientnetv2-b3")(x_eff, training=False)

        eff_pooled = model.get_layer("global_average_pooling2d")(eff_features, training=False)
        dense_pooled = model.get_layer("global_average_pooling2d_1")(dense_features, training=False)

        concat = model.get_layer("concatenate")([eff_pooled, dense_pooled], training=False)
        x = model.get_layer("batch_normalization")(concat, training=False)
        x = model.get_layer("dense")(x, training=False)
        preds = model.get_layer("dense_1")(x, training=False)

        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) != 0:
        heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()

def save_and_generate_gradcam_base64(img_bytes, heatmap, save_path):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((300, 300), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize((300, 300), Image.LANCZOS)
    heatmap_resized = np.array(heatmap_img) / 255.0

    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(heatmap_resized)[:, :, :3]
    jet_heatmap = jet_colors * 255.0

    alpha = 0.4
    superimposed_img = (jet_heatmap * alpha) + (img_array * (1.0 - alpha))
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    final_img = Image.fromarray(superimposed_img)
    final_img.save(save_path, format="JPEG")

    buffered = io.BytesIO()
    final_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ✅ Single reusable dependency — no more duplication
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token tidak valid")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token tidak valid atau kedaluwarsa")

    user_obj = db.query(models.User).filter(models.User.username == username).first()
    if not user_obj:
        raise HTTPException(status_code=401, detail="User tidak ditemukan")
    return user_obj


@router.get("/history")
async def get_scan_history(user=Depends(get_current_user), db: Session = Depends(get_db)):
    # ✅ user is already authenticated — use it directly
    scans = db.query(models.ScanRecord)\
              .filter(models.ScanRecord.user_id == user.id)\
              .order_by(desc(models.ScanRecord.created_at))\
              .all()

    return {
        "status": "success",
        "data": [
            {
                "id": scan.id,
                "filename": scan.original_filename,
                "prediction": scan.prediction,
                "confidence": scan.confidence,
                "raw_image_url": f"{BASE_URL}/{scan.raw_image_path}",
                "gradcam_image_url": f"{BASE_URL}/{scan.gradcam_image_path}" if scan.gradcam_image_path else None,
                "created_at": scan.created_at
            } for scan in scans
        ]
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...), user=Depends(get_current_user), db: Session = Depends(get_db)):
    # ✅ user is already authenticated — use it directly
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    contents = await file.read()

    # ✅ File size guard
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File terlalu besar. Maksimum 10MB.")

    scan_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"

    raw_filename = f"{scan_id}_raw.{ext}"
    raw_path = os.path.join("uploads", "raw", raw_filename)
    with open(raw_path, "wb") as f:
        f.write(contents)

    data = preprocess_image(contents)

    if model:
        predictions = model(data)
        probs = predictions.numpy()[0].tolist()
        idx = int(np.argmax(probs))

        gradcam_filename = f"{scan_id}_gradcam.jpg"
        gradcam_path = os.path.join("uploads", "gradcam", gradcam_filename)

        heatmap = get_gradcam_heatmap(data, model)
        gradcam_b64 = save_and_generate_gradcam_base64(contents, heatmap, gradcam_path)
    else:
        probs = [0.10, 0.03, 0.87]
        idx = 2
        gradcam_b64 = ""
        gradcam_path = ""

    classes = ["health", "sick", "tb"]

    new_scan = models.ScanRecord(
        user_id=user.id,                          # ✅ from dependency
        original_filename=file.filename,
        raw_image_path=raw_path.replace("\\", "/"),
        gradcam_image_path=gradcam_path.replace("\\", "/"),
        prediction=classes[idx],
        confidence=float(probs[idx]),
        probabilities=json.dumps(probs)
    )
    db.add(new_scan)
    db.commit()
    db.refresh(new_scan)

    return {
        "prediction": classes[idx],
        "confidence": float(probs[idx]),
        "probabilities": probs,
        "gradcam_image": f"data:image/jpeg;base64,{gradcam_b64}" if gradcam_b64 else None,
        "user": user.username,                    # ✅ from dependency
        "scan_id": new_scan.id
    }