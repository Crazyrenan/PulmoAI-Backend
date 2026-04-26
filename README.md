# PulmoAI Backend

FastAPI backend for Tuberculosis (TBC) detection using deep learning on chest X-ray images.

## Features
- Image classification (health, sick, tb)
- Keras model inference
- REST API with FastAPI

## Endpoint

### POST /predict

Upload image file.

**Request:**
- form-data → file (image)

**Response:**
```json
{
  "prediction": "tb",
  "confidence": 0.99,
  "probabilities": [0.01, 0.02, 0.97]
}