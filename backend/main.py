import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import models
from database import engine
from routers import auth, prediction

load_dotenv()

models.Base.metadata.create_all(bind=engine)

os.makedirs("uploads/raw", exist_ok=True)
os.makedirs("uploads/gradcam", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_credentials=True,  
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(auth.router, tags=["Authentication"])
app.include_router(prediction.router, tags=["Prediction"])  

@app.on_event("startup")
def startup_event():
    prediction.load_model()