import os
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
import models
from schemas import UserRequest
from database import get_db

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_key")
ALGORITHM = "HS256"

@router.post("/register")
def register_user(user: UserRequest, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username sudah terdaftar")
    
    hashed_pw = pwd_context.hash(user.password)
    new_user = models.User(username=user.username, hashed_password=hashed_pw)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Registrasi berhasil", "user_id": new_user.id}

@router.post("/login")
def login(user: UserRequest, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Kredensial tidak valid")
    
    access_token_expires = timedelta(minutes=60)
    expire = datetime.now(timezone.utc) + access_token_expires
    to_encode = {"sub": db_user.username, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return {"access_token": encoded_jwt, "token_type": "bearer", "username": db_user.username}