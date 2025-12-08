from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.user import User
from app.utils.security import hash_password, verify_password
from app.utils.jwt_handler import create_access_token
from app.schemas.user import UserCreate, UserLogin

router = APIRouter(prefix="/auth", tags=["Auth"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/signup")
def signup(data: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already exists")

    hashed = hash_password(data.password)

    new_user = User(
        email=data.email,
        password_hash=hashed,
        full_name=data.full_name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": str(new_user.id), "email": new_user.email})
    return {"token": token, "user": new_user.email}


@router.post("/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return {"token": token, "user": user.email}
