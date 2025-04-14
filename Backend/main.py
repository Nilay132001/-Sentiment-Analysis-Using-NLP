from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from pymongo import MongoClient
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# === CONFIG ===
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_db"]
users = db["users"]

# Password hashing and OAuth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === NLTK Setup ===
nltk.data.path.append("/Users/nilay/nltk_data")  # Adjust to your local path
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# === Load Sentiment Model ===
with open("model.pkl", "rb") as f:
    tfidf, model = pickle.load(f)

# === Pydantic Models ===


class User(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TextInput(BaseModel):
    text: str

# === Utility Functions ===


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def hash_password(password):
    return pwd_context.hash(password)


def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = users.find_one({"username": payload.get("sub")})
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid user")
        return user
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+|\S+\.\S{2,}', "", text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def apply_stemming(text):
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# === ROUTES ===

# Auth: Signup


@app.post("/signup")
def signup(user: User):
    if users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_pw = hash_password(user.password)
    users.insert_one({"username": user.username, "password": hashed_pw})
    return {"message": "Signup successful"}

# Auth: Login


@app.post("/login", response_model=Token)
def login(credentials: User):
    user = users.find_one({"username": credentials.username})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(
            status_code=400, detail="Incorrect username or password")
    token = create_token({"sub": credentials.username})
    return {"access_token": token, "token_type": "bearer"}

# Auth: Protected route


@app.get("/me")
def read_users_me(current_user=Depends(get_current_user)):
    return {"username": current_user["username"]}

# Sentiment Prediction


@app.post("/predict")
async def predict_sentiment(input_text: TextInput, current_user=Depends(get_current_user)):
    raw_text = input_text.text
    cleaned = clean_text(raw_text)
    stemmed = apply_stemming(cleaned)
    transformed = tfidf.transform([stemmed])
    prediction = int(model.predict(transformed)[0])
    return {"prediction": prediction}
