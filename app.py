# FastAPI Chatbot Server
import random
import numpy as np
import pickle
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Create FastAPI app
app = FastAPI(
    title="AI Chatbot API",
    description="Modern AI Chatbot powered by FastAPI",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and data
try:
    model = load_model("chatbot_model.h5")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    with open("intents.json", "r") as f:
        intents = json.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or data files: {str(e)}")

# Request model
class MessageRequest(BaseModel):
    msg: str

# Response model
class ChatResponse(BaseModel):
    response: str
    confidence: float

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.post("/api/chat", response_model=ChatResponse)
async def chatbot_response(request: MessageRequest):
    if not request.msg or request.msg.strip() == "":
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    msg = request.msg.strip()
    
    try:
        # Handle name patterns
        if msg.lower().startswith('my name is'):
            name = msg[11:].strip()
            ints = predict_class(msg, model)
            confidence = float(ints[0]["probability"]) if ints else 0.0
            res1 = getResponse(ints, intents)
            res = res1.replace("{n}", name)
        elif msg.lower().startswith('hi my name is'):
            name = msg[14:].strip()
            ints = predict_class(msg, model)
            confidence = float(ints[0]["probability"]) if ints else 0.0
            res1 = getResponse(ints, intents)
            res = res1.replace("{n}", name)
        elif msg.lower().startswith('i am'):
            name = msg[4:].strip()
            ints = predict_class(msg, model)
            confidence = float(ints[0]["probability"]) if ints else 0.0
            res1 = getResponse(ints, intents)
            res = res1.replace("{n}", name)
        else:
            ints = predict_class(msg, model)
            confidence = float(ints[0]["probability"]) if ints else 0.0
            res = getResponse(ints, intents)
        
        return ChatResponse(response=res, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}

# Chat processing functions
def clean_up_sentence(sentence):
    """Tokenize and lemmatize sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    """Create bag of words array for sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


def predict_class(sentence, model):
    """Predict intent class from sentence"""
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    """Get random response for predicted intent"""
    if not ints:
        return "Sorry, I didn't understand that. Could you please rephrase?"
    
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            return result
    return "Sorry, I didn't understand that. Could you please rephrase?"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

