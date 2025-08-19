from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load
from pydantic import BaseModel
import uvicorn

# Load trained model
model = load("fake_news_pipeline.joblib")

# Initialize FastAPI app
app = FastAPI()

# Templates & static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# For AJAX requests
class NewsRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(news: NewsRequest):
    # Predict label
    prediction = model.predict([news.text])[0]
    
    # Predict probability (Real = 1, Fake = 0)
    proba = model.predict_proba([news.text])[0]
    real_prob = round(proba[1] * 100, 2)
    fake_prob = round(proba[0] * 100, 2)

    result = "✅ Real News" if prediction == 1 else "❌ Fake News"
    return {"result": result, "real_prob": real_prob, "fake_prob": fake_prob}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
