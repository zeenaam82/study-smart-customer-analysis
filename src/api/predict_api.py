from fastapi import FastAPI
from pydantic import BaseModel
from src.model.predict import predict

app = FastAPI(title="Smart Customer Classifier API")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def get_prediction(request: TextRequest):
    prediction = predict(request.text)  # predict 함수 호출
    return {"predicted_class": prediction}

