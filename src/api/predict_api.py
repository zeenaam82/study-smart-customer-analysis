from fastapi import APIRouter
from src.model.predict import predict

router = APIRouter()

# 이 엔드포인트는 텍스트 데이터 (리뷰)를 받아서 모델을 통해 예측을 수행하는 API입니다.
# 요청 본문에서 `review` 데이터를 받아 모델에 전달하여, 예측된 레이블을 반환합니다.
# 예측 결과를 반환하는 역할을 합니다.
@router.post("/")
async def predict_sentiment(text: str):
    label = predict(text)
    return {"predicted_label": label}
