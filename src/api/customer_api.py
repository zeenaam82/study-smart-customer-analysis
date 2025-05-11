from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db.db_models import CustomerData

router = APIRouter()

# 이 엔드포인트는 고객 데이터를 데이터베이스에 추가하는 API입니다.
# 요청 본문에서 `review`와 `label` 데이터를 받아 데이터베이스에 새 레코드로 저장합니다.
# 주로 데이터를 DB에 삽입하는 기능을 담당합니다.
@router.post("/add/")
async def add_customer_data(review: str, label: int, db: Session = Depends(get_db)):
    new_data = CustomerData(review=review, label=label)
    db.add(new_data)
    db.commit()
    db.refresh(new_data)
    return {"message": "Data added successfully", "data": {"id": new_data.id, "review": new_data.review, "label": new_data.label}}

# 이 엔드포인트는 데이터베이스에서 특정 고객 데이터를 조회하는 API입니다.
# URL 경로에서 `data_id` 값을 받아 해당 ID에 맞는 데이터를 데이터베이스에서 조회하여 반환합니다.
# 특정 고객 데이터 하나를 가져오는 기능을 담당합니다.
@router.get("/get/{data_id}")
async def get_customer_data(data_id: int, db: Session = Depends(get_db)):
    data = db.query(CustomerData).filter(CustomerData.id == data_id).first()
    if not data:
        raise HTTPException(status_code=404, detail="Data not found")
    return {"id": data.id, "review": data.review, "label": data.label}
