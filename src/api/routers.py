from fastapi import APIRouter
from .predict_api import router as predict_router
from .customer_api import router as customer_router

# 메인 라우터 생성
router = APIRouter()

# 서브 라우터 등록
router.include_router(predict_router, prefix="/predict", tags=["Predict"])
router.include_router(customer_router, prefix="/customer", tags=["Customer Data"])
