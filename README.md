# 스마트 고객 분석 및 추천 시스템 (Django + PostgreSQL + PostGIS + ORM + PyTorch)

## 개요
> 고객 데이터를 분석하여 개인화된 추천 시스템을 제공하는 서비스. <br>
> etc/architecture,jobflow 참고

## DATA SET
> https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

## 실행
> 서버 실행
>   > uvicorn src.run:app --host 0.0.0.0 --port 8000

> 모델 학습
>   > python -m src.run train

> 예측 실행
>   > python -m src.run predict --text "제품 텍스트"

> Pytest
>   > pytest --maxfail=1 --disable-warnings -q