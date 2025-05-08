# 스마트 고객 분석 및 추천 시스템 (Django + PostgreSQL + PostGIS + ORM + PyTorch)

## 개요
> 고객 데이터를 분석하여 개인화된 추천 시스템을 제공하는 서비스. <br>
> etc/architecture,jobflow 참고

## DATE SET
> https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

## 실행
> API 서버 실행
>   > uvicorn src.run:app --host 0.0.0.0 --port 8000
>   > 이 명령어는 API 서버를 실행합니다. uvicorn이 API 서버를 0.0.0.0:8000에서 실행합니다.

> 모델 학습
>   > python -m src.run train --epochs 5

> 예측 실행
>   > python -m src.run predict --text "제품 텍스트"

> Pytest
>   > pytest --maxfail=1 --disable-warnings -q