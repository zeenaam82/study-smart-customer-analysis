# 스마트 고객 분석 및 추천 시스템 (Django + PostgreSQL + PostGIS + ORM + PyTorch)

## 개요
> 고객 데이터를 분석하여 개인화된 추천 시스템을 제공하는 서비스입니다. <br>
> 해당 시스템은 Amazon 제품 리뷰 데이터를 사용하여 고객의 취향을 분석하고, 그에 맞는 제품을 추천합니다. <br>
> 시스템은 FastAPI를 사용한 서버와 PostgreSQL 데이터베이스를 기반으로 하며, PyTorch를 사용하여 텍스트 분류 모델을 학습합니다. <br>
> 프로젝트의 아키텍처와 작업 흐름에 대한 자세한 사항은 `etc/architecture`와 `etc/jobflow` 폴더를 참고해 주세요.

## 데이터셋
- 데이터셋: [Amazon Product Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

## 실행
### 서버 실행
> uvicorn src.run:app --host 0.0.0.0 --port 8000
### 모델 학습
> python -m src.run train
### 예측 실행
> python -m src.run predict --text "제품 텍스트"
### Pytest
> pytest --maxfail=1 --disable-warnings -q