# SMART-CUSTOMER-ANALYSIS ARCHITECTURE

+----------------+       +-------------------+      +-----------------+
|                |       |                   |      |                 |
|  Data Source   | --->  |   Data Processing  | ---> |   ML Model      |
|  (CSV, DB)     |       |   & Preprocessing  |      |   (PyTorch)     |
|                |       |                   |      |   (KMeans)      |
+----------------+       +-------------------+      +-----------------+
                                  |                        |
                                  v                        v
                            +------------------+     +------------------+
                            |   PostgreSQL DB  |     |    FastAPI API   |
                            |  (ORM, GIS)      | <--> | (Prediction API) |
                            +------------------+     +------------------+
                                                             |
                                                             v
                                                      +----------------+
                                                      | Docker / CI/CD |
                                                      +----------------+

## DIRECTORYT TREE

smart-customer-analysis/
├── data/
│   ├── test.ft.txt
│   └── train.ft.txt
│
├── src/
│   ├── __init__.py
│   ├── run.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── customer_api.py    # DB에 고객 데이터를 추가 및 조회 API
│   │   ├── predict_api.py    # 예측 API
│   │   └── routers.py        # API Router 설정
│   ├── data/ 
│   │   ├── __init__.py
│   │   ├── data_loader.py         # 데이터 로딩 및 전처리
│   │   └── dataset.py             # TextDataset 클래스
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py          # 모델 정의
│   │   ├── predict.py        # 예측 코드
│   │   └── train.py          # 학습 코드
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # 설정 파일
│       ├── logger.py         # 로깅 코드
│       └── utils.py               # 데이터 전처리, 평가 관련 유틸리티
│
├── db_init/
│   └── init.sql        # DB 자동화 설정
│
├── docker/
│   ├── .dockerignore        # Docker 예외 처리 파일
│   └── Dockerfile            # Docker 자동화 설정
│
├── data/        # Dataset 저장
│
├── save_models/        # 학습된 모델 저장
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py         # 데이터 로딩 및 전처리 테스트
│   └── test_model.py        # 모델 정의 및 학습 테스트
│
├── .github
│   └── workflows/
│       └── ci-cd.yml         # Git 자동화 설정
│
├── .gitignore          # Git 예외 처리 파일
├── docker-compose.yml        # Docker Container Script
├── requirements.txt         # 패키지 의존성 목록
└── README.md                # 프로젝트 설명