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
                                                     /      \
                                                    /        \
                                          +-------------+   +----------------+
                                          | Frontend UI |   | Docker / CI/CD |
                                          +-------------+   +----------------+

## DIRECTORYT TREE

smart-customer-analysis/
├── data/
│   ├── test.ft.txt
│   └── train.ft.txt
│
├── src/
│   ├── __init__.py
│   ├── run.py
│   ├── data/ 
│   │   ├── __init__.py
│   │   ├── data_loader.py         # 데이터 로딩 및 전처리
│   │   └── dataset.py             # TextDataset 클래스
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py               # 모델 정의 (예: BERT 모델)
│   │   └── train.py               # 모델 학습 코드
│   └── utils/
│       ├── __init__.py
│       └── utils.py               # 데이터 전처리, 평가 관련 유틸리티
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py         # 데이터 로딩 및 전처리 테스트
│   ├── test_model.py        # 모델 정의 및 학습 테스트
│   └── test_train.py        # 모델 학습 및 동작 테스트
│
├── main.py                  # 전체 실행 컨트롤러 (선택사항)
├── requirements.txt         # 패키지 의존성 목록
└── README.md                # 프로젝트 설명