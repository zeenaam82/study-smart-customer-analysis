# SMART-CUSTOMER ARCHITECTURE

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

# DIRECTORYT TREE

smart-customer-analysis/
│
├── main.py                 # FastAPI 서버 실행
├── src/                    # 주요 로직과 API 라우터
│   ├── data_loader.py      # 데이터 로딩 및 전처리
│   ├── recommendation.py  # 추천 알고리즘 구현
│   ├── model.py           # PyTorch 모델 학습 및 저장
│   ├── routers/             # API 엔드포인트 관련 코드
│   │   └── recommendation.py # 추천 시스템 API 엔드포인트
│   └── utils.py           # 유틸리티 함수들
├── models/                 # 학습된 모델 저장
│   └── recommendation_model.pt  # 추천 시스템 모델 (PyTorch)
├── data/                   # 원본 데이터 및 전처리된 데이터
│   ├── train.csv           # 훈련 데이터
│   └── test.csv            # 테스트 데이터
├── tests/                  # 테스트 코드
│   ├── test_data_loader.py # 데이터 로딩 테스트
│   └── test_recommendation.py  # 추천 시스템 테스트
├── requirements.txt        # 필요한 패키지 목록
├── Dockerfile              # Docker 설정 파일
└── README.md               # 프로젝트 설명 파일