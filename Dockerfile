# 1. 베이스 이미지 선택 (Python 3.10)
FROM python:3.10-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 로컬의 requirements.txt 복사 후 의존성 설치
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. 전체 프로젝트 복사
COPY . .

# 5. uvicorn 실행 (포트는 8000으로 설정)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]